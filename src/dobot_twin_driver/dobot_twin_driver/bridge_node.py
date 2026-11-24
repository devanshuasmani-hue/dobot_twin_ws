"""ROS 2 bridge between the PHYSICAL Dobot and the twin. Runs in namespace `real`.

Read side (always on; sends read requests only)
  joint_states   sensor_msgs/JointState        -> robot_state_publisher => the physical twin in RViz
  tool_pose      geometry_msgs/PoseStamped     TCP in the twin frame
  raw_pose       std_msgs/Float64MultiArray    [x,y,z,r,j1,j2,j3,j4] exactly as the robot reports (mm/deg)
  twin_status    std_msgs/String               shown in the RViz label (CONNECTED | ARMED 42s | NO DATA ...)
  cmd_result     std_msgs/String               accept/reject message for every command
Write side (guarded, see core.py)
  cmd/arm (Bool) cmd/speed (Float64 %) cmd/stop (Empty) cmd/clear_alarms (Empty)
  cmd/goal_xyz  cmd/line_xyz (PointStamped, twin frame, m)   cmd/joints (JointState, twin convention)
  cmd/gripper (Float64: >0 open, 0 close)   cmd/home (Empty)
With fake:=true a byte-level simulated robot replaces the serial port (demo / testing without hardware);
topic fake/drag_joints (Float64MultiArray, deg, robot convention) emulates moving the arm by hand.
"""
import math
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64, Float64MultiArray, String

from dobot_twin_kinematics.kinematics import Geometry, Limits, Magician
from dobot_twin_kinematics.params import load_config

from .core import CoreConfig, RealTwinCore
from .fake_dobot import FakeDobot
from .frame_map import DEFAULT_CAL_PATH, Calibration
from .link import DobotLink
from .transport import open_serial


class DobotBridge(Node):
    def __init__(self):
        super().__init__('dobot_bridge')
        d = self.declare_parameter
        d('port', '')
        d('baud', 115200)
        d('fake', False)
        d('fake_calibrated', True)          # fake only: pretend a calibration exists (demo convenience)
        d('poll_hz', 25.0)
        d('calibration_file', DEFAULT_CAL_PATH)
        d('limits_profile', '')
        d('limits_file', '')
        d('frame_id', 'real/magician_root_link')
        d('arm_timeout_s', 60.0)
        d('speed_pct', 20.0)
        d('max_speed_pct', 50.0)
        d('z_margin_m', 0.010)
        d('gripper_inverted', False)
        g = lambda n: self.get_parameter(n).value      # noqa: E731

        cfg = load_config(g('limits_file') or None, g('limits_profile') or None)
        self.robot = Magician(Geometry(tcp_dz=float(cfg.get('tcp_dz', Geometry.tcp_dz))),
                              Limits.from_deg_dict(cfg['profiles'][cfg['profile_name']]))
        self.fake = bool(g('fake'))
        if self.fake and g('fake_calibrated'):
            cal = Calibration('absolute', -12.0, 35.0, 'j1+j4', True, 0.0, 0)
            self.get_logger().warn('FAKE robot: using a built-in demo calibration')
        else:
            cal = Calibration.load(g('calibration_file') or DEFAULT_CAL_PATH)
        if not cal.calibrated:
            self.get_logger().warn('no calibration found: physical twin is READ-ONLY. Run: ros2 run dobot_twin_driver twin_calibrate')
        ccfg = CoreConfig(arm_timeout_s=float(g('arm_timeout_s')), speed_pct=float(g('speed_pct')),
                          max_speed_pct=float(g('max_speed_pct')), z_margin_m=float(g('z_margin_m')),
                          gripper_inverted=bool(g('gripper_inverted')))
        self.cal, self.ccfg = cal, ccfg
        self.lock = threading.RLock()
        self.core = None
        self.frame = g('frame_id')

        self.pub_js = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_pose = self.create_publisher(PoseStamped, 'tool_pose', 10)
        self.pub_raw = self.create_publisher(Float64MultiArray, 'raw_pose', 10)
        self.pub_status = self.create_publisher(String, 'twin_status', 10)
        self.pub_result = self.create_publisher(String, 'cmd_result', 10)

        self.create_subscription(Bool, 'cmd/arm', self._arm, 10)
        self.create_subscription(Float64, 'cmd/speed', lambda m: self._res(self._with(lambda c: c.set_speed(m.data))), 10)
        self.create_subscription(Empty, 'cmd/stop', lambda _m: self._res(self._with(lambda c: c.stop())), 10)
        self.create_subscription(Empty, 'cmd/clear_alarms', lambda _m: self._res(self._with(lambda c: c.clear_alarms())), 10)
        self.create_subscription(Empty, 'cmd/home', lambda _m: self._res(self._with(lambda c: c.home())), 10)
        self.create_subscription(PointStamped, 'cmd/goal_xyz', lambda m: self._xyz(m, False), 10)
        self.create_subscription(PointStamped, 'cmd/line_xyz', lambda m: self._xyz(m, True), 10)
        self.create_subscription(JointState, 'cmd/joints', self._joints, 10)
        self.create_subscription(Float64, 'cmd/gripper', lambda m: self._res(self._with(lambda c: c.command_gripper(m.data > 0.0))), 10)
        self.create_subscription(Empty, 'cmd/pump_off', lambda _m: self._res(self._with(lambda c: c.pump_off())), 10)
        if self.fake:
            self.create_subscription(Float64MultiArray, 'fake/drag_joints', self._drag, 10)

        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ helpers
    def _with(self, fn):
        with self.lock:
            if self.core is None:
                from .core import Result
                return Result(False, 'rejected: no connection to the robot')
            return fn(self.core)

    def _res(self, r):
        msg = r.msg.removeprefix('rejected: ')
        (self.get_logger().info if r.ok else self.get_logger().warn)(msg)
        self.pub_result.publish(String(data=('OK: ' if r.ok else 'REJECTED: ') + msg))

    def _arm(self, m: Bool):
        self._res(self._with(lambda c: c.arm() if m.data else c.disarm()))

    def _xyz(self, m: PointStamped, line: bool):
        p = [m.point.x, m.point.y, m.point.z]
        self._res(self._with(lambda c: c.command_xyz(p, line)))

    def _joints(self, m: JointState):
        def go(c):
            import numpy as np
            names, cur = c.twin_joint_state() or (None, None)
            if names is None:
                from .core import Result
                return Result(False, 'rejected: no fresh pose')
            t = dict(zip(names, cur))
            t.update(dict(zip(m.name, m.position)))
            return c.command_joints(np.array([t[n] for n in names[:4]]))
        self._res(self._with(go))

    def _drag(self, m: Float64MultiArray):
        if self.fake and getattr(self, '_fake_dev', None) is not None:
            self._fake_dev.drag_to(list(m.data)[:4])

    # ------------------------------------------------------------------ connection + loop
    def _connect(self):
        if self.fake:
            dev = FakeDobot(self.robot)
            self._fake_dev = dev
            transport = dev
        else:
            transport = open_serial(self.get_parameter('port').value or None, int(self.get_parameter('baud').value))
        link = DobotLink(transport, timeout=0.3)
        name = link.get_device_name()
        if link.get_pose() is None:
            link.close()
            raise OSError('serial port opened but the robot does not answer GetPose (wrong port? robot off?)')
        self.get_logger().info(f'connected to Dobot{" (" + name + ")" if name else ""}')
        return link

    def _loop(self):
        period = 1.0 / float(self.get_parameter('poll_hz').value)
        last_status, link, retry_at, lost_since = 0.0, None, 0.0, None
        while not self._stop and rclpy.ok():
            t0 = time.monotonic()
            if self.core is None:
                if t0 >= retry_at:
                    try:
                        link = self._connect()
                        with self.lock:
                            self.core = RealTwinCore(link, self.robot, self.cal, self.ccfg)
                    except Exception as e:  # noqa: BLE001
                        self.get_logger().warn(f'connect failed: {e} (retrying)', throttle_duration_sec=5.0)
                        retry_at = t0 + 2.0
                        self.pub_status.publish(String(data='NO DATA - cannot connect to the robot'))
                time.sleep(0.1)
                continue
            with self.lock:
                self.core.poll()
                st = self.core.twin_joint_state()
                tcp = self.core.tcp_twin()
                status = self.core.status_text()
                pose = self.core.pose
                fresh = self.core.fresh
            stamp = self.get_clock().now().to_msg()
            if st is not None:
                js = JointState()
                js.header.stamp = stamp
                js.name, js.position = st
                self.pub_js.publish(js)
                ps = PoseStamped()
                ps.header.stamp, ps.header.frame_id = stamp, self.frame
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in tcp)
                ps.pose.orientation.w = 1.0
                self.pub_pose.publish(ps)
                self.pub_raw.publish(Float64MultiArray(data=[pose.x, pose.y, pose.z, pose.r, pose.j1, pose.j2, pose.j3, pose.j4]))
            if t0 - last_status > 0.5:
                last_status = t0
                self.pub_status.publish(String(data=status))
            if fresh:
                lost_since = None
            elif lost_since is None:
                lost_since = t0
            elif t0 - lost_since > 3.0:
                # cable unplugged / robot power-cycled: drop the dead connection and reconnect (stays DISARMED)
                self.get_logger().warn('no data from the robot for 3 s: dropping the connection and reconnecting')
                with self.lock:
                    self.core = None
                try:
                    link.close()
                except Exception:  # noqa: BLE001
                    pass
                lost_since, retry_at = None, t0 + 1.0
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def shutdown(self):
        self._stop = True
        try:
            with self.lock:
                if self.core is not None and self.core.armed:
                    self.core.stop()
        except Exception:  # noqa: BLE001
            pass


def main():
    rclpy.init()
    node = DobotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
