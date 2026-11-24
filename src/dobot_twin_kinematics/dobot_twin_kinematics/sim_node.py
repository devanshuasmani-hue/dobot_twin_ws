"""ROS 2 wrapper around TwinMotionController: the SIMULATED twin's joint-state source.

Run inside namespace `sim` (the launch file does this) so every topic below is /sim/...

  Subscribes (commands)
    cmd/goal_xyz   geometry_msgs/PointStamped   Cartesian goal, joint-space move (PTP)
    cmd/line_xyz   geometry_msgs/PointStamped   Cartesian goal, straight line
    cmd/jog        geometry_msgs/Twist          linear.xyz [m/s], angular.z = J4 rate [rad/s]  (dead-man)
    cmd/joints     sensor_msgs/JointState       joint-space goal (names magician_joint_1..4 / _prismatic_l)
    cmd/gripper    std_msgs/Float64             opening per jaw [m], 0 closed .. 0.0135 open
    cmd/home       std_msgs/Empty
    cmd/stop       std_msgs/Empty
    cmd/sync_joints sensor_msgs/JointState     teleport the simulated twin to this joint state
  Publishes
    joint_states   sensor_msgs/JointState       -> robot_state_publisher (/sim/joint_states)
    tool_pose      geometry_msgs/PoseStamped    TCP in <frame_id>
    status         std_msgs/String              IDLE / MOVING / JOG / LIMIT:.. / ERROR:..
    markers        visualization_msgs/MarkerArray  TCP + goal markers for RViz
Cartesian coordinates are metres in the simulated robot's root frame (see kinematics.py).
"""
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from .kinematics import Geometry, Limits, Magician
from .motion import MotionConfig, TwinMotionController
from .params import load_config


class TwinSimController(Node):
    def __init__(self):
        super().__init__('twin_sim_controller')
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('limits_file', '')
        self.declare_parameter('limits_profile', '')
        self.declare_parameter('frame_id', 'sim/magician_root_link')
        self.declare_parameter('tcp_speed', 0.10)
        self.declare_parameter('joint_speed_deg', 90.0)

        cfg = load_config(self.get_parameter('limits_file').value or None,
                          self.get_parameter('limits_profile').value or None)
        robot = Magician(Geometry(tcp_dz=float(cfg.get('tcp_dz', Geometry.tcp_dz))),
                         Limits.from_deg_dict(cfg['profiles'][cfg['profile_name']]))
        mc = MotionConfig(tcp_speed=float(self.get_parameter('tcp_speed').value))
        v = math.radians(float(self.get_parameter('joint_speed_deg').value))
        mc.vmax = (v, v, v, v * 4 / 3)
        self.ctrl = TwinMotionController(robot, mc)
        self.frame = self.get_parameter('frame_id').value
        self.get_logger().info(f"SIMULATION controller ready, limits profile '{cfg['profile_name']}'")

        self.pub_js = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_pose = self.create_publisher(PoseStamped, 'tool_pose', 10)
        self.pub_status = self.create_publisher(String, 'status', 10)
        self.pub_marker = self.create_publisher(MarkerArray, 'markers', 1)

        self.create_subscription(PointStamped, 'cmd/goal_xyz', lambda m: self._xyz(m, False), 10)
        self.create_subscription(PointStamped, 'cmd/line_xyz', lambda m: self._xyz(m, True), 10)
        self.create_subscription(Twist, 'cmd/jog', self._jog, 10)
        self.create_subscription(JointState, 'cmd/joints', self._joints, 10)
        self.create_subscription(Float64, 'cmd/gripper', lambda m: self.ctrl.command_gripper(m.data), 10)
        self.create_subscription(Empty, 'cmd/home', lambda _m: self._report(self.ctrl.home()), 10)
        self.create_subscription(Empty, 'cmd/stop', lambda _m: self.ctrl.stop(), 10)
        self.create_subscription(JointState, 'cmd/sync_joints', self._sync, 10)

        self._last = self.get_clock().now()
        self._last_status = ''
        self.create_timer(1.0 / float(self.get_parameter('rate').value), self._tick)

    # ------------------------------------------------------------- callbacks
    def _report(self, cmd):
        if not cmd.ok:
            self.get_logger().warn('command rejected: ' + cmd.msg)

    def _xyz(self, msg: PointStamped, line: bool):
        p = [msg.point.x, msg.point.y, msg.point.z]
        self._report(self.ctrl.command_line_xyz(p) if line else self.ctrl.command_ptp_xyz(p))

    def _jog(self, msg: Twist):
        self.ctrl.jog([msg.linear.x, msg.linear.y, msg.linear.z], msg.angular.z)

    def _joints(self, msg: JointState):
        names, cur = self.ctrl.joint_state()
        target = dict(zip(names, cur))
        for n, p in zip(msg.name, msg.position):
            if n in target:
                target[n] = p
        self._report(self.ctrl.command_joints([target[n] for n in names[:4]]))
        self.ctrl.command_gripper(target['magician_joint_prismatic_l'])

    def _sync(self, msg: JointState):
        """Teleport the SIMULATED twin to the given joint state (e.g. copy the physical robot's pose)."""
        names, cur = self.ctrl.joint_state()
        t = dict(zip(names, cur))
        for n, p in zip(msg.name, msg.position):
            if n in t:
                t[n] = p
        self.ctrl.set_state([t[n] for n in names[:4]], t['magician_joint_prismatic_l'])

    # ------------------------------------------------------------------ tick
    def _tick(self):
        now = self.get_clock().now()
        dt = min((now - self._last).nanoseconds * 1e-9, 0.1)
        self._last = now
        self.ctrl.tick(dt)
        stamp = now.to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name, js.position = self.ctrl.joint_state()
        self.pub_js.publish(js)

        xyz, yaw = self.ctrl.tcp()
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, self.frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in xyz)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pub_pose.publish(ps)

        if self.ctrl.status != self._last_status or int(self.ctrl.clock * 5) != getattr(self, '_k', None):
            self._k = int(self.ctrl.clock * 5)
            self._last_status = self.ctrl.status
            self.pub_status.publish(String(data=self.ctrl.status))
        self._markers(stamp, xyz)

    def _markers(self, stamp, xyz):
        arr = MarkerArray()

        def sphere(i, p, rgb, size):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self.frame
            m.ns, m.id, m.type, m.action = 'sim_tcp', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in p)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 0.9)
            return m
        arr.markers.append(sphere(0, xyz, (0.2, 0.55, 1.0), 0.012))
        goal = self.ctrl.goal_xyz
        g = sphere(1, goal if goal is not None else xyz, (0.1, 0.9, 0.3), 0.016)
        g.action = Marker.ADD if goal is not None and self.ctrl.busy else Marker.DELETE
        arr.markers.append(g)
        self.pub_marker.publish(arr)


def main():
    rclpy.init()
    node = TwinSimController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
