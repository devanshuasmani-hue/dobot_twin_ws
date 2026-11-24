"""Guarded core of the physical-robot bridge (ROS-independent, unit-tested against FakeDobot).

READ path (always on): poll pose / alarms / gripper -> twin joint state.  Sends only read requests.
WRITE path (motion) is blocked unless ALL of these hold:
    calibrated  - the frame mapping was fitted from hand-set poses (twin_calibrate)
    armed       - explicit ARM within the last `arm_timeout_s` (refreshed by each accepted command)
    fresh pose  - a valid pose newer than `stale_after_s`
    no alarm    - the robot reports no alarm bit
and the goal passes the SAME checks as the simulated twin (reach, joint limits, floor + extra margin),
including intermediate path samples. STOP is always allowed. Losing the pose stream while armed stops + disarms.
"""
import math
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from dobot_twin_kinematics.kinematics import Magician

from .frame_map import Calibration, DobotFrame
from .link import DobotLink, LinkError
from .protocol import Pose, PTPMode

GRIPPER_OPEN_M = 0.0135


@dataclass
class CoreConfig:
    stale_after_s: float = 0.5
    arm_timeout_s: float = 60.0
    speed_pct: float = 20.0
    accel_pct: float = 20.0
    max_speed_pct: float = 50.0          # hard cap, whatever is requested
    z_margin_m: float = 0.010            # extra clearance above the table for real motion
    path_step_m: float = 0.004
    gripper_inverted: bool = False
    aux_every: int = 10                  # poll alarms/gripper every N pose polls


@dataclass
class Result:
    ok: bool
    msg: str = ''


class RealTwinCore:
    def __init__(self, link: DobotLink, robot: Magician, cal: Calibration, cfg: Optional[CoreConfig] = None,
                 clock=time.monotonic):
        self.link, self.robot, self.cal = link, robot, cal
        self.cfg = cfg or CoreConfig()
        self.clock = clock
        self.frame = DobotFrame(robot, cal)
        lim = replace(robot.limits, floor_z=(robot.limits.floor_z or 0.0) + self.cfg.z_margin_m)
        self.safe = Magician(robot.g, lim)             # stricter envelope used for every real command
        self.pose: Optional[Pose] = None
        self.pose_time = -1e9
        self.alarms: List[int] = []
        self.gripped: Optional[bool] = None
        self.armed_until: Optional[float] = None
        self.speed_pct = min(self.cfg.speed_pct, self.cfg.max_speed_pct)
        self.n_polls = 0
        self.last_error = ''
        self.link_error = False

    # ------------------------------------------------------------------ read path
    def poll(self) -> bool:
        ok = False
        try:
            p = self.link.get_pose()
            self.link_error = False
        except LinkError as e:
            p, self.link_error, self.last_error = None, True, f'link: {e}'
        if p is not None:
            self.pose, self.pose_time, ok = p, self.clock(), True
        self.n_polls += 1
        if ok and self.n_polls % self.cfg.aux_every == 1:
            try:
                a = self.link.get_alarms()
                if a is not None:
                    self.alarms = a
                g = self.link.get_gripper()
                if g is not None and g[0]:
                    self.gripped = g[1]
            except LinkError:
                pass
        if self.armed and not self.fresh:
            self.last_error = 'pose stream lost while armed: stopped'
            self.stop()
        if self.armed_until is not None and self.clock() > self.armed_until:
            self.last_error = 'arm timeout: stopped'
            self.stop()
        return ok

    @property
    def fresh(self) -> bool:
        return self.pose is not None and (self.clock() - self.pose_time) <= self.cfg.stale_after_s

    @property
    def armed(self) -> bool:
        return self.armed_until is not None

    def twin_joint_state(self) -> Optional[Tuple[List[str], List[float]]]:
        """Joint state of the PHYSICAL robot in twin convention, or None when there is no fresh data.
        Deliberately not limit-clamped: the twin must show whatever the robot reports."""
        if not self.fresh:
            return None
        q = self.frame.joints_to_twin(self.pose.joints_deg())
        if self.gripped is None:
            g = GRIPPER_OPEN_M
        else:
            closed = self.gripped != self.cfg.gripper_inverted
            g = 0.0 if closed else GRIPPER_OPEN_M
        names = ['magician_joint_1', 'magician_joint_2', 'magician_joint_3', 'magician_joint_4',
                 'magician_joint_prismatic_l']
        return names, [float(v) for v in q] + [g]

    def tcp_twin(self) -> Optional[np.ndarray]:
        if not self.fresh:
            return None
        return self.robot.fk(self.frame.joints_to_twin(self.pose.joints_deg()))[0]

    def status_text(self) -> str:
        if not self.fresh:
            return 'NO DATA - ' + ('link error (cable/power?)' if self.link_error else 'waiting for robot')
        parts = ['CONNECTED']
        if self.alarms:
            parts.append(f'ALARM {self.alarms}')
        if not self.cal.calibrated:
            parts.append('UNCALIBRATED (read-only)')
        elif self.armed:
            parts.append(f'ARMED {max(0, int(self.armed_until - self.clock()))}s')
        else:
            parts.append('read-only')
        return ' | '.join(parts)

    # ------------------------------------------------------------------ arming
    def arm(self) -> Result:
        if not self.cal.calibrated:
            return self._no('cannot arm: not calibrated (run twin_calibrate)')
        if not self.fresh:
            return self._no('cannot arm: no fresh pose from the robot')
        if self.alarms:
            return self._no(f'cannot arm: robot alarm {self.alarms} (clear it first)')
        try:
            ok = (self.link.queue_stop() and self.link.queue_clear() and
                  self.link.set_speed(self.speed_pct, self.speed_pct) and self.link.queue_start())
        except LinkError as e:
            return self._no(f'cannot arm: {e}')
        if not ok:
            return self._no('cannot arm: robot did not acknowledge setup commands')
        self.armed_until = self.clock() + self.cfg.arm_timeout_s
        return Result(True, 'armed')

    def disarm(self) -> Result:
        return self.stop()

    def stop(self) -> Result:
        """Always allowed. Halts and clears the queue, then disarms."""
        ok = False
        try:
            ok = self.link.queue_stop() and self.link.queue_clear()
        except LinkError:
            pass
        self.armed_until = None
        return Result(ok, 'stopped' if ok else 'stop NOT acknowledged - use the physical button!')

    def set_speed(self, pct: float) -> Result:
        self.speed_pct = max(1.0, min(float(pct), self.cfg.max_speed_pct))
        if self.armed:
            try:
                self.link.set_speed(self.speed_pct, self.speed_pct)
            except LinkError as e:
                return self._no(str(e))
        return Result(True, f'speed {self.speed_pct:.0f} %')

    def clear_alarms(self) -> Result:
        try:
            ok = self.link.clear_alarms()
        except LinkError as e:
            return self._no(str(e))
        self.alarms = []
        return Result(ok, 'alarms cleared' if ok else 'no reply')

    # ------------------------------------------------------------------ guarded commands
    def _no(self, msg: str) -> Result:
        self.last_error = msg
        return Result(False, msg)

    def _gate(self) -> Optional[str]:
        if not self.cal.calibrated:
            return 'rejected: not calibrated'
        if not self.armed:
            return 'rejected: not armed'
        if not self.fresh:
            return 'rejected: no fresh pose from the robot'
        if self.alarms:
            return f'rejected: robot alarm {self.alarms}'
        return None

    def _q_now(self) -> np.ndarray:
        return self.frame.joints_to_twin(self.pose.joints_deg())

    def _joint_path_ok(self, q0, q1, n: int = 25) -> Optional[str]:
        for f in np.linspace(0, 1, n):
            q = q0 + f * (q1 - q0)
            xyz = self.safe.fk(q)[0]
            v = self.safe.limits.violations(q, xyz[2])
            if v:
                return f'joint path at {100 * f:.0f}%: ' + '; '.join(v)
        return None

    def command_joints(self, q_target) -> Result:
        err = self._gate()
        if err:
            return self._no(err)
        q1 = np.asarray(q_target, float)
        v = self.safe.limits.violations(q1, self.safe.fk(q1)[0][2])
        if v:
            return self._no('rejected: ' + '; '.join(v))
        pe = self._joint_path_ok(self._q_now(), q1)
        if pe:
            return self._no('rejected: ' + pe)
        j = self.frame.joints_to_dobot(q1)
        return self._send(PTPMode.MOVJ_ANGLE, *j)

    def home(self) -> Result:
        return self.command_joints(np.zeros(4))

    def command_xyz(self, xyz, line: bool = False) -> Result:
        err = self._gate()
        if err:
            return self._no(err)
        q0 = self._q_now()
        p0 = self.safe.fk(q0)[0]
        p1 = np.asarray(xyz, float)
        r = self.safe.ik(p1, q4=q0[3], seed=q0)
        if not r.ok:
            return self._no('rejected: ' + r.reason)
        if line:
            L = float(np.linalg.norm(p1 - p0))
            n = max(2, int(math.ceil(L / self.cfg.path_step_m)) + 1)
            seed = q0
            for i in range(1, n):
                p = p0 + (p1 - p0) * i / (n - 1)
                ri = self.safe.ik(p, q4=q0[3], seed=seed)
                if not ri.ok:
                    return self._no(f'rejected: line blocked at {100 * i / (n - 1):.0f}%: {ri.reason}')
                seed = ri.q
        else:
            pe = self._joint_path_ok(q0, r.q)
            if pe:
                return self._no('rejected: ' + pe)
        j_new = self.frame.joints_to_dobot(r.q)
        x, y, z = self.frame.xyz_to_dobot(p1)
        return self._send(PTPMode.MOVL_XYZ if line else PTPMode.MOVJ_XYZ, x, y, z, self.frame.r_for(j_new))

    def command_gripper(self, open_: bool) -> Result:
        err = self._gate()
        if err:
            return self._no(err)
        grip = (not open_) != self.cfg.gripper_inverted
        try:
            ok = self.link.gripper(grip)
        except LinkError as e:
            return self._no(str(e))
        if ok:
            self.armed_until = self.clock() + self.cfg.arm_timeout_s
            self.gripped = grip                      # raw 'isGripped' bit as the robot reports it
        return Result(ok, 'gripper command sent' if ok else 'gripper: no reply')

    def pump_off(self) -> Result:
        """Switch the gripper's pump fully OFF. Allowed without ARM (it is the safe state), but a held object will drop."""
        try:
            ok = self.link.pump_off()
        except LinkError as e:
            return self._no(str(e))
        return Result(ok, 'pump off' if ok else 'pump off: no reply')

    def _send(self, mode, a, b, c, d) -> Result:
        try:
            idx = self.link.ptp(mode, float(a), float(b), float(c), float(d))
        except LinkError as e:
            return self._no(str(e))
        if idx is None:
            return self._no('robot did not acknowledge the motion command')
        self.armed_until = self.clock() + self.cfg.arm_timeout_s
        return Result(True, f'sent (queue index {idx})')
