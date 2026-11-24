"""Simulated motion controller for the twin (ROS-independent, deterministic, unit-tested).

State: 4 arm joints q=[q1..q4] (rad) + gripper opening g (m per jaw, 0=closed .. 0.0135=open).
Commands (all validated against joint limits / floor BEFORE any motion starts):
  command_joints(q)        joint-space point-to-point
  command_ptp_xyz(xyz)     Cartesian goal, joint-space interpolation (like Dobot "MOVJ")
  command_line_xyz(xyz)    Cartesian goal, straight line in space  (like Dobot "MOVL")
  jog(v_xyz, w_j4)         Cartesian velocity (dead-man: expires if not refreshed)
  command_gripper(open_m)
Motion is time-parameterised with a minimum-jerk profile and respects per-joint velocity and
acceleration limits. tick(dt) advances the simulation.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .kinematics import D2R, Magician, wrap

MINJERK_PEAK_VEL = 1.875     # peak / mean velocity of the min-jerk profile
MINJERK_PEAK_ACC = 5.7735    # peak acc * T^2 / distance


def minjerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


@dataclass
class MotionConfig:
    vmax: Tuple[float, ...] = (90 * D2R, 90 * D2R, 90 * D2R, 120 * D2R)     # rad/s per joint (demo-friendly)
    amax: Tuple[float, ...] = (300 * D2R, 300 * D2R, 300 * D2R, 400 * D2R)  # rad/s^2
    tcp_speed: float = 0.10           # m/s  Cartesian line speed (peak)
    gripper_speed: float = 0.04       # m/s  per jaw
    line_step: float = 0.002          # m    path sampling for straight lines
    jog_timeout: float = 0.12         # s    dead-man: jog stops if no command arrives (senders refresh at >=20 Hz)
    min_duration: float = 0.05


class PathTrajectory:
    """Joint-space polyline traversed with a min-jerk time law."""

    def __init__(self, q_path: np.ndarray, duration: float):
        self.q = np.asarray(q_path, float)
        self.T = max(duration, 1e-6)
        self.t = 0.0

    def sample(self, t: float) -> np.ndarray:
        s = minjerk(t / self.T) * (len(self.q) - 1)
        i = min(int(math.floor(s)), len(self.q) - 2)
        f = s - i
        return self.q[i] * (1 - f) + self.q[i + 1] * f

    @property
    def done(self) -> bool:
        return self.t >= self.T


@dataclass
class Command:
    ok: bool
    msg: str = ''
    duration: float = 0.0


class TwinMotionController:
    def __init__(self, robot: Magician, cfg: Optional[MotionConfig] = None):
        self.robot = robot
        self.cfg = cfg or MotionConfig()
        self.q = np.zeros(4)
        self.g = 0.0
        self.g_target = 0.0
        self.traj: Optional[PathTrajectory] = None
        self.goal_xyz: Optional[np.ndarray] = None
        self.clock = 0.0
        self._jog = None            # (v_xyz, w_j4, stamp)
        self.status = 'IDLE'
        self.last_error = ''

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        return self.traj is not None or self._jog_active()

    def tcp(self) -> Tuple[np.ndarray, float]:
        return self.robot.fk(self.q)

    def joint_state(self):
        names = ['magician_joint_1', 'magician_joint_2', 'magician_joint_3', 'magician_joint_4',
                 'magician_joint_prismatic_l']
        return names, [float(v) for v in self.q] + [float(self.g)]

    def set_state(self, q, g=None):
        """Teleport (used to start from a known pose)."""
        self.q = np.asarray(q, float).copy()
        if g is not None:
            self.g = self.g_target = float(g)
        self.traj, self._jog, self.goal_xyz = None, None, None
        self.status = 'IDLE'

    # --------------------------------------------------------------- durations
    def _duration(self, q_path: np.ndarray, extra: float = 0.0) -> float:
        dq = np.abs(np.diff(q_path, axis=0)).sum(axis=0)           # total travel per joint
        vmax, amax = np.array(self.cfg.vmax), np.array(self.cfg.amax)
        t_v = (dq / vmax).max() * MINJERK_PEAK_VEL
        t_a = math.sqrt((MINJERK_PEAK_ACC * dq / amax).max())
        return max(t_v, t_a, extra, self.cfg.min_duration)

    def _start(self, q_path, extra=0.0) -> Command:
        q_path = np.asarray(q_path, float)
        if np.abs(q_path[-1] - q_path[0]).max() < 1e-9 and len(q_path) == 2:
            self.traj, self.status = None, 'IDLE'
            return Command(True, 'already at target', 0.0)
        T = self._duration(q_path, extra)
        self.traj, self._jog = PathTrajectory(q_path, T), None
        self.status = 'MOVING'
        return Command(True, 'ok', T)

    def _reject(self, msg) -> Command:
        self.last_error = msg
        self.status = 'ERROR: ' + msg
        return Command(False, msg)

    # ---------------------------------------------------------------- commands
    def command_joints(self, q_target) -> Command:
        q_target = np.asarray(q_target, float)
        xyz, _ = self.robot.fk(q_target)
        v = self.robot.limits.violations(q_target, xyz[2])
        if v:
            return self._reject('joint target rejected: ' + '; '.join(v))
        self.goal_xyz = xyz
        return self._start(np.vstack([self.q, q_target]))

    def command_ptp_xyz(self, xyz, yaw: Optional[float] = None) -> Command:
        r = self.robot.ik(xyz, yaw=yaw, seed=self.q)
        if not r.ok:
            return self._reject(r.reason)
        self.goal_xyz = np.asarray(xyz, float)
        return self._start(np.vstack([self.q, r.q]))

    def command_line_xyz(self, xyz, yaw: Optional[float] = None) -> Command:
        p0, yaw0 = self.tcp()
        p1 = np.asarray(xyz, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6 and yaw is None:
            return Command(True, 'already at target', 0.0)
        n = max(2, int(math.ceil(L / self.cfg.line_step)) + 1)
        path, seed = [self.q.copy()], self.q.copy()
        for i in range(1, n):
            f = i / (n - 1)
            p = p0 + f * (p1 - p0)
            yw = None if yaw is None else wrap(yaw0 + f * wrap(yaw - yaw0))
            r = self.robot.ik(p, yaw=yw, q4=None if yw is not None else self.q[3], seed=seed)
            if not r.ok:
                return self._reject(f'line blocked at {100 * f:.0f}% '
                                    f'({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}): {r.reason}')
            path.append(r.q)
            seed = r.q
        self.goal_xyz = p1
        return self._start(np.array(path), extra=L / self.cfg.tcp_speed * MINJERK_PEAK_VEL)

    def home(self) -> Command:
        return self.command_joints(np.zeros(4))

    def command_gripper(self, opening: float) -> Command:
        lo, hi = self.robot.limits.gripper
        self.g_target = min(max(float(opening), lo), hi)
        return Command(True, 'ok')

    def jog(self, v_xyz, w_j4: float = 0.0):
        self._jog = (np.asarray(v_xyz, float), float(w_j4), self.clock)
        self.traj = None

    def stop(self):
        self.traj, self._jog = None, None
        self.status = 'IDLE'

    def _jog_active(self) -> bool:
        return self._jog is not None and (self.clock - self._jog[2]) <= self.cfg.jog_timeout

    # -------------------------------------------------------------------- tick
    def tick(self, dt: float) -> np.ndarray:
        self.clock += dt
        if self._jog_active():
            self._tick_jog(dt)
        elif self.traj is not None:
            self._jog = None
            self.traj.t += dt
            self.q = self.traj.sample(self.traj.t)
            if self.traj.done:
                self.q = self.traj.q[-1].copy()
                self.traj, self.status = None, 'IDLE'
        elif self.status in ('JOG', 'MOVING') or self.status.startswith('LIMIT'):
            self.status = 'IDLE' if not self.status.startswith('LIMIT') else self.status
            if self._jog is not None:
                self._jog = None
                self.status = 'IDLE'
        # gripper
        dg = self.g_target - self.g
        step = self.cfg.gripper_speed * dt
        self.g = self.g_target if abs(dg) <= step else self.g + math.copysign(step, dg)
        return self.q

    def _tick_jog(self, dt: float):
        v, w, _ = self._jog
        p, _ = self.tcp()
        q4 = self.q[3] + w * dt
        r = self.robot.ik(p + v * dt, q4=q4, seed=self.q)
        if not r.ok:
            self.status = 'LIMIT: ' + r.reason
            return
        dq = np.array([wrap(r.q[i] - self.q[i]) for i in range(4)])
        scale = min(1.0, float((np.array(self.cfg.vmax) * dt / np.maximum(np.abs(dq), 1e-12)).min()))
        self.q = self.q + dq * scale
        self.goal_xyz = None
        self.status = 'JOG'
