"""A byte-level simulated Dobot Magician (transport-compatible). Lets the whole stack - protocol, link,
bridge, calibration, safety gating - be tested and demonstrated WITHOUT hardware.

It deliberately hides a "true" convention/offset (J3 absolute, radial/height offsets) that the driver
must discover through calibration, like with the real robot. Motion is joint-space interpolation at a
speed set by the PTP velocity ratio (a stand-in for the real controller, not a dynamics model).
"""
import math
import struct
import time
from typing import Callable, List, Optional

import numpy as np

from dobot_twin_kinematics.kinematics import Magician

from .frame_map import Calibration, DobotFrame
from .protocol import ID, Frame, FrameParser, PTPMode, Pose, encode, make_ctrl

MAX_JOINT_SPEED_DEG = 90.0      # deg/s at 100 % ratio (illustrative)


class FakeDobot:
    def __init__(self, robot: Magician, truth: Optional[Calibration] = None,
                 time_fn: Callable[[], float] = time.monotonic, alarms: Optional[List[int]] = None):
        self.robot = robot
        self.truth = truth or Calibration(j3_convention='absolute', dr_mm=-12.0, dz_mm=35.0, r_mode='j1+j4', calibrated=True)
        self.frame = DobotFrame(robot, self.truth)
        self.time = time_fn
        self.j = np.zeros(4)                  # current joints, deg, robot convention
        self.alarm_bits = list(alarms or [])
        self.queue_running = False
        self.queue: List[tuple] = []          # pending (target_joints_deg,)
        self.active = None                    # (j0, j1, t0, duration)
        self.vel_ratio = 20.0
        self.gripped = False
        self.pump_on = True
        self.index = 0
        self.parser = FrameParser()
        self.out = bytearray()
        self.last_t = self.time()
        self.connected = True
        self.log: List[str] = []              # protocol-level log of commands received (for tests)

    # -- test helper: the user drags the arm by hand ----------------------------------------------
    def drag_to(self, j_deg):
        self.j = np.array(j_deg, float)
        self.active = None

    # -- transport interface ----------------------------------------------------------------------
    @property
    def in_waiting(self) -> int:
        return len(self.out)

    def read(self, n: int = 1) -> bytes:
        self._advance()
        if not self.connected:
            return b''
        k = min(n, len(self.out))
        data, self.out = bytes(self.out[:k]), self.out[k:]
        return data

    def close(self):
        pass

    def write(self, data: bytes):
        self._advance()
        if not self.connected:
            return
        for f in self.parser.feed(data):
            self._handle(f)

    # -- simulation -------------------------------------------------------------------------------
    def _advance(self):
        now = self.time()
        dt = now - self.last_t
        self.last_t = now
        while dt > 0 or (self.active is None and self.queue and self.queue_running):
            if self.active is None:
                if self.queue_running and self.queue:
                    tgt = self.queue.pop(0)[0]
                    dmax = float(np.abs(tgt - self.j).max())
                    dur = dmax / (MAX_JOINT_SPEED_DEG * self.vel_ratio / 100.0) if dmax > 1e-9 else 0.0
                    self.active = [self.j.copy(), tgt, 0.0, dur]
                    continue
                break
            j0, j1, t, dur = self.active
            step = min(dt, dur - t)
            t += step
            dt -= step
            self.active[2] = t
            f = 1.0 if dur <= 0 else t / dur
            self.j = j0 + (j1 - j0) * f
            if t >= dur - 1e-12:
                self.j = j1.copy()
                self.active = None
            else:
                break

    def pose(self) -> Pose:
        # the robot reports x,y,z from its joint-based TCP incl. its own offsets (like a real controller)
        q = self.frame.joints_to_twin(self.j)
        tcp = self.robot.fk(q)[0]
        r_t = tcp[0] * math.cos(q[0]) + tcp[1] * math.sin(q[0])
        rd = r_t * 1000.0 + self.truth.dr_mm
        x, y = rd * math.cos(q[0]), rd * math.sin(q[0])
        z = tcp[2] * 1000.0 + self.truth.dz_mm
        return Pose(x, y, z, self.frame.r_for(self.j), *self.j)

    # -- protocol handler -------------------------------------------------------------------------
    def _reply(self, f: Frame, params: bytes = b''):
        self.out += encode(Frame(f.id, f.ctrl, params))

    def _handle(self, f: Frame):
        self.log.append(f'id={f.id} ctrl={f.ctrl} len={len(f.params)}')
        if f.id == ID.POSE and not f.write:
            self._reply(f, self.pose().to_params())
        elif f.id == ID.ALARMS:
            if f.write:
                self.alarm_bits = []
                self._reply(f)
            else:
                b = bytearray(16)
                for k in self.alarm_bits:
                    b[k // 8] |= 1 << (k % 8)
                self._reply(f, bytes(b))
        elif f.id == ID.DEVICE_NAME and not f.write:
            self._reply(f, b'Magician-SIM\x00')
        elif f.id == ID.QUEUE_START:
            self.queue_running = True
            self._reply(f)
        elif f.id == ID.QUEUE_STOP:
            self.queue_running = False
            self._reply(f)
        elif f.id == ID.QUEUE_CLEAR:
            self.queue.clear()
            self.active = None
            self._reply(f)
        elif f.id == ID.PTP_COMMON_PARAMS and f.write:
            self.vel_ratio = max(1.0, min(100.0, struct.unpack_from('<2f', f.params, 0)[0]))
            self._queued_reply(f)
        elif f.id == ID.PTP_CMD and f.write:
            mode = f.params[0]
            a, b, c, d = struct.unpack_from('<4f', f.params, 1)
            if mode in (PTPMode.MOVJ_ANGLE, PTPMode.MOVL_ANGLE, PTPMode.JUMP_ANGLE):
                tgt = np.array([a, b, c, d])
            else:       # *_XYZ in the robot frame: IK through the hidden truth model
                twin_xyz = self.frame.xyz_to_twin([a, b, c])
                q1 = math.atan2(b, a)
                q4 = math.radians(d) - (q1 if self.truth.r_mode == 'j1+j4' else 0.0)
                r = self.robot.ik(twin_xyz, q4=q4, seed=self.frame.joints_to_twin(self.j), check_limits=False)
                if not r.ok and r.q is None:
                    self._queued_reply(f)
                    return
                tgt = self.frame.joints_to_dobot(r.q)
            self.queue.append((tgt,))
            self._queued_reply(f)
        elif f.id == ID.GRIPPER:
            if f.write:
                self.pump_on = bool(f.params[0]) if f.params else self.pump_on
                self.gripped = bool(f.params[1]) if len(f.params) > 1 else self.gripped
                self._queued_reply(f)
            else:
                self._reply(f, bytes([1, 1 if self.gripped else 0]))
        elif f.id == ID.QUEUE_INDEX:
            self._reply(f, struct.pack('<Q', self.index))
        else:
            self._reply(f)

    def _queued_reply(self, f: Frame):
        if f.queued:
            self.index += 1
            self._reply(f, struct.pack('<Q', self.index))
        else:
            self._reply(f)
