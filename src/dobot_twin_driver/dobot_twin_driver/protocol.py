"""Dobot Magician serial protocol (USB virtual COM, 115200 8N1). Pure Python, no I/O.

Frame:  AA AA | len | id | ctrl | params... | checksum
  len      = 2 + len(params)
  ctrl     bit0 = rw (0 read, 1 write), bit1 = isQueued
  checksum = (256 - (id + ctrl + sum(params)) % 256) % 256
e.g. GetPose request: AA AA 02 0A 00 F6
"""
import struct
from dataclasses import dataclass
from typing import List, Optional

HEADER = b'\xAA\xAA'
MAX_PAYLOAD = 128      # real Dobot frames are < 40 bytes; anything larger is a mis-aligned header


class ID:
    DEVICE_SN = 0
    DEVICE_NAME = 1
    DEVICE_VERSION = 2
    POSE = 10
    ALARMS = 20
    GRIPPER = 63
    PTP_COORD_PARAMS = 81
    PTP_COMMON_PARAMS = 83
    PTP_CMD = 84
    QUEUE_START = 240
    QUEUE_STOP = 241
    QUEUE_CLEAR = 245
    QUEUE_INDEX = 246


class PTPMode:
    JUMP_XYZ = 0
    MOVJ_XYZ = 1
    MOVL_XYZ = 2
    JUMP_ANGLE = 3
    MOVJ_ANGLE = 4
    MOVL_ANGLE = 5


@dataclass
class Frame:
    id: int
    ctrl: int = 0
    params: bytes = b''

    @property
    def write(self) -> bool:
        return bool(self.ctrl & 1)

    @property
    def queued(self) -> bool:
        return bool(self.ctrl & 2)


def make_ctrl(write: bool = False, queued: bool = False) -> int:
    return (1 if write else 0) | (2 if queued else 0)


def checksum(payload: bytes) -> int:
    return (256 - (sum(payload) % 256)) % 256


def encode(f: Frame) -> bytes:
    payload = bytes([f.id, f.ctrl]) + bytes(f.params)
    if len(payload) > 255:
        raise ValueError('payload too long')
    return HEADER + bytes([len(payload)]) + payload + bytes([checksum(payload)])


class FrameParser:
    """Incremental, resynchronising parser: feed() raw bytes, get complete valid Frames."""

    def __init__(self):
        self.buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> List[Frame]:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(HEADER)
            if i < 0:
                self.buf = self.buf[-1:] if self.buf[-1:] == b'\xAA' else bytearray()
                break
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 3:
                break
            n = self.buf[2]
            if n < 2 or n > MAX_PAYLOAD:       # impossible length (e.g. AA AA AA ..): skip one byte, resync
                del self.buf[:1]
                self.bad_frames += 1
                continue
            if len(self.buf) < 3 + n + 1:
                break                          # incomplete
            payload = bytes(self.buf[3:3 + n])
            chk = self.buf[3 + n]
            if checksum(payload) != chk:
                del self.buf[:1]               # corrupted: resync after this byte
                self.bad_frames += 1
                continue
            out.append(Frame(payload[0], payload[1], payload[2:]))
            del self.buf[:4 + n]
        return out


# ---- payload helpers ---------------------------------------------------------------------------
@dataclass
class Pose:
    x: float      # mm
    y: float
    z: float
    r: float      # deg  (end-effector rotation as the robot defines it)
    j1: float     # deg
    j2: float
    j3: float
    j4: float

    @classmethod
    def from_params(cls, p: bytes) -> 'Pose':
        if len(p) < 32:
            raise ValueError(f'pose payload too short ({len(p)} bytes)')
        return cls(*struct.unpack_from('<8f', p, 0))

    def to_params(self) -> bytes:
        return struct.pack('<8f', self.x, self.y, self.z, self.r, self.j1, self.j2, self.j3, self.j4)

    def joints_deg(self):
        return (self.j1, self.j2, self.j3, self.j4)


def req_pose() -> Frame:
    return Frame(ID.POSE, make_ctrl(False))


def req_alarms() -> Frame:
    return Frame(ID.ALARMS, make_ctrl(False))


def req_clear_alarms() -> Frame:
    return Frame(ID.ALARMS, make_ctrl(True))


def req_device_name() -> Frame:
    return Frame(ID.DEVICE_NAME, make_ctrl(False))


def req_queue_start() -> Frame:
    return Frame(ID.QUEUE_START, make_ctrl(True))


def req_queue_stop() -> Frame:
    return Frame(ID.QUEUE_STOP, make_ctrl(True))


def req_queue_clear() -> Frame:
    return Frame(ID.QUEUE_CLEAR, make_ctrl(True))


def req_ptp_common(velocity_pct: float, accel_pct: float) -> Frame:
    return Frame(ID.PTP_COMMON_PARAMS, make_ctrl(True, True), struct.pack('<2f', velocity_pct, accel_pct))


def req_ptp(mode: int, a: float, b: float, c: float, d: float) -> Frame:
    """mode PTPMode.*; for *_XYZ: x,y,z[mm],r[deg]; for *_ANGLE: j1..j4 [deg]."""
    return Frame(ID.PTP_CMD, make_ctrl(True, True), bytes([mode]) + struct.pack('<4f', a, b, c, d))


def req_gripper(grip: bool, enable: bool = True, queued: bool = True) -> Frame:
    """enable control + grip(close)/release. enable=False switches the gripper's pump OFF.
    Pump-off is sent NOT queued (immediate), so it also works while the motion queue is stopped."""
    return Frame(ID.GRIPPER, make_ctrl(True, queued), bytes([1 if enable else 0, 1 if grip else 0]))


def req_get_gripper() -> Frame:
    return Frame(ID.GRIPPER, make_ctrl(False))


def parse_alarms(p: bytes) -> List[int]:
    """List of set alarm bit indices (0..127); empty list = no alarm."""
    bits = []
    for i, byte in enumerate(p[:16]):
        for b in range(8):
            if byte & (1 << b):
                bits.append(i * 8 + b)
    return bits


def parse_queue_index(p: bytes) -> Optional[int]:
    return struct.unpack_from('<Q', p, 0)[0] if len(p) >= 8 else None
