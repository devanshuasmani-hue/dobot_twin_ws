"""Request/response link to a Dobot over any transport (real serial or FakeDobot). Thread-safe."""
import threading
import time
from typing import List, Optional

from .protocol import (ID, Frame, FrameParser, Pose, PTPMode, encode, parse_alarms, parse_queue_index, req_alarms,
                       req_clear_alarms, req_device_name, req_get_gripper, req_gripper, req_pose, req_ptp,
                       req_ptp_common, req_queue_clear, req_queue_start, req_queue_stop)


class LinkError(Exception):
    pass


class DobotLink:
    def __init__(self, transport, timeout: float = 0.3):
        self.t = transport
        self.timeout = timeout
        self.clock = time.monotonic        # real time on purpose: I/O timeouts must always expire
        self.parser = FrameParser()
        self.lock = threading.RLock()
        self.errors = 0

    def close(self):
        self.t.close()

    def request(self, frame: Frame, timeout: Optional[float] = None) -> Optional[Frame]:
        """Send and wait for the reply with the same id. Returns None on timeout (never raises on I/O silence)."""
        timeout = self.timeout if timeout is None else timeout
        with self.lock:
            try:
                while self.t.in_waiting:                  # drop stale bytes from earlier timeouts
                    self.t.read(self.t.in_waiting)
                self.parser.buf.clear()                   # synchronous protocol: partial old frames are useless
                self.t.write(encode(frame))
            except Exception as e:  # noqa: BLE001  (unplugged cable etc.)
                self.errors += 1
                raise LinkError(str(e)) from e
            deadline = self.clock() + timeout
            while self.clock() < deadline:
                try:
                    data = self.t.read(self.t.in_waiting or 1)
                except Exception as e:  # noqa: BLE001
                    self.errors += 1
                    raise LinkError(str(e)) from e
                for f in self.parser.feed(data):
                    if f.id == frame.id:
                        return f
            self.errors += 1
            return None

    # ---- reads (always safe) ---------------------------------------------------------------------
    def get_pose(self) -> Optional[Pose]:
        r = self.request(req_pose())
        if r is None:
            return None
        try:
            return Pose.from_params(r.params)
        except ValueError:
            self.errors += 1
            return None

    def get_alarms(self) -> Optional[List[int]]:
        r = self.request(req_alarms())
        return None if r is None else parse_alarms(r.params)

    def get_device_name(self) -> Optional[str]:
        r = self.request(req_device_name())
        return None if r is None else r.params.split(b'\x00')[0].decode(errors='replace')

    def get_gripper(self):
        """(control_enabled, gripped) or None."""
        r = self.request(req_get_gripper())
        return None if r is None or len(r.params) < 2 else (bool(r.params[0]), bool(r.params[1]))

    # ---- writes (only called by the guarded core) --------------------------------------------------
    def clear_alarms(self) -> bool:
        return self.request(req_clear_alarms()) is not None

    def queue_start(self) -> bool:
        return self.request(req_queue_start()) is not None

    def queue_stop(self) -> bool:
        return self.request(req_queue_stop()) is not None

    def queue_clear(self) -> bool:
        return self.request(req_queue_clear()) is not None

    def set_speed(self, velocity_pct: float, accel_pct: float) -> bool:
        return self.request(req_ptp_common(velocity_pct, accel_pct)) is not None

    def ptp(self, mode: int, a: float, b: float, c: float, d: float) -> Optional[int]:
        r = self.request(req_ptp(mode, a, b, c, d))
        return None if r is None else parse_queue_index(r.params)

    def pump_off(self) -> bool:
        return self.request(req_gripper(False, enable=False, queued=False)) is not None

    def gripper(self, grip: bool) -> bool:
        return self.request(req_gripper(grip)) is not None
