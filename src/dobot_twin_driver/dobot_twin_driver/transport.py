"""Byte transports for the Dobot link: real serial port, plus port auto-detection.
A transport needs: write(bytes), read(n) -> bytes (may return fewer, may block briefly), in_waiting, close()."""
import glob
from typing import List, Optional


class SerialTransport:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        import serial        # pyserial (apt: python3-serial); imported lazily so tests run without it
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=timeout, write_timeout=0.5)

    def write(self, data: bytes):
        self.ser.write(data)
        self.ser.flush()

    def read(self, n: int = 1) -> bytes:
        return self.ser.read(n)

    @property
    def in_waiting(self) -> int:
        return self.ser.in_waiting

    def close(self):
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


def candidate_ports() -> List[str]:
    return sorted(glob.glob('/dev/ttyUSB*')) + sorted(glob.glob('/dev/ttyACM*'))


def open_serial(port: Optional[str], baud: int = 115200) -> SerialTransport:
    """Open `port`; with an empty port try every USB serial device (the caller verifies by handshake)."""
    if port:
        return SerialTransport(port, baud)
    errs = []
    for p in candidate_ports():
        try:
            return SerialTransport(p, baud)
        except Exception as e:  # noqa: BLE001
            errs.append(f'{p}: {e}')
    raise OSError('no serial device found (' + '; '.join(errs or ['none of /dev/ttyUSB*, /dev/ttyACM*']) + ')')
