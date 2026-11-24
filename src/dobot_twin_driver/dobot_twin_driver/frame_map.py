"""Mapping between the physical Dobot's reported values and the twin's frame, and a calibration
that is FITTED FROM HAND-POSITIONED POSES (no motion commands needed).

Unknowns that cannot be known without the robot, so they are estimated:
  * j3_convention : is the robot's J3 relative to the rear arm (URDF style) or the forearm angle
                    below horizontal (= J2 + J3)?
  * dr_mm, dz_mm  : the robot's Cartesian reading = twin TCP + (radial offset, height offset), because the
                    robot's reported point depends on its configured end-effector / z origin.
  * r_mode        : whether the robot's `r` equals J4 or J1 + J4.
"""
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

from dobot_twin_kinematics.kinematics import D2R, Magician, j3_dobot_to_urdf, j3_urdf_to_dobot, wrap

DEFAULT_CAL_PATH = os.path.expanduser('~/.dobot_twin/calibration.yaml')


@dataclass
class Calibration:
    j3_convention: str = 'absolute'     # 'relative' | 'absolute'  (unverified default until calibrated)
    dr_mm: float = 0.0
    dz_mm: float = 0.0
    r_mode: str = 'j1+j4'               # 'j4' | 'j1+j4'
    calibrated: bool = False
    residual_mm: Optional[float] = None
    samples: int = 0

    def save(self, path: str = DEFAULT_CAL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def load(cls, path: str = DEFAULT_CAL_PATH) -> 'Calibration':
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DobotFrame:
    """Converts between robot units (mm, deg, robot J3 convention) and twin units (m, rad, URDF)."""

    def __init__(self, robot: Magician, cal: Calibration):
        self.robot, self.cal = robot, cal

    # joints -----------------------------------------------------------------------------------
    def joints_to_twin(self, j_deg: Sequence[float]) -> np.ndarray:
        q1, q2, j3, q4 = (math.radians(v) for v in j_deg)
        return np.array([q1, q2, j3_dobot_to_urdf(q2, j3, self.cal.j3_convention), q4])

    def joints_to_dobot(self, q: Sequence[float]) -> np.ndarray:
        q1, q2, q3, q4 = q
        return np.degrees([q1, q2, j3_urdf_to_dobot(q2, q3, self.cal.j3_convention), q4])

    # cartesian --------------------------------------------------------------------------------
    def xyz_to_dobot(self, xyz_m: Sequence[float]) -> np.ndarray:
        x, y, z = xyz_m
        r = math.hypot(x, y)
        if r < 1e-6:
            raise ValueError('target on the base axis')
        rd = r * 1000.0 + self.cal.dr_mm
        return np.array([x / r * rd, y / r * rd, z * 1000.0 + self.cal.dz_mm])

    def xyz_to_twin(self, xyz_mm: Sequence[float]) -> np.ndarray:
        x, y, z = xyz_mm
        rd = math.hypot(x, y)
        if rd < 1e-6:
            raise ValueError('reading on the base axis')
        r = (rd - self.cal.dr_mm) / 1000.0
        return np.array([x / rd * r, y / rd * r, (z - self.cal.dz_mm) / 1000.0])

    def r_for(self, j_deg: Sequence[float]) -> float:
        return j_deg[3] + (j_deg[0] if self.cal.r_mode == 'j1+j4' else 0.0)

    def tcp_from_joints(self, j_deg: Sequence[float]) -> np.ndarray:
        """TCP in the twin frame computed from the robot's joint readings."""
        return self.robot.fk(self.joints_to_twin(j_deg))[0]


# --------------------------------------------------------------------------------------------
@dataclass
class FitReport:
    ok: bool
    message: str
    cal: Optional[Calibration] = None
    details: Optional[dict] = None


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def fit_calibration(samples: List[Tuple[Sequence[float], Sequence[float], float]], robot: Magician,
                    min_samples: int = 4, min_j2_spread_deg: float = 15.0, accept_mm: float = 3.0,
                    ambiguity_mm: float = 0.5) -> FitReport:
    """samples: [(joints_deg[4], xyz_mm[3], r_deg), ...] read from the robot at hand-set poses."""
    if len(samples) < min_samples:
        return FitReport(False, f'need at least {min_samples} poses, got {len(samples)}')
    j2 = [s[0][1] for s in samples]
    j3 = [s[0][2] for s in samples]
    if max(j2) - min(j2) < min_j2_spread_deg:
        return FitReport(False, f'poses too similar: move the rear arm (J2) through at least {min_j2_spread_deg:.0f} deg '
                                f'(spread {max(j2) - min(j2):.1f}); the J3 convention cannot be told otherwise')
    if max(j3) - min(j3) < 5.0:
        return FitReport(False, 'poses too similar: also change the forearm (J3)')
    res = {}
    for conv in ('relative', 'absolute'):
        cal = Calibration(j3_convention=conv)
        fr = DobotFrame(robot, cal)
        dr, dz, perp = [], [], []
        for j, xyz, _r in samples:
            q = fr.joints_to_twin(j)
            tcp = robot.fk(q)[0] * 1000.0
            q1 = q[0]
            rf = tcp[0] * math.cos(q1) + tcp[1] * math.sin(q1)
            rd_proj = xyz[0] * math.cos(q1) + xyz[1] * math.sin(q1)
            dr.append(rd_proj - rf)
            dz.append(xyz[2] - tcp[2])
            perp.append(-xyz[0] * math.sin(q1) + xyz[1] * math.cos(q1))
        dr, dz, perp = np.array(dr), np.array(dz), np.array(perp)
        err = math.sqrt(dr.var() + dz.var() + float(np.mean(perp ** 2)))
        res[conv] = dict(dr=float(dr.mean()), dz=float(dz.mean()), err=err, dr_std=float(dr.std()),
                         dz_std=float(dz.std()), perp=float(np.abs(perp).max()))
    best = min(res, key=lambda c: res[c]['err'])
    other = 'absolute' if best == 'relative' else 'relative'
    d = res[best]
    details = dict(per_convention=res)
    if res[other]['err'] - d['err'] < ambiguity_mm:
        return FitReport(False, f'ambiguous: both J3 conventions fit within {ambiguity_mm} mm '
                                f'(relative {res["relative"]["err"]:.2f}, absolute {res["absolute"]["err"]:.2f} mm). '
                                'Use poses with J2 well away from 0 and different J3.', None, details)
    if d['err'] > accept_mm:
        return FitReport(False, f'model mismatch: best fit ({best}) still has {d["err"]:.1f} mm error '
                                f'(> {accept_mm} mm). Not trusting it; motion stays disabled.', None, details)
    # r mode
    e_j4 = np.mean([abs(_wrap_deg(s[2] - s[0][3])) for s in samples])
    e_j14 = np.mean([abs(_wrap_deg(s[2] - (s[0][0] + s[0][3]))) for s in samples])
    r_mode = 'j4' if e_j4 <= e_j14 else 'j1+j4'
    details['r_mode_err_deg'] = dict(j4=float(e_j4), j1_plus_j4=float(e_j14))
    cal = Calibration(j3_convention=best, dr_mm=d['dr'], dz_mm=d['dz'], r_mode=r_mode, calibrated=True,
                      residual_mm=d['err'], samples=len(samples))
    return FitReport(True, f'J3 convention = {best}; radial offset {d["dr"]:.1f} mm, height offset {d["dz"]:.1f} mm; '
                           f'fit error {d["err"]:.2f} mm; r = {r_mode}', cal, details)
