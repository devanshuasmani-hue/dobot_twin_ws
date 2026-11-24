"""Closed-form forward / inverse kinematics of the Dobot Magician (4-DOF, vertical tool).

ROS-independent. Joint convention = the ORIGINAL URDF:
  q1  base yaw                     (axis +Z)
  q2  rear-arm pitch from vertical (axis +Y, positive leans forward)
  q3  forearm pitch RELATIVE to the rear arm (axis +Y); forearm angle below horizontal = q2 + q3
  q4  end-effector yaw relative to q1
The parallelogram keeps the wrist/tool vertical, so the tool only has yaw (q1 + q4).

Cartesian coordinates: metres in the robot root frame (origin on the J1 axis at table level,
X forward, Y left, Z up). Position is of the TCP = fingertip centre of the standard gripper.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

D2R = math.pi / 180.0


def wrap(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Geometry:
    """Numbers taken from magician.urdf.xacro (link_base_length, link_2/3/4_length) and the gripper mesh."""
    h: float = 0.131305      # shoulder height above table plane
    l2: float = 0.135        # rear arm
    l3: float = 0.147        # forearm
    l4: float = 0.06         # forearm end -> tool axis (horizontal offset)
    tcp_dz: float = 0.1048   # tool-flange -> fingertip, measured from the gripper meshes (verified in tests)


@dataclass
class Limits:
    """Joint limits in radians. j3_rel is the URDF-style relative limit, j3_abs the datasheet-style
    (forearm angle below horizontal = q2 + q3). Either may be None. floor_z keeps the TCP above the table."""
    j1: Tuple[float, float] = (-125 * D2R, 125 * D2R)
    j2: Tuple[float, float] = (0 * D2R, 85 * D2R)
    j3_rel: Optional[Tuple[float, float]] = (-15 * D2R, 70 * D2R)
    j3_abs: Optional[Tuple[float, float]] = (-10 * D2R, 95 * D2R)
    j4: Tuple[float, float] = (-90 * D2R, 90 * D2R)
    gripper: Tuple[float, float] = (0.0, 0.0135)
    floor_z: Optional[float] = 0.0

    @classmethod
    def from_deg_dict(cls, d: dict) -> 'Limits':
        def pair(v):
            return None if v is None else (float(v[0]) * D2R, float(v[1]) * D2R)
        return cls(j1=pair(d['j1']), j2=pair(d['j2']), j3_rel=pair(d.get('j3_rel')), j3_abs=pair(d.get('j3_abs')),
                   j4=pair(d['j4']), gripper=tuple(d.get('gripper', (0.0, 0.0135))), floor_z=d.get('floor_z', 0.0))

    def _excess(self, v, lim):
        return max(lim[0] - v, v - lim[1], 0.0)

    def excess(self, q, z_tcp: Optional[float] = None) -> float:
        """Total amount (rad / m mixed) by which q violates the limits; 0 when feasible."""
        q1, q2, q3, q4 = q
        e = self._excess(q1, self.j1) + self._excess(q2, self.j2) + self._excess(q4, self.j4)
        if self.j3_rel:
            e += self._excess(q3, self.j3_rel)
        if self.j3_abs:
            e += self._excess(q2 + q3, self.j3_abs)
        if self.floor_z is not None and z_tcp is not None:
            e += max(self.floor_z - z_tcp, 0.0)
        return e

    def violations(self, q, z_tcp: Optional[float] = None, tol: float = 1e-9) -> List[str]:
        q1, q2, q3, q4 = q
        out = []

        def chk(name, v, lim, unit='deg'):
            if lim and (v < lim[0] - tol or v > lim[1] + tol):
                f = 1 / D2R
                out.append(f'{name}={v * f:.1f} deg outside [{lim[0] * f:.0f},{lim[1] * f:.0f}]')
        chk('J1', q1, self.j1)
        chk('J2', q2, self.j2)
        chk('J3(rel)', q3, self.j3_rel)
        chk('J3(abs)', q2 + q3, self.j3_abs)
        chk('J4', q4, self.j4)
        if self.floor_z is not None and z_tcp is not None and z_tcp < self.floor_z - tol:
            out.append(f'TCP z={z_tcp * 1000:.0f} mm below floor {self.floor_z * 1000:.0f} mm')
        return out


@dataclass
class IKResult:
    ok: bool
    q: Optional[np.ndarray] = None      # [q1,q2,q3,q4]
    reason: str = ''
    solutions: list = field(default_factory=list)   # all limit-feasible solutions


class Magician:
    def __init__(self, geometry: Geometry = Geometry(), limits: Limits = Limits()):
        self.g = geometry
        self.limits = limits

    # ---------------------------------------------------------------- FK
    def flange_radius_height(self, q):
        g = self.g
        _, q2, q3, _ = q
        r_w = g.l2 * math.sin(q2) + g.l3 * math.cos(q2 + q3)
        z_w = g.h + g.l2 * math.cos(q2) - g.l3 * math.sin(q2 + q3)
        return r_w + g.l4, z_w

    def fk(self, q) -> Tuple[np.ndarray, float]:
        """TCP position [x,y,z] (m) and world yaw (rad) for joints q=[q1,q2,q3,q4]."""
        rf, zf = self.flange_radius_height(q)
        q1 = q[0]
        xyz = np.array([rf * math.cos(q1), rf * math.sin(q1), zf - self.g.tcp_dz])
        return xyz, wrap(q[0] + q[3])

    def fk_points(self, q) -> dict:
        """Key points in the root frame (for plotting / tests)."""
        g = self.g
        q1, q2, q3, _ = q
        c, s = math.cos(q1), math.sin(q1)

        def P(r, z):
            return np.array([r * c, r * s, z])
        r_e, z_e = g.l2 * math.sin(q2), g.h + g.l2 * math.cos(q2)
        r_w, z_w = r_e + g.l3 * math.cos(q2 + q3), z_e - g.l3 * math.sin(q2 + q3)
        return dict(shoulder=P(0, g.h), elbow=P(r_e, z_e), wrist=P(r_w, z_w),
                    flange=P(r_w + g.l4, z_w), tcp=P(r_w + g.l4, z_w - g.tcp_dz))

    # ---------------------------------------------------------------- IK
    def ik(self, xyz, yaw: Optional[float] = None, q4: Optional[float] = None,
           seed=None, check_limits: bool = True) -> IKResult:
        """Inverse kinematics for a TCP position. Tool yaw: give `yaw` (world) or `q4` (joint);
        otherwise q4 is kept from `seed` (or 0). With `seed` the solution closest to it is returned
        (continuity); without, the one with the smallest |q1|."""
        g = self.g
        x, y, z = (float(v) for v in xyz)
        pz = z + g.tcp_dz - g.h
        r = math.hypot(x, y)
        seed = None if seed is None else np.asarray(seed, float)
        if r < 1e-6:
            q1s = [(seed[0] if seed is not None else 0.0, 0.0)]
        else:
            a = math.atan2(y, x)
            q1s = [(a, r), (wrap(a + math.pi), -r)]

        cands, reach_msgs = [], []
        for q1, rf in q1s:
            pr = rf - g.l4
            d2 = pr * pr + pz * pz
            d = math.sqrt(d2)
            if d > g.l2 + g.l3 + 1e-9 or d < abs(g.l2 - g.l3) - 1e-9 or d < 1e-9:
                reach_msgs.append(f'distance {d * 1000:.0f} mm outside arm reach '
                                  f'[{abs(g.l2 - g.l3) * 1000:.0f},{(g.l2 + g.l3) * 1000:.0f}] mm')
                continue
            cg = max(-1.0, min(1.0, (d2 + g.l2 ** 2 - g.l3 ** 2) / (2 * d * g.l2)))
            gam, beta = math.acos(cg), math.atan2(pr, pz)
            for sgn in (+1.0, -1.0):
                a1 = beta + sgn * gam
                a2 = math.atan2(pr - g.l2 * math.sin(a1), pz - g.l2 * math.cos(a1))
                q2, q3 = wrap(a1), wrap(a2 - a1 - math.pi / 2)
                if q4 is not None:
                    q4_ = q4
                elif yaw is not None:
                    q4_ = wrap(yaw - q1)
                else:
                    q4_ = float(seed[3]) if seed is not None else 0.0
                cands.append(np.array([q1, q2, q3, q4_]))

        if not cands:
            return IKResult(False, None, 'unreachable: ' + (reach_msgs[0] if reach_msgs else 'no solution'))
        ok = [c for c in cands if not check_limits or self.limits.excess(c, self.fk(c)[0][2]) <= 1e-9]
        if not ok:
            best = min(cands, key=lambda c: self.limits.excess(c, self.fk(c)[0][2]))
            msg = '; '.join(self.limits.violations(best, self.fk(best)[0][2])) or 'limit violation'
            return IKResult(False, best, 'violates limits: ' + msg)

        def cost(c):
            if seed is not None:
                return sum(abs(wrap(c[i] - seed[i])) for i in range(3))
            return abs(c[0]) * 10 + abs(c[1])
        ok.sort(key=cost)
        return IKResult(True, ok[0], 'ok', ok)

    # ---------------------------------------------------------------- helpers
    def feasible(self, q) -> Tuple[bool, List[str]]:
        v = self.limits.violations(q, self.fk(q)[0][2])
        return (not v), v

    def sample_workspace(self, n: int = 20000, seed: int = 0) -> np.ndarray:
        """Random limit-feasible joint configs -> TCP positions (n_feasible x 3)."""
        rng = np.random.default_rng(seed)
        L = self.limits
        q = np.column_stack([rng.uniform(*L.j1, n), rng.uniform(*L.j2, n),
                             rng.uniform(*(L.j3_rel or (-1.0, 1.7)), n), np.zeros(n)])
        pts = []
        for row in q:
            if self.limits.excess(row, self.fk(row)[0][2]) <= 1e-9:
                pts.append(self.fk(row)[0])
        return np.array(pts)


# ---- convention helpers for the physical robot (Phase 3) --------------------------------------
def j3_urdf_to_dobot(q2: float, q3_rel: float, convention: str = 'relative') -> float:
    """URDF J3 (relative to rear arm) -> value as reported by the robot."""
    return q3_rel if convention == 'relative' else q2 + q3_rel


def j3_dobot_to_urdf(q2: float, j3_reported: float, convention: str = 'relative') -> float:
    return j3_reported if convention == 'relative' else j3_reported - q2
