"""Tiny URDF kinematic model (FK, mimic, mesh placement) for verification/rendering.
Independent of ROS. Mimic joints are resolved the way robot_state_publisher does it
(position = multiplier * parent + offset, no limit clamping)."""
import math
import os
import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_T(el):
    M = np.eye(4)
    if el is not None:
        M[:3, :3] = rpy_matrix(*[float(v) for v in el.get('rpy', '0 0 0').split()])
        M[:3, 3] = [float(v) for v in el.get('xyz', '0 0 0').split()]
    return M


def axis_angle_T(axis, a):
    ax = np.array(axis, float)
    n = np.linalg.norm(ax)
    M = np.eye(4)
    if n == 0:
        return M
    ax /= n
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    M[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return M


class UrdfModel:
    def __init__(self, root, pkg_root):
        self.root, self.pkg_root = root, pkg_root
        self.links = {l.get('name'): l for l in root.findall('link')}
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.children = {}
        for j in self.joints.values():
            self.children.setdefault(j.find('parent').get('link'), []).append(j)
        child_set = {j.find('child').get('link') for j in self.joints.values()}
        roots = [n for n in self.links if n not in child_set]
        self.root_link = roots[0] if len(roots) == 1 else None
        self.roots = roots

    # -- queries -------------------------------------------------------
    def actuated(self):
        """Joints a user/driver commands (revolute/prismatic without mimic)."""
        return [n for n, j in self.joints.items()
                if j.get('type') in ('revolute', 'prismatic', 'continuous') and j.find('mimic') is None]

    def mimics(self):
        return {n: j.find('mimic') for n, j in self.joints.items() if j.find('mimic') is not None}

    def limits(self, name):
        l = self.joints[name].find('limit')
        return (float(l.get('lower')), float(l.get('upper'))) if l is not None else (None, None)

    def mesh_path(self, link_name):
        v = self.links[link_name].find('visual')
        if v is None or v.find('geometry/mesh') is None:
            return None
        f = v.find('geometry/mesh').get('filename')
        return os.path.join(self.pkg_root, f.split('dobot_description/', 1)[1])

    # -- kinematics ----------------------------------------------------
    def resolve(self, q):
        q = dict(q)
        for n, m in self.mimics().items():
            q[n] = float(m.get('multiplier', 1)) * q.get(m.get('joint'), 0.0) + float(m.get('offset', 0))
        return q

    def fk(self, q):
        q = self.resolve(q)
        W = {self.root_link: np.eye(4)}

        def rec(n):
            for j in self.children.get(n, []):
                M = W[n] @ origin_T(j.find('origin'))
                t, v = j.get('type'), q.get(j.get('name'), 0.0)
                ax = [float(x) for x in j.find('axis').get('xyz').split()] if j.find('axis') is not None else [0, 0, 1]
                if t in ('revolute', 'continuous'):
                    M = M @ axis_angle_T(ax, v)
                elif t == 'prismatic':
                    P = np.eye(4)
                    P[:3, 3] = np.array(ax) / (np.linalg.norm(ax) or 1) * v
                    M = M @ P
                c = j.find('child').get('link')
                W[c] = M
                rec(c)

        rec(self.root_link)
        return W

    def visual_T(self, link_name):
        v = self.links[link_name].find('visual')
        return origin_T(v.find('origin')) if v is not None else np.eye(4)
