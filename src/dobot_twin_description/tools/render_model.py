#!/usr/bin/env python3
"""Offline renderer: draws the real DAE meshes placed by the URDF kinematics.
Used to *demonstrate/verify* the model without RViz. Not part of the runtime twin.

  python3 render_model.py --pkg ../../dobot_description --out ../docs/images
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import PolyCollection    # noqa: E402
import numpy as np                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dae_loader import load_dae                       # noqa: E402
from mini_xacro import expand_file                    # noqa: E402
from urdf_model import UrdfModel                      # noqa: E402

D = math.radians
LIGHT = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])


def camera(az, el):
    c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r = np.cross([0, 0, 1], c)
    r /= np.linalg.norm(r)
    u = np.cross(c, r)
    return c, r, u


def scene_triangles(m, q, only=None):
    """World-space triangles + colours of every visual mesh for pose q."""
    W = m.fk(q)
    T, C = [], []
    for l in m.links:
        p = m.mesh_path(l)
        if not p or (only and l not in only):
            continue
        d = load_dae(p)
        M = W[l] @ m.visual_T(l)
        t = d['tris'].reshape(-1, 3)
        T.append((np.c_[t, np.ones(len(t))] @ M.T)[:, :3].reshape(-1, 3, 3))
        C.append(d['colors'])
    return np.concatenate(T), np.concatenate(C), W


def draw(ax, m, q, az, el, markers=False, zoom=None, title=None, only=None):
    T, C, W = scene_triangles(m, q, only)
    c, r, u = camera(az, el)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nl == 0, 1, nl)
    facing = n @ c
    n = n * np.sign(facing + 1e-12)[:, None]          # two-sided
    shade = 0.28 + 0.72 * np.clip(n @ LIGHT, 0, 1) * 0.8 + 0.1 * np.abs(n @ c)
    col = np.clip(C * shade[:, None] * 1.25, 0, 1)
    depth = (T.mean(1) @ c)
    order = np.argsort(depth)
    sx, sy = T @ r, T @ u
    polys = np.stack([sx, sy], -1)[order]
    pc = PolyCollection(polys, facecolors=col[order], edgecolors=col[order], linewidths=0.25, antialiaseds=False)
    # floor grid (table plane z=0)
    g = np.arange(-0.3, 0.31, 0.05)
    for v in g:
        for a, b in (((v, -0.3, 0), (v, 0.3, 0)), ((-0.3, v, 0), (0.3, v, 0))):
            a, b = np.array(a), np.array(b)
            ax.plot([a @ r, b @ r], [a @ u, b @ u], color='#c9ced6', lw=0.5, zorder=0)
    ax.add_collection(pc)
    pc.set_zorder(2)
    if markers:
        for jn, j in m.joints.items():
            if j.get('type') != 'revolute' or j.find('mimic') is not None:
                continue
            ch = j.find('child').get('link')
            p = W[ch][:3, 3]
            axis = W[ch][:3, :3] @ np.array([float(x) for x in j.find('axis').get('xyz').split()])
            ax.plot(p @ r, p @ u, 'o', color='#e4572e', ms=5, zorder=5)
            a2 = p + axis * 0.07
            ax.plot([p @ r, a2 @ r], [p @ u, a2 @ u], color='#e4572e', lw=1.6, zorder=5)
            ax.annotate(jn.replace('magician_', ''), (p @ r, p @ u), textcoords='offset points', xytext=(6, 6),
                        fontsize=7, color='#9b2d12', zorder=6)
    # world axes at root
    for vec, colr in (((0.08, 0, 0), '#d62828'), ((0, 0.08, 0), '#2a9d8f'), ((0, 0, 0.08), '#1d4ed8')):
        v = np.array(vec)
        ax.plot([0, v @ r], [0, v @ u], color=colr, lw=1.4, zorder=4)
    ax.set_aspect('equal')
    ext = zoom or 0.36
    ctr = np.array([0.08, 0, 0.12])
    ax.set_xlim(ctr @ r - ext, ctr @ r + ext)
    ax.set_ylim(ctr @ u - ext * 0.9, ctr @ u + ext * 0.9)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = os.path.abspath(a.pkg)
    os.makedirs(a.out, exist_ok=True)
    m = UrdfModel(expand_file(os.path.join(pkg, 'model', 'magician_standalone.urdf.xacro'), pkg,
                              DOF=4, tool='gripper', use_camera='false'), pkg)

    # Figure 1: zero pose, four views, joint pivots/axes overlaid
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), dpi=110)
    views = [('3/4 view', D(35), D(22)), ('side (+x toward right)', D(-90), D(0)),
             ('front', D(0), D(0)), ('top', D(-90), D(89))]
    for ax, (t, az, el) in zip(axs.flat, views):
        draw(ax, m, {}, az, el, markers=True, title=f'zero pose - {t}')
    fig.suptitle('Dobot Magician (4-DOF + gripper) - original DAE meshes via original xacro. '
                 'Red = revolute joint pivots/axes; axes at root: X red, Y green, Z blue', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, '01_zero_pose_views.png'))
    plt.close(fig)

    # Figure 2: pose gallery
    P = 'magician_joint_'
    poses = [
        ('home  (all 0)', {}),
        ('J1 = +60 deg', {P + '1': D(60)}),
        ('J2 = 60, J3 = 10', {P + '2': D(60), P + '3': D(10)}),
        ('J2 = 0, J3 = 60 (folded up)', {P + '2': 0.0, P + '3': D(60)}),
        ('J1 = -45, J2 = 45, J3 = 35, J4 = 90', {P + '1': D(-45), P + '2': D(45), P + '3': D(35), P + '4': D(90)}),
        ('J2 = 50, J3 = -15 (max reach), gripper open', {P + '2': D(50), P + '3': D(-15), 'magician_joint_prismatic_l': 0.0135}),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), dpi=100)
    for ax, (t, q) in zip(axs.flat, poses):
        draw(ax, m, q, D(40), D(20), title=t)
    fig.suptitle('Pose gallery - wrist/tool remain vertical (parallelogram), gripper rotates with J1 + J4', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, '02_pose_gallery.png'))
    plt.close(fig)

    # Figure 3: gripper closed vs open, close-up (end effector only, front-oblique view)
    G = {'magician_link_gripper_core', 'magician_link_gripper_jaw_left', 'magician_link_gripper_jaw_right'}
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=110)
    for ax, (t, s_) in zip(axs, (('gripper closed (prismatic_l = 0)', 0.0), ('gripper open (prismatic_l = 0.0135 m)', 0.0135))):
        q = {'magician_joint_prismatic_l': s_}
        draw(ax, m, q, D(0), D(10), title=t, only=G)
        W = m.fk(q)
        ctr = W['magician_link_gripper_core'][:3, 3] + np.array([0, 0, -0.045])
        c, r, u = camera(D(0), D(10))
        ax.set_xlim(ctr @ r - 0.06, ctr @ r + 0.06)
        ax.set_ylim(ctr @ u - 0.06, ctr @ u + 0.06)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, '03_gripper_open_closed.png'))
    plt.close(fig)
    print('wrote', os.listdir(a.out))


if __name__ == '__main__':
    main()
