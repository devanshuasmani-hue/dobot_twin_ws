#!/usr/bin/env python3
"""Phase 2 verification + figures (no ROS needed).
  python3 phase2_demo.py --out ../docs
Writes docs/PHASE2_VERIFICATION.md and docs/images/phase2_*.png
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(SRC, 'dobot_twin_description', 'tools'))
from dobot_twin_kinematics.kinematics import D2R, wrap          # noqa: E402
from dobot_twin_kinematics.motion import TwinMotionController   # noqa: E402
from dobot_twin_kinematics.params import load_magician          # noqa: E402
from mini_xacro import expand_file                              # noqa: E402
from urdf_model import UrdfModel                                # noqa: E402
import render_model as rm                                       # noqa: E402

PKG = os.path.join(SRC, 'dobot_description')
rows = []


def rec(status, name, detail):
    rows.append((status, name, detail))
    print(f'[{status}] {name} -- {detail}')


def accuracy(robot):
    rng, n, e_pos, e_q, e_yaw = np.random.default_rng(11), 0, 0.0, 0.0, 0.0
    L = robot.limits
    for _ in range(20000):
        q = np.array([rng.uniform(*L.j1), rng.uniform(*L.j2), rng.uniform(*L.j3_rel), rng.uniform(*L.j4)])
        xyz, yaw = robot.fk(q)
        if robot.limits.excess(q, xyz[2]) > 0:
            continue
        n += 1
        r = robot.ik(xyz, yaw=yaw, seed=q)
        assert r.ok
        x2, y2 = robot.fk(r.q)
        e_pos = max(e_pos, np.abs(x2 - xyz).max())
        e_yaw = max(e_yaw, abs(wrap(y2 - yaw)))
        e_q = max(e_q, np.abs(r.q - q).max())
    rec('PASS' if e_pos < 1e-9 else 'FAIL', f'FK -> IK -> FK round trip ({n} feasible random poses)',
        f'max position error {e_pos:.1e} m, yaw {e_yaw:.1e} rad, joint-recovery {e_q:.1e} rad')

    m = UrdfModel(expand_file(os.path.join(PKG, 'model', 'magician_standalone.urdf.xacro'), PKG,
                              DOF=4, tool='gripper', use_camera='false'), PKG)
    worst = 0.0
    for _ in range(1000):
        q = np.array([rng.uniform(*L.j1), rng.uniform(*L.j2), rng.uniform(*L.j3_rel), rng.uniform(*L.j4)])
        W = m.fk(dict(zip(['magician_joint_%d' % i for i in range(1, 5)], q)))
        t = (W['magician_link_gripper_core'] @ np.array([0, 0, -robot.g.tcp_dz, 1]))[:3]
        worst = max(worst, np.abs(t - robot.fk(q)[0]).max())
    rec('PASS' if worst < 1e-9 else 'FAIL', 'closed-form FK == original URDF/xacro kinematic chain (1000 poses)',
        f'max TCP difference {worst:.1e} m')
    return m


def workspace(out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), dpi=100)
    res = {}
    for name, col in (('conservative', '#1f77b4'), ('datasheet', '#ff7f0e')):
        rb = load_magician(profile=name)
        rb.limits.j3_rel = rb.limits.j3_rel     # as configured
        # sample on a dense grid in (q1=0 plane) for the side view
        P = []
        for q2 in np.linspace(*rb.limits.j2, 140):
            lo, hi = rb.limits.j3_rel or (-100 * D2R, 100 * D2R)
            for q3 in np.linspace(lo, hi, 140):
                q = np.array([0, q2, q3, 0])
                if rb.limits.excess(q, rb.fk(q)[0][2]) <= 1e-9:
                    P.append(rb.fk(q)[0])
        P = np.array(P)
        res[name] = (P[:, 0].max(), P[:, 0].min(), P[:, 2].min(), P[:, 2].max())
        axs[0].scatter(P[:, 0] * 1000, P[:, 2] * 1000, s=2, c=col, label=f'{name} (max r {P[:, 0].max() * 1000:.0f} mm)', alpha=0.5)
    axs[0].axvline(320, color='k', ls='--', lw=1)
    axs[0].text(322, 5, 'datasheet reach 320 mm', fontsize=8)
    axs[0].set_title('Side view: reachable gripper-tip positions (J1 = 0)')
    axs[0].set_xlabel('radius r [mm]'); axs[0].set_ylabel('z [mm]'); axs[0].legend(markerscale=5, fontsize=8)
    axs[0].set_aspect('equal'); axs[0].grid(alpha=0.3)

    rb = load_magician()
    pts = rb.sample_workspace(60000)
    axs[1].scatter(pts[:, 0] * 1000, pts[:, 1] * 1000, s=1, c=pts[:, 2], cmap='viridis')
    axs[1].set_title('Top view (conservative profile), colour = z')
    axs[1].set_xlabel('X [mm]'); axs[1].set_ylabel('Y [mm]'); axs[1].set_aspect('equal'); axs[1].grid(alpha=0.3)
    # reachable height at several radii
    for r_mm, c in ((120, 'C0'), (180, 'C1'), (240, 'C2'), (270, 'C3')):
        zs = [z for (rx, z) in ((p[0], p[2]) for p in pts if abs(math.hypot(p[0], p[1]) - r_mm / 1000) < 0.004)]
        if zs:
            axs[2].plot([r_mm, r_mm], [min(zs) * 1000, max(zs) * 1000], lw=8, c=c, solid_capstyle='butt')
            axs[2].text(r_mm + 4, (min(zs) + max(zs)) * 500, f'{min(zs) * 1000:.0f}..{max(zs) * 1000:.0f}', fontsize=8)
    axs[2].set_title('Usable Z range of the tip vs radius (conservative)')
    axs[2].set_xlabel('radius [mm]'); axs[2].set_ylabel('z [mm]'); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'images', 'phase2_workspace.png'))
    plt.close(fig)
    c, d = res['conservative'], res['datasheet']
    rec('INFO', 'workspace (tip radius / z) per limits profile',
        f'conservative: r {c[1] * 1000:.0f}..{c[0] * 1000:.0f} mm, z {c[2] * 1000:.0f}..{c[3] * 1000:.0f} mm | '
        f'datasheet: r {d[1] * 1000:.0f}..{d[0] * 1000:.0f} mm, z {d[2] * 1000:.0f}..{d[3] * 1000:.0f} mm')
    return pts


def path_demo(robot, urdf_model, out):
    """Square in the XY plane at z=60 mm, straight-line moves, simulated at 100 Hz."""
    c = TwinMotionController(robot)
    corners = [(0.20, -0.04, 0.06), (0.26, -0.04, 0.06), (0.26, 0.04, 0.06), (0.20, 0.04, 0.06), (0.20, -0.04, 0.06)]
    cmd = c.command_ptp_xyz(corners[0])
    assert cmd.ok
    log = []
    seg_dev = []

    def run(T):
        for _ in range(int(T / 0.01) + 20):
            c.tick(0.01)
            log.append((c.clock, c.q.copy(), c.tcp()[0].copy()))
            if not c.busy:
                break
    run(cmd.duration)
    start_idx = len(log)
    for a, b in zip(corners[:-1], corners[1:]):
        k0 = len(log)
        cm = c.command_line_xyz(b)
        assert cm.ok, cm.msg
        run(cm.duration)
        P = np.array([l[2] for l in log[k0:]])
        A, B = np.array(a), np.array(b)
        seg_dev.append(np.linalg.norm(np.cross(P - A, B - A) / np.linalg.norm(B - A), axis=1).max())
    t = np.array([l[0] for l in log]); Q = np.array([l[1] for l in log]); P = np.array([l[2] for l in log])
    V = np.abs(np.diff(Q, axis=0)) / 0.01
    vmax = np.array(c.cfg.vmax)
    feas = all(robot.feasible(q)[0] for q in Q)
    rec('PASS' if max(seg_dev) < 1e-3 else 'FAIL', 'Cartesian straight-line square (60 x 80 mm, z = 60 mm)',
        f'max deviation from ideal line {max(seg_dev) * 1000:.3f} mm; end error {np.linalg.norm(P[-1] - np.array(corners[-1])) * 1000:.4f} mm')
    rec('PASS' if (V.max(axis=0) <= vmax * 1.05).all() else 'FAIL', 'joint velocity limits during path',
        'peak [deg/s] ' + ', '.join(f'{v:.0f}' for v in np.degrees(V.max(axis=0))) + ' (limit '
        + ', '.join(f'{v:.0f}' for v in np.degrees(vmax)) + ')')
    rec('PASS' if feas else 'FAIL', 'every simulated sample inside joint limits and above the floor', f'{len(Q)} samples checked')

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), dpi=100)
    ax = axs[0, 0]
    sq = np.array(corners)
    ax.plot(sq[:, 0] * 1000, sq[:, 1] * 1000, 'k--', lw=1, label='commanded square')
    ax.plot(P[start_idx:, 0] * 1000, P[start_idx:, 1] * 1000, c='C0', lw=2, label='simulated tip path')
    ax.set_aspect('equal'); ax.set_title('Tip path in XY'); ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.legend(); ax.grid(alpha=.3)
    axs[0, 1].plot(t, np.degrees(Q))
    axs[0, 1].legend(['J1', 'J2', 'J3', 'J4'], ncol=4, fontsize=8); axs[0, 1].set_title('Joint angles [deg]'); axs[0, 1].set_xlabel('t [s]'); axs[0, 1].grid(alpha=.3)
    axs[1, 0].plot(t[1:], np.degrees(V))
    for v, col in zip(np.degrees(vmax), ['C0', 'C1', 'C2', 'C3']):
        axs[1, 0].axhline(v, color=col, ls=':', lw=1)
    axs[1, 0].set_title('Joint speed [deg/s] (dotted = limits)'); axs[1, 0].set_xlabel('t [s]'); axs[1, 0].grid(alpha=.3)
    axs[1, 1].plot(t, P[:, 2] * 1000, c='C4'); axs[1, 1].set_ylim(0, 120)
    axs[1, 1].set_title('Tip height Z [mm] (should stay 60)'); axs[1, 1].set_xlabel('t [s]'); axs[1, 1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'images', 'phase2_path_demo.png')); plt.close(fig)

    # robot renders at the four corners
    fig, axs = plt.subplots(1, 4, figsize=(18, 5), dpi=100)
    for ax, p in zip(axs, corners[:4]):
        r = robot.ik(p, q4=0.0)
        q = {f'magician_joint_{i + 1}': r.q[i] for i in range(4)}
        rm.draw(ax, urdf_model, q, rm.D(40), rm.D(25), title=f'tip at X {p[0] * 1000:.0f}, Y {p[1] * 1000:.0f}, Z {p[2] * 1000:.0f} mm')
        cm_, rr, uu = rm.camera(rm.D(40), rm.D(25))
        ax.plot(np.array(p) @ rr, np.array(p) @ uu, 'o', color='#00d060', ms=7, zorder=9)
        fk_p = urdf_model.fk(q)['magician_link_gripper_core'] @ np.array([0, 0, -robot.g.tcp_dz, 1])
        ax.plot(fk_p[:3] @ rr, fk_p[:3] @ uu, 'x', color='red', ms=9, mew=2, zorder=10)
    fig.suptitle('IK -> original URDF meshes: green dot = commanded tip position, red x = tip computed from the URDF chain', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'images', 'phase2_corners.png')); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'docs'))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    robot = load_magician()
    m = accuracy(robot)
    workspace(a.out)
    path_demo(robot, m, a.out)
    lines = ['# Phase 2 verification report', '', 'Generated by `tools/phase2_demo.py`.', '',
             '| Status | Check | Detail |', '|---|---|---|']
    lines += [f'| {s} | {n} | {d} |' for s, n, d in rows]
    open(os.path.join(a.out, 'PHASE2_VERIFICATION.md'), 'w').write('\n'.join(lines) + '\n')
    sys.exit(1 if any(r[0] == 'FAIL' for r in rows) else 0)


if __name__ == '__main__':
    main()
