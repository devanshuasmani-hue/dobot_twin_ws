#!/usr/bin/env python3
"""Phase 1 verification of the Dobot Magician model (limits, transforms, meshes).

ROS-independent. Uses the real `xacro` binary if on PATH (cross-checked against the
built-in mini expander), otherwise only the mini expander.

  python3 verify_model.py --pkg ../../dobot_description [--checksums ../../../ORIGINAL_SHA256.txt]
                          [--report ../docs/PHASE1_VERIFICATION.md]
Exit code 1 if any check FAILs. WARN = known issue in the original model, documented.
"""
import argparse
import hashlib
import itertools
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dae_loader import load_dae          # noqa: E402
from mini_xacro import expand_file       # noqa: E402
from urdf_model import UrdfModel         # noqa: E402

D = math.radians
# Published datasheet (Dobot Magician flyer): base +-135, rear arm 0..85, forearm -10..95, EE +-90
DATASHEET = {'joint_1': (-135, 135), 'joint_2': (0, 85), 'joint_3': (-10, 95), 'joint_4': (-90, 90)}
DATASHEET_REACH_M = 0.320

# valid launch configurations (from display.launch.py)
CONFIGS = [dict(DOF=3, tool='none', use_camera='false'), dict(DOF=3, tool='pen', use_camera='false')] + \
          [dict(DOF=4, tool=t, use_camera=c) for t in ('gripper', 'extended_gripper', 'suction_cup')
           for c in ('false', 'true')]

results = []


def rec(status, name, msg=''):
    results.append((status, name, msg))
    print(f"[{status:4s}] {name}" + (f" -- {msg}" if msg else ''))


def load_model(pkg, **args):
    root = expand_file(os.path.join(pkg, 'model', 'magician_standalone.urdf.xacro'), pkg, **args)
    return UrdfModel(root, pkg)


def verts(m, link, W):
    P = load_dae(m.mesh_path(link))['tris'].reshape(-1, 3)
    M = W[link] @ m.visual_T(link)
    return (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]


# ---------------------------------------------------------------------------
def check_provenance(pkg, checksum_file):
    if not checksum_file or not os.path.exists(checksum_file):
        rec('WARN', 'provenance', 'no checksum manifest given')
        return
    base = os.path.dirname(pkg)
    bad = []
    for line in open(checksum_file):
        h, p = line.split(None, 1)
        p = p.strip()
        fp = os.path.join(base, p)
        if not os.path.exists(fp) or hashlib.sha256(open(fp, 'rb').read()).hexdigest() != h:
            bad.append(p)
    rec('FAIL' if bad else 'PASS', 'original package unmodified (sha256 of all files)',
        f"{len(bad)} differ: {bad[:3]}" if bad else 'all files identical to the supplied zip')


def check_configs(pkg):
    for c in CONFIGS:
        tag = f"config DOF={c['DOF']} tool={c['tool']} camera={c['use_camera']}"
        try:
            m = load_model(pkg, **c)
        except Exception as e:  # noqa: BLE001
            rec('FAIL', tag, f'expansion error: {e!r}')
            continue
        names = [l.get('name') for l in m.root.findall('link')]
        jn = [j.get('name') for j in m.root.findall('joint')]
        problems = []
        if len(set(names)) != len(names) or len(set(jn)) != len(jn):
            problems.append('duplicate names')
        if m.root_link != 'magician_root_link' or len(m.roots) != 1:
            problems.append(f'roots={m.roots}')
        for j in m.joints.values():
            for k in ('parent', 'child'):
                if j.find(k).get('link') not in m.links:
                    problems.append('dangling ' + k)
        reach = set()
        W = None
        try:
            W = m.fk({})
            reach = set(W)
        except Exception as e:  # noqa: BLE001
            problems.append(f'fk {e!r}')
        if reach != set(m.links):
            problems.append('graph not a single connected tree')
        for l in m.links:
            p = m.mesh_path(l)
            if p and not os.path.exists(p):
                problems.append('missing mesh ' + p)
        rec('FAIL' if problems else 'PASS', tag,
            '; '.join(problems) if problems else f'{len(names)} links, {len(jn)} joints, tree OK, meshes exist')


def _norm_attr(el):
    return {k: v for k, v in sorted(el.attrib.items())}


def _vec(s):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(0)


def check_clean_equivalence(pkg):
    clean = UrdfModel(ET.parse(os.path.join(pkg, 'model', 'clean_model_no_macros.urdf')).getroot(), pkg)
    exp = load_model(pkg, DOF=4, tool='gripper', use_camera='false')
    problems = []
    if set(clean.links) != set(exp.links):
        problems.append(f'link sets differ {set(clean.links) ^ set(exp.links)}')
    if set(clean.joints) != set(exp.joints):
        problems.append(f'joint sets differ {set(clean.joints) ^ set(exp.joints)}')
    worst = 0.0
    for n in set(clean.joints) & set(exp.joints):
        a, b = clean.joints[n], exp.joints[n]
        if a.get('type') != b.get('type') or a.find('parent').get('link') != b.find('parent').get('link') \
                or a.find('child').get('link') != b.find('child').get('link'):
            problems.append(f'{n}: type/parent/child differ')
        for tag, attrs in (('origin', ('xyz', 'rpy')), ('axis', ('xyz',)), ('limit', ('lower', 'upper'))):
            for at in attrs:
                va, vb = _vec(a.find(tag).get(at)), _vec(b.find(tag).get(at))
                worst = max(worst, float(np.abs(va - vb).max()))
        ma, mb = a.find('mimic'), b.find('mimic')
        if (ma is None) != (mb is None) or (ma is not None and (ma.get('joint'), ma.get('multiplier')) != (mb.get('joint'), mb.get('multiplier'))):
            problems.append(f'{n}: mimic differs')
    for n in set(clean.links) & set(exp.links):
        va, vb = clean.links[n].find('visual'), exp.links[n].find('visual')
        if (va is None) != (vb is None):
            problems.append(f'{n}: visual presence differs')
            continue
        if va is None:
            continue
        ga, gb = va.find('geometry/mesh'), vb.find('geometry/mesh')
        if (ga is None) != (gb is None) or (ga is not None and ga.get('filename') != gb.get('filename')):
            problems.append(f'{n}: mesh differs')
        for at in ('xyz', 'rpy'):
            worst = max(worst, float(np.abs(_vec(va.find('origin').get(at)) - _vec(vb.find('origin').get(at))).max()))
    # base joint z differs? standalone uses 0.131305 in both
    rec('FAIL' if problems or worst > 1e-9 else 'PASS',
        'clean_model_no_macros.urdf == expansion(standalone, DOF=4, gripper)',
        '; '.join(problems) if problems else f'identical structure; max numeric deviation {worst:.2e}')


def check_real_xacro(pkg):
    exe = shutil.which('xacro')
    if not exe:
        rec('WARN', 'real xacro cross-check', 'xacro not installed here; run this script on a ROS machine to cross-check')
        return
    worst = 0.0
    for c in CONFIGS:
        out = subprocess.run([exe, os.path.join(pkg, 'model', 'magician_standalone.urdf.xacro')] +
                             [f'{k}:={v}' for k, v in c.items()], capture_output=True, text=True)
        if out.returncode:
            rec('FAIL', 'real xacro cross-check', out.stderr[:200])
            return
        a = UrdfModel(ET.fromstring(out.stdout), pkg)
        b = load_model(pkg, **c)
        for n in b.joints:
            for tag, at in (('origin', 'xyz'), ('origin', 'rpy'), ('limit', 'upper')):
                worst = max(worst, float(np.abs(_vec(a.joints[n].find(tag).get(at)) - _vec(b.joints[n].find(tag).get(at))).max()))
    rec('FAIL' if worst > 1e-9 else 'PASS', 'real xacro vs mini expander (all configs)', f'max deviation {worst:.2e}')


def check_meshes(pkg, m):
    bad = []
    info = []
    for l in m.links:
        p = m.mesh_path(l)
        if not p:
            continue
        d = load_dae(p)
        t = d['tris']
        if abs(d['unit'] - 1.0) > 1e-9 or d['up'] != 'Z_UP':
            bad.append(f"{os.path.basename(p)} unit={d['unit']} up={d['up']}")
        if not np.isfinite(t).all() or len(t) == 0:
            bad.append(f'{os.path.basename(p)} empty/non-finite')
        ext = t.reshape(-1, 3).max(0) - t.reshape(-1, 3).min(0)
        if ext.max() > 0.5:
            bad.append(f'{os.path.basename(p)} extent {ext.max():.2f} m (units?)')
        info.append(len(t))
    rec('FAIL' if bad else 'PASS', 'meshes (default config): meters, Z_UP, finite, sane extents',
        '; '.join(bad) if bad else f'{len(info)} meshes, {sum(info)} triangles')


def check_limits(pkg, m):
    # 1. malformed limits
    for n in m.actuated() + list(m.mimics()):
        lo, hi = m.limits(n)
        if lo > hi:
            rec('WARN', f'limits {n}', f'lower {lo} > upper {hi} (invalid per URDF spec; original model defect; benign in robot_state_publisher)')
    # 2. mimic requirement vs declared limits
    for n, mm in m.mimics().items():
        if m.joints[n].get('type') != 'revolute':
            continue
        src = mm.get('joint')
        slo, shi = m.limits(src)
        mult = float(mm.get('multiplier'))
        need = sorted([mult * slo, mult * shi])
        lo, hi = m.limits(n)
        if need[0] < lo - 1e-9 or need[1] > hi + 1e-9:
            rec('WARN', f'mimic {n}', f'needs [{math.degrees(need[0]):.0f},{math.degrees(need[1]):.0f}] deg but declares '
                f'[{math.degrees(lo):.0f},{math.degrees(hi):.0f}]; clamping loaders would break the level wrist '
                f'(RViz2/robot_state_publisher does not clamp)')
    # 3. vs datasheet
    rows = []
    for k, (dlo, dhi) in DATASHEET.items():
        lo, hi = (math.degrees(x) for x in m.limits('magician_' + k))
        rows.append(f'{k}: URDF [{lo:.0f},{hi:.0f}] vs datasheet [{dlo},{dhi}]')
    rec('WARN', 'URDF limits vs datasheet', ' | '.join(rows) + '  (J3 URDF value is relative to the upper arm)')


def planar_fk(q1, q2, q3rel, L=(0.131305, 0.135, 0.147, 0.06)):
    h, l2, l3, l4 = L
    r = l2 * math.sin(q2) + l3 * math.cos(q2 + q3rel)
    z = h + l2 * math.cos(q2) - l3 * math.sin(q2 + q3rel)
    return np.array([r * math.cos(q1), r * math.sin(q1), z]), np.array([(r + l4) * math.cos(q1), (r + l4) * math.sin(q1), z])


def check_kinematics(m):
    rng = np.random.default_rng(7)
    lim = {n: m.limits('magician_' + n) for n in ('joint_1', 'joint_2', 'joint_3', 'joint_4')}
    worst_pos, worst_lvl, worst_yaw = 0.0, 0.0, 0.0
    for _ in range(2000):
        q = {f'magician_{n}': rng.uniform(*lim[n]) for n in lim}
        W = m.fk(q)
        wrist, tool = planar_fk(q['magician_joint_1'], q['magician_joint_2'], q['magician_joint_3'])
        worst_pos = max(worst_pos, np.abs(W['magician_link_4'][:3, 3] - wrist).max(),
                        np.abs(W['magician_link_gripper_core'][:3, 3] - tool).max())
        worst_lvl = max(worst_lvl, abs(W['magician_link_4'][2, 2] - 1), abs(W['magician_link_gripper_core'][2, 2] - 1))
        # tool yaw about world Z must be j1 + j4
        yaw = math.atan2(W['magician_link_gripper_core'][1, 0], W['magician_link_gripper_core'][0, 0])
        exp = q['magician_joint_1'] + q['magician_joint_4']
        worst_yaw = max(worst_yaw, abs(math.atan2(math.sin(yaw - exp), math.cos(yaw - exp))))
    rec('PASS' if worst_pos < 1e-9 else 'FAIL', 'FK (URDF chain) == closed-form planar model, 2000 random poses',
        f'max position error {worst_pos:.2e} m')
    rec('PASS' if worst_lvl < 1e-9 else 'FAIL', 'wrist/tool stay vertical (parallelogram via mimic joints)', f'max tilt error {worst_lvl:.2e}')
    rec('PASS' if worst_yaw < 1e-9 else 'FAIL', 'tool yaw (world) == joint_1 + joint_4', f'max error {worst_yaw:.2e} rad')


def check_workspace(m):
    gl = [l for l in m.links if 'gripper' in l and m.mesh_path(l)]
    # subsample vertices for speed
    pts = {l: load_dae(m.mesh_path(l))['tris'].reshape(-1, 3)[::25] for l in gl}

    def lowest(q):
        W = m.fk(q)
        zmin = 9
        for l in gl:
            M = W[l] @ m.visual_T(l)
            zmin = min(zmin, (np.c_[pts[l], np.ones(len(pts[l]))] @ M.T)[:, 2].min())
        return zmin

    j2l, j3l = m.limits('magician_joint_2'), m.limits('magician_joint_3')
    best_r, low = 0, 9
    low_q = None
    for a in np.linspace(*j2l, 24):
        for b in np.linspace(*j3l, 24):
            W = m.fk({'magician_joint_2': a, 'magician_joint_3': b})
            best_r = max(best_r, W['magician_link_gripper_core'][0, 3])
            z = lowest({'magician_joint_2': a, 'magician_joint_3': b})
            if z < low:
                low, low_q = z, (math.degrees(a), math.degrees(b))
    # datasheet envelope assuming datasheet J3 is absolute (j3_rel = j3_abs - j2) and relative URDF limit also holds
    best_r_d, low_d = 0, 9
    for a in np.linspace(D(0), D(85), 24):
        for babs in np.linspace(D(-10), D(95), 28):
            brel = babs - a
            if not (j3l[0] <= brel <= j3l[1]):
                continue
            q = {'magician_joint_2': a, 'magician_joint_3': brel}
            W = m.fk(q)
            best_r_d = max(best_r_d, W['magician_link_gripper_core'][0, 3])
            low_d = min(low_d, lowest(q))
    rec('INFO', 'workspace: URDF per-joint limits (independent)',
        f'max tool-axis radius {best_r*1000:.0f} mm; lowest gripper point z={low*1000:.0f} mm at j2,j3=({low_q[0]:.0f},{low_q[1]:.0f}) deg '
        f'(<0 means below table plane -> twin needs coupled limits / z floor)')
    rec('INFO', 'workspace: datasheet envelope (J3 assumed absolute) AND URDF relative-J3 limit',
        f'max tool-axis radius {best_r_d*1000:.0f} mm (datasheet reach {DATASHEET_REACH_M*1000:.0f} mm incl. tool offsets); '
        f'lowest gripper point z={low_d*1000:.0f} mm')


def check_alignment(m):
    W0 = m.fk({})
    # base centred on the J1 axis
    vb = verts(m, 'magician_base_link', W0)
    c = (vb.min(0) + vb.max(0)) / 2
    ok = abs(c[0]) < 2e-3 and abs(c[1]) < 2e-3
    rec('PASS' if ok else 'FAIL', 'base mesh centred on J1 axis', f'bbox centre xy = ({c[0]*1000:.2f},{c[1]*1000:.2f}) mm')
    rec('PASS' if abs(vb[:, 2].min()) < 2e-3 else 'FAIL', 'base mesh sits on the table plane (z=0 at root)',
        f'lowest base vertex z = {vb[:, 2].min()*1000:.2f} mm')
    # pivot consistency: adjacent meshes keep contact across the full joint range
    for jn, tol in (('magician_joint_1', 2.0), ('magician_joint_2', 2.0), ('magician_joint_3', 2.0), ('magician_joint_4', 5.0)):
        j = m.joints[jn]
        p, c_ = j.find('parent').get('link'), j.find('child').get('link')
        lo, hi = m.limits(jn)
        gaps = []
        for v in np.linspace(lo, hi, 9):
            W = m.fk({jn: v})
            a, b = verts(m, p, W)[::5], verts(m, c_, W)[::5]
            gaps.append(cKDTree(a).query(b)[0].min() * 1000)
        spread = max(gaps) - min(gaps)
        rec('PASS' if max(gaps) < tol and spread < 1.5 else 'FAIL', f'pivot alignment {jn}: {p[9:]} <-> {c_[9:]}',
            f'min mesh gap over full range {min(gaps):.2f}..{max(gaps):.2f} mm (spread {spread:.2f})')
    # parallelogram: link_4 mesh orientation constant => its bbox size constant
    sizes = []
    for q2, q3 in ((0, 0), (D(60), D(20)), (D(85), D(-10)), (D(-5), D(60))):
        W = m.fk({'magician_joint_2': q2, 'magician_joint_3': q3})
        v = verts(m, 'magician_link_4', W)
        sizes.append(v.max(0) - v.min(0))
    sp = np.ptp(np.array(sizes), axis=0).max()
    rec('PASS' if sp < 1e-6 else 'FAIL', 'link_4 mesh keeps orientation across poses', f'bbox size spread {sp:.2e} m')
    # gripper: closed at 0, open at 0.0135, symmetric about the tool x-z plane
    for s, name in ((0.0, 'closed'), (0.0135, 'open')):
        W = m.fk({'magician_joint_prismatic_l': s})
        Wi = np.linalg.inv(W['magician_link_gripper_core'])

        def loc(link):
            v = verts(m, link, W)
            return (np.c_[v, np.ones(len(v))] @ Wi.T)[:, :3]
        L, R = loc('magician_link_gripper_jaw_left'), loc('magician_link_gripper_jaw_right')
        tl, tr = L[L[:, 2] < L[:, 2].min() + 0.006], R[R[:, 2] < R[:, 2].min() + 0.006]
        gap = tl[:, 1].min() - tr[:, 1].max()
        exp = 2 * s
        rec('PASS' if abs(gap - exp) < 1e-3 and (s > 0 or abs(L[:, 1].min()) < 2e-3) else 'FAIL',
            f'gripper {name} (prismatic_l={s} m)', f'finger-tip gap {gap*1000:.2f} mm (expected {exp*1000:.1f}); '
            f'left jaw moves +y, right jaw -y, symmetric')


def write_report(path):
    lines = ['# Phase 1 verification report', '',
             'Generated by `tools/verify_model.py` (do not edit by hand).', '',
             '| Status | Check | Detail |', '|---|---|---|']
    for s, n, msg in results:
        lines.append(f'| {s} | {n} | {msg.replace("|", "/")} |')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', required=True, help='path to the (unmodified) dobot_description package')
    ap.add_argument('--checksums')
    ap.add_argument('--report')
    a = ap.parse_args()
    pkg = os.path.abspath(a.pkg)
    check_provenance(pkg, a.checksums)
    check_configs(pkg)
    check_clean_equivalence(pkg)
    check_real_xacro(pkg)
    m = load_model(pkg, DOF=4, tool='gripper', use_camera='false')
    check_meshes(pkg, m)
    check_limits(pkg, m)
    check_kinematics(m)
    check_workspace(m)
    check_alignment(m)
    n = {s: sum(1 for r in results if r[0] == s) for s in ('PASS', 'FAIL', 'WARN', 'INFO')}
    print(f"\nSUMMARY: {n}")
    if a.report:
        write_report(a.report)
    sys.exit(1 if n['FAIL'] else 0)


if __name__ == '__main__':
    main()
