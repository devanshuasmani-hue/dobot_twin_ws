import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dobot_twin_kinematics.kinematics import D2R, Geometry, Limits, Magician, wrap   # noqa: E402
from dobot_twin_kinematics.params import load_magician                              # noqa: E402

SRC = os.path.abspath(os.path.join(HERE, '..', '..'))
PKG = os.path.join(SRC, 'dobot_description')
sys.path.insert(0, os.path.join(SRC, 'dobot_twin_description', 'tools'))


@pytest.fixture(scope='module')
def robot():
    return load_magician()


def rand_q(rng, lim):
    return np.array([rng.uniform(*lim.j1), rng.uniform(*lim.j2), rng.uniform(*lim.j3_rel), rng.uniform(*lim.j4)])


def test_fk_matches_original_urdf_chain(robot):
    """Closed-form FK == FK through the ORIGINAL xacro/URDF chain (+ TCP offset), 500 random poses."""
    from mini_xacro import expand_file
    from urdf_model import UrdfModel
    m = UrdfModel(expand_file(os.path.join(PKG, 'model', 'magician_standalone.urdf.xacro'), PKG,
                              DOF=4, tool='gripper', use_camera='false'), PKG)
    rng, worst = np.random.default_rng(3), 0.0
    for _ in range(500):
        q = rand_q(rng, robot.limits)
        W = m.fk({'magician_joint_1': q[0], 'magician_joint_2': q[1], 'magician_joint_3': q[2], 'magician_joint_4': q[3]})
        tcp_urdf = (W['magician_link_gripper_core'] @ np.array([0, 0, -robot.g.tcp_dz, 1]))[:3]
        xyz, yaw = robot.fk(q)
        worst = max(worst, np.abs(tcp_urdf - xyz).max())
        yaw_u = math.atan2(W['magician_link_gripper_core'][1, 0], W['magician_link_gripper_core'][0, 0])
        assert abs(wrap(yaw_u - yaw)) < 1e-9
    assert worst < 1e-9


def test_tcp_offset_matches_fingertip_mesh(robot):
    """tcp_dz equals the real fingertip depth measured from the gripper DAE meshes (closed gripper)."""
    from dae_loader import load_dae
    from mini_xacro import expand_file
    from urdf_model import UrdfModel
    m = UrdfModel(expand_file(os.path.join(PKG, 'model', 'magician_standalone.urdf.xacro'), PKG,
                              DOF=4, tool='gripper', use_camera='false'), PKG)
    W = m.fk({})
    zs = []
    for l in ('magician_link_gripper_jaw_left', 'magician_link_gripper_jaw_right'):
        P = load_dae(m.mesh_path(l))['tris'].reshape(-1, 3)
        M = np.linalg.inv(W['magician_link_gripper_core']) @ W[l] @ m.visual_T(l)
        zs.append((np.c_[P, np.ones(len(P))] @ M.T)[:, 2].min())
    assert abs(-min(zs) - robot.g.tcp_dz) < 5e-4


def test_roundtrip_ik_fk_exact(robot):
    rng = np.random.default_rng(1)
    n_ok = 0
    for _ in range(3000):
        q = rand_q(rng, robot.limits)
        xyz, yaw = robot.fk(q)
        if robot.limits.excess(q, xyz[2]) > 0:
            continue
        n_ok += 1
        r = robot.ik(xyz, yaw=yaw, seed=q)
        assert r.ok, (q, r.reason)
        xyz2, yaw2 = robot.fk(r.q)
        assert np.abs(xyz2 - xyz).max() < 1e-9 and abs(wrap(yaw2 - yaw)) < 1e-9
        assert np.abs(r.q[:3] - q[:3]).max() < 1e-6          # seeded -> same branch recovered
    assert n_ok > 500


def test_both_elbow_branches_exist_but_limits_pick_valid_one(robot):
    r = robot.ik([0.20, 0.0, 0.05])
    assert r.ok and len(r.solutions) >= 1
    assert robot.feasible(r.q)[0]


def test_unreachable_far(robot):
    r = robot.ik([0.50, 0.0, 0.1])
    assert not r.ok and 'unreachable' in r.reason


def test_unreachable_too_close(robot):
    r = robot.ik([0.0, 0.0, 0.1])         # directly above the base axis
    assert not r.ok


def test_floor_enforced(robot):
    assert not robot.ik([0.2, 0.0, -0.02]).ok
    assert 'floor' in robot.ik([0.2, 0.0, -0.02]).reason or 'limits' in robot.ik([0.2, 0.0, -0.02]).reason


def test_limit_violation_reports_joint(robot):
    # far to the side behind: needs |q1| > 125 deg
    r = robot.ik([-0.2, 0.01, 0.05])
    assert not r.ok and 'J1' in r.reason


def test_yaw_semantics(robot):
    r = robot.ik([0.2, 0.1, 0.05], yaw=30 * D2R)
    assert r.ok
    assert abs(wrap(robot.fk(r.q)[1] - 30 * D2R)) < 1e-9


def test_zero_pose_matches_expected_numbers(robot):
    xyz, yaw = robot.fk([0, 0, 0, 0])
    g = robot.g
    assert np.allclose(xyz, [g.l3 + g.l4, 0, g.h + g.l2 - g.tcp_dz])


def test_profiles_load_and_differ():
    c, d = load_magician(profile='conservative'), load_magician(profile='datasheet')
    assert c.limits.j3_rel is not None and d.limits.j3_rel is None
    # datasheet profile reaches further than the conservative one
    assert d.sample_workspace(4000)[:, :2].max() > c.sample_workspace(4000)[:, :2].max()
    with pytest.raises(ValueError):
        load_magician(profile='nope')
