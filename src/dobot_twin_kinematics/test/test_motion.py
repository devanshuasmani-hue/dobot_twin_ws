import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dobot_twin_kinematics.kinematics import D2R   # noqa: E402
from dobot_twin_kinematics.motion import MotionConfig, TwinMotionController   # noqa: E402
from dobot_twin_kinematics.params import load_magician   # noqa: E402


def run(c, seconds, dt=0.01, log=None):
    for _ in range(int(seconds / dt)):
        c.tick(dt)
        if log is not None:
            log.append((c.clock, c.q.copy(), c.tcp()[0].copy()))


def new():
    return TwinMotionController(load_magician())


def test_ptp_reaches_target_and_respects_velocity_and_limits():
    c = new()
    cmd = c.command_ptp_xyz([0.20, 0.10, 0.05])
    assert cmd.ok and cmd.duration > 0
    log = []
    run(c, cmd.duration + 0.2, log=log)
    assert c.status == 'IDLE'
    assert np.allclose(c.tcp()[0], [0.20, 0.10, 0.05], atol=1e-9)
    qs = np.array([l[1] for l in log])
    vel = np.abs(np.diff(qs, axis=0)) / 0.01
    assert (vel.max(axis=0) <= np.array(c.cfg.vmax) * 1.02).all()
    lim = c.robot.limits
    assert (qs[:, 0] >= lim.j1[0] - 1e-9).all() and (qs[:, 0] <= lim.j1[1] + 1e-9).all()
    assert (qs[:, 1] >= lim.j2[0] - 1e-9).all() and (qs[:, 1] <= lim.j2[1] + 1e-9).all()


def test_line_is_straight_and_stays_feasible():
    c = new()
    assert c.command_ptp_xyz([0.20, -0.08, 0.08]).ok
    run(c, 5)
    cmd = c.command_line_xyz([0.20, 0.08, 0.08])
    assert cmd.ok
    log = []
    run(c, cmd.duration + 0.2, log=log)
    pts = np.array([l[2] for l in log])
    a, b = np.array([0.20, -0.08, 0.08]), np.array([0.20, 0.08, 0.08])
    d = np.cross(pts - a, b - a) / np.linalg.norm(b - a)
    dev = np.linalg.norm(d, axis=1).max()
    assert dev < 1e-3                                  # < 1 mm from the ideal line (path sampling 2 mm)
    for _, q, p in log:
        assert c.robot.feasible(q)[0]
    assert np.allclose(c.tcp()[0], b, atol=1e-9)


def test_unreachable_is_rejected_without_motion():
    c = new()
    q0 = c.q.copy()
    cmd = c.command_ptp_xyz([0.6, 0, 0.1])
    assert not cmd.ok and 'unreachable' in cmd.msg and c.traj is None
    run(c, 0.5)
    assert np.allclose(c.q, q0)


def test_line_blocked_midway_is_rejected_before_moving():
    c = new()
    c.set_state(c.robot.ik([0.22, 0.0, 0.10]).q)
    q0 = c.q.copy()
    cmd = c.command_line_xyz([0.22, 0.0, -0.05])       # straight down through the table
    assert not cmd.ok and ('blocked' in cmd.msg)
    run(c, 0.3)
    assert np.allclose(c.q, q0)


def test_joint_command_checks_limits_and_floor():
    c = new()
    assert not c.command_joints([0, 85 * D2R, 0, 0]).ok          # tool below the table
    assert not c.command_joints([140 * D2R, 0, 0, 0]).ok         # J1 beyond limit
    assert c.command_joints([30 * D2R, 40 * D2R, 20 * D2R, 10 * D2R]).ok


def test_jog_moves_and_stops_on_timeout_and_at_boundary():
    c = new()
    c.set_state(c.robot.ik([0.20, 0.0, 0.08]).q)
    p0 = c.tcp()[0]
    for _ in range(50):                                # 0.5 s at 40 mm/s -> 20 mm
        c.jog([0.04, 0, 0])
        c.tick(0.01)
    p1 = c.tcp()[0]
    assert abs((p1 - p0)[0] - 0.02) < 1e-3 and abs((p1 - p0)[1]) < 1e-6
    run(c, 0.6)                                        # no refresh -> dead-man stop
    p2 = c.tcp()[0]
    assert c.status == 'IDLE'
    assert 0 <= (p2 - p1)[0] <= 0.04 * (c.cfg.jog_timeout + 0.02)   # overshoot bounded by the timeout
    p1 = p2
    run(c, 0.5)
    assert np.allclose(c.tcp()[0], p1)                 # and stays put
    # jog outward until the reach limit: must stop, never exceed limits
    for _ in range(1500):
        c.jog([0.1, 0, 0])
        c.tick(0.01)
        assert c.robot.feasible(c.q)[0]
    assert c.status.startswith('LIMIT')
    assert c.tcp()[0][0] < 0.30


def test_gripper_moves_at_limited_speed_and_clamps():
    c = new()
    c.command_gripper(1.0)                             # clamped to 0.0135
    run(c, 0.1)
    assert 0 < c.g < 0.0135
    run(c, 1.0)
    assert abs(c.g - 0.0135) < 1e-12
    c.command_gripper(-1.0)
    run(c, 2.0)
    assert c.g == 0.0


def test_home_returns_to_zero():
    c = new()
    cmd = c.command_ptp_xyz([0.25, 0.0, 0.06])
    run(c, cmd.duration + 0.1)
    h = c.home()
    run(c, h.duration + 0.1)
    assert np.allclose(c.q, 0, atol=1e-9)
