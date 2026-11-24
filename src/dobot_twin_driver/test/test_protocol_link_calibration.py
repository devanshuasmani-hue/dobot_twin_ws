import math
import struct

import numpy as np
import pytest

from dobot_twin_driver.fake_dobot import FakeDobot
from dobot_twin_driver.frame_map import Calibration, DobotFrame, fit_calibration
from dobot_twin_driver.link import DobotLink
from dobot_twin_driver.protocol import (ID, Frame, FrameParser, Pose, PTPMode, encode, req_pose)
from dobot_twin_kinematics.params import load_magician


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def robot():
    return load_magician()


# ---------------------------------------------------------------- protocol
def test_known_getpose_frame_bytes():
    assert encode(req_pose()) == bytes([0xAA, 0xAA, 0x02, 0x0A, 0x00, 0xF6])


def test_roundtrip_and_resync():
    p = FrameParser()
    f = Frame(10, 0, struct.pack('<8f', 1, 2, 3, 4, 5, 6, 7, 8))
    raw = encode(f)
    # garbage, a corrupted frame, then two good frames split at arbitrary points
    bad = bytearray(raw)
    bad[-1] ^= 0xFF
    stream = b'\x00\x12\xAA' + bytes(bad) + raw + raw
    out = []
    for i in range(0, len(stream), 7):
        out += p.feed(stream[i:i + 7])
    assert len(out) == 2 and out[0].id == 10 and p.bad_frames >= 1
    assert Pose.from_params(out[0].params).j4 == 8.0


# ---------------------------------------------------------------- link + fake
def make(robot, **kw):
    clk = Clock()
    fake = FakeDobot(robot, time_fn=clk, **kw)
    return fake, DobotLink(fake, timeout=0.3), clk


def test_get_pose_matches_hidden_model(robot):
    fake, link, _ = make(robot)
    fake.drag_to([20, 30, 40, 10])
    pose = link.get_pose()
    assert pose is not None
    assert np.allclose([pose.j1, pose.j2, pose.j3, pose.j4], [20, 30, 40, 10], atol=1e-4)
    assert np.allclose([pose.x, pose.y, pose.z], fake.pose().__dict__ and [fake.pose().x, fake.pose().y, fake.pose().z], atol=1e-3)


def test_timeout_when_disconnected(robot):
    import time
    fake = FakeDobot(robot)
    fake.connected = False
    link = DobotLink(fake, timeout=0.05)
    t0 = time.monotonic()
    assert link.get_pose() is None
    assert time.monotonic() - t0 < 0.5 and link.errors >= 1


def test_alarms_read_and_clear(robot):
    fake, link, _ = make(robot, alarms=[3, 17])
    assert link.get_alarms() == [3, 17]
    assert link.clear_alarms()
    assert link.get_alarms() == []


def test_queued_motion_only_runs_after_queue_start_and_uses_speed_ratio(robot):
    fake, link, clk = make(robot)
    assert link.set_speed(50, 50)
    idx = link.ptp(PTPMode.MOVJ_ANGLE, 45, 0, 0, 0)
    assert idx == 2
    clk.t += 2.0
    assert abs(link.get_pose().j1) < 1e-6                    # queue not started: no motion
    assert link.queue_start()
    clk.t += 0.5                                              # 50 % of 90 deg/s = 45 deg/s -> 22.5 deg
    assert link.get_pose().j1 == pytest.approx(22.5, abs=0.1)
    clk.t += 1.0
    assert link.get_pose().j1 == pytest.approx(45.0, abs=1e-4)


def test_queue_stop_and_clear_halt_pending_moves(robot):
    fake, link, clk = make(robot)
    link.set_speed(20, 20)
    link.queue_start()
    link.ptp(PTPMode.MOVJ_ANGLE, 90, 0, 0, 0)
    clk.t += 1.0
    link.queue_stop()
    link.queue_clear()
    j = link.get_pose().j1
    clk.t += 10
    assert link.get_pose().j1 == pytest.approx(j, abs=1e-6) and j < 90


# ---------------------------------------------------------------- calibration
def sample_poses(fake, link, js):
    out = []
    for j in js:
        fake.drag_to(j)
        p = link.get_pose()
        out.append((p.joints_deg(), (p.x, p.y, p.z), p.r))
    return out


GOOD = [[0, 10, 5, 0], [30, 40, 30, 20], [-40, 60, 20, -30], [10, 25, 60, 45], [50, 70, 10, 0], [-20, 15, 45, 10]]


@pytest.mark.parametrize('conv', ['absolute', 'relative'])
@pytest.mark.parametrize('rmode', ['j1+j4', 'j4'])
def test_calibration_recovers_hidden_truth(robot, conv, rmode):
    truth = Calibration(j3_convention=conv, dr_mm=-12.0, dz_mm=35.0, r_mode=rmode, calibrated=True)
    fake, link, _ = make(robot, truth=truth)
    rep = fit_calibration(sample_poses(fake, link, GOOD), robot)
    assert rep.ok, rep.message
    c = rep.cal
    assert c.j3_convention == conv and c.r_mode == rmode
    assert c.dr_mm == pytest.approx(-12.0, abs=1e-3) and c.dz_mm == pytest.approx(35.0, abs=1e-3)
    assert c.residual_mm < 1e-3


def test_calibration_refuses_similar_poses(robot):
    fake, link, _ = make(robot)
    rep = fit_calibration(sample_poses(fake, link, [[0, 30, 30, 0], [10, 31, 30, 0], [20, 30, 33, 0], [30, 32, 31, 0]]), robot)
    assert not rep.ok and 'similar' in rep.message


def test_calibration_refuses_too_few_poses(robot):
    fake, link, _ = make(robot)
    assert not fit_calibration(sample_poses(fake, link, GOOD[:2]), robot).ok


def test_calibration_refuses_model_mismatch(robot):
    fake, link, _ = make(robot)
    s = sample_poses(fake, link, GOOD)
    bad = [(j, (x + (8 if i % 2 else -8), y, z + (6 if i % 3 else -6)), r) for i, (j, (x, y, z), r) in enumerate(s)]
    rep = fit_calibration(bad, robot)
    assert not rep.ok and 'mismatch' in rep.message


def test_frame_roundtrip_with_offsets(robot):
    cal = Calibration('absolute', dr_mm=-12, dz_mm=35, r_mode='j1+j4', calibrated=True)
    fr = DobotFrame(robot, cal)
    xyz = np.array([0.21, 0.07, 0.06])
    assert np.allclose(fr.xyz_to_twin(fr.xyz_to_dobot(xyz)), xyz, atol=1e-12)
    q = np.radians([10, 40, 30, 5])
    assert np.allclose(fr.joints_to_twin(fr.joints_to_dobot(q)), q, atol=1e-12)
