import numpy as np
import pytest

from dobot_twin_driver.core import CoreConfig, RealTwinCore
from dobot_twin_driver.fake_dobot import FakeDobot
from dobot_twin_driver.frame_map import Calibration, DobotFrame
from dobot_twin_driver.link import DobotLink
from dobot_twin_driver.protocol import ID, PTPMode
from dobot_twin_kinematics.params import load_magician

TRUTH = Calibration('absolute', dr_mm=-12.0, dz_mm=35.0, r_mode='j1+j4', calibrated=True)
WRITE_IDS = {ID.PTP_CMD, ID.QUEUE_START, ID.QUEUE_STOP, ID.QUEUE_CLEAR, ID.PTP_COMMON_PARAMS}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def build(calibrated=True, cfg=None, **fake_kw):
    robot = load_magician()
    clk = Clock()
    fake = FakeDobot(robot, truth=TRUTH, time_fn=clk, **fake_kw)
    link = DobotLink(fake, timeout=0.2)
    cal = Calibration(**{**TRUTH.__dict__, 'calibrated': calibrated})
    core = RealTwinCore(link, robot, cal, cfg or CoreConfig(), clock=clk)
    return core, fake, clk, robot


def n_writes(fake):
    return sum(1 for l in fake.log if int(l.split()[1].split('=')[1]) & 1)


def test_connect_and_poll_is_strictly_read_only():
    core, fake, clk, _ = build()
    fake.drag_to([15, 30, 40, 5])
    for _ in range(30):
        assert core.poll()
        clk.t += 0.04
    assert n_writes(fake) == 0                                   # not a single write frame was sent
    names, pos = core.twin_joint_state()
    assert np.allclose(np.degrees(pos[:2]), [15, 30], atol=1e-3)
    assert np.degrees(pos[2]) == pytest.approx(40 - 30, abs=1e-3)   # robot J3 is absolute -> URDF relative = J3 - J2
    assert 'read-only' in core.status_text()


def test_all_commands_rejected_when_not_armed_or_not_calibrated():
    core, fake, clk, _ = build()
    core.poll()
    assert not core.command_xyz([0.2, 0, 0.05]).ok and 'not armed' in core.last_error
    assert not core.command_joints([0, 0.1, 0.1, 0]).ok
    assert not core.command_gripper(True).ok
    assert not core.home().ok
    core2, fake2, _, _ = build(calibrated=False)
    core2.poll()
    assert not core2.arm().ok and 'not calibrated' in core2.last_error
    assert n_writes(fake) == 0 and n_writes(fake2) == 0


def test_no_data_state_and_no_arm_without_fresh_pose():
    core, fake, clk, _ = build()
    assert core.twin_joint_state() is None and 'NO DATA' in core.status_text()
    assert not core.arm().ok
    core.poll()
    assert core.twin_joint_state() is not None
    clk.t += 1.0                                                 # stale
    assert core.twin_joint_state() is None


def run_until_idle(core, fake, clk, seconds=10.0, dt=0.05):
    for _ in range(int(seconds / dt)):
        clk.t += dt
        core.poll()


def test_armed_xyz_move_arrives_and_twin_matches_command():
    core, fake, clk, robot = build()
    core.poll()
    assert core.arm().ok and core.armed
    target = np.array([0.22, 0.06, 0.07])
    assert core.command_xyz(target, line=False).ok
    run_until_idle(core, fake, clk, 12)
    tcp = core.tcp_twin()
    assert np.allclose(tcp, target, atol=1e-4)                   # the PHYSICAL twin reaches the commanded point
    assert core.cfg.max_speed_pct >= fake.vel_ratio == pytest.approx(20.0)
    # and the robot's own cartesian reading is the commanded point in ITS frame (offsets applied)
    p = fake.pose()
    exp = DobotFrame(robot, TRUTH).xyz_to_dobot(target)
    assert np.allclose([p.x, p.y, p.z], exp, atol=0.1)


def test_armed_line_and_joint_and_home():
    core, fake, clk, robot = build()
    core.poll()
    core.arm()
    assert core.command_joints(np.radians([20, 30, 10, 0])).ok
    run_until_idle(core, fake, clk, 12)
    assert np.allclose(core.twin_joint_state()[1][:3], np.radians([20, 30, 10]), atol=1e-4)
    assert core.command_xyz([0.23, -0.05, 0.08], line=True).ok
    run_until_idle(core, fake, clk, 12)
    assert np.allclose(core.tcp_twin(), [0.23, -0.05, 0.08], atol=1e-4)
    assert core.home().ok
    run_until_idle(core, fake, clk, 15)
    assert np.allclose(core.twin_joint_state()[1][:4], 0, atol=1e-4)


def test_unsafe_goals_rejected_and_nothing_sent():
    core, fake, clk, _ = build()
    core.poll()
    core.arm()
    w = n_writes(fake)
    for bad in ([0.22, 0.0, 0.005],          # inside the 10 mm safety margin above the table
                [0.22, 0.0, -0.02],          # below the table
                [0.45, 0.0, 0.1],            # out of reach
                [-0.2, 0.01, 0.05]):         # behind the base (J1 limit)
        assert not core.command_xyz(bad).ok, bad
        assert not core.command_xyz(bad, line=True).ok, bad
    assert not core.command_joints(np.radians([0, 85, 0, 0])).ok       # tool through the table
    assert n_writes(fake) == w


def test_speed_is_capped():
    core, fake, clk, _ = build(cfg=CoreConfig(speed_pct=20, max_speed_pct=40))
    core.poll()
    core.set_speed(100)
    core.arm()
    assert fake.vel_ratio == 40.0


def test_stop_always_works_and_halts_motion():
    core, fake, clk, _ = build()
    core.poll()
    core.arm()
    core.command_joints(np.radians([60, 40, 20, 0]))
    clk.t += 1.0
    core.poll()
    j_mid = core.pose.j1
    assert core.stop().ok and not core.armed
    run_until_idle(core, fake, clk, 10)
    assert core.pose.j1 == pytest.approx(j_mid, abs=1e-3) and j_mid < 60
    core3, fake3, _, _ = build()
    assert core3.stop().ok                                            # works even when never armed


def test_arm_timeout_stops_robot():
    core, fake, clk, _ = build(cfg=CoreConfig(arm_timeout_s=5))
    core.poll()
    core.arm()
    clk.t += 6
    core.poll()
    assert not core.armed and 'timeout' in core.last_error
    assert not core.command_xyz([0.22, 0, 0.06]).ok


def test_lost_pose_while_armed_stops_and_disarms():
    core, fake, clk, _ = build()
    core.poll()
    core.arm()
    fake.connected = False
    clk.t += 1.0
    core.poll()
    assert not core.armed and 'lost' in core.last_error
    assert 'NO DATA' in core.status_text()


def test_alarm_blocks_arm_and_motion():
    core, fake, clk, _ = build(alarms=[5])
    core.poll()
    assert not core.arm().ok and 'alarm' in core.last_error
    assert core.clear_alarms().ok
    core.poll(); core.n_polls = 0; core.poll()
    assert core.arm().ok


@pytest.mark.parametrize('inv', [False, True])
def test_gripper_command_and_inversion(inv):
    core, fake, clk, _ = build(cfg=CoreConfig(gripper_inverted=inv))
    core.poll(); core.arm()
    assert core.command_gripper(open_=False).ok                      # request CLOSE
    assert fake.gripped == (not inv)                                 # inverted robots use the opposite bit
    names, pos = core.twin_joint_state()
    assert pos[4] == 0.0                                             # twin shows closed
    assert core.command_gripper(open_=True).ok
    assert core.twin_joint_state()[1][4] == pytest.approx(0.0135)


def test_pump_off_is_immediate_and_allowed_when_disarmed():
    core, fake, clk, _ = build()
    core.poll()
    assert not core.armed
    assert fake.pump_on is True
    assert core.pump_off().ok
    assert fake.pump_on is False
