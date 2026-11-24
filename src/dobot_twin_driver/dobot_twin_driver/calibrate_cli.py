"""Hands-only calibration of the twin against the physical Dobot (sends NO motion commands).

  ros2 run dobot_twin_driver twin_calibrate [--port /dev/ttyUSB0] [--poses 6]
  ros2 run dobot_twin_driver twin_calibrate --fake          # self-test against the built-in simulated robot

You move the arm BY HAND (hold the unlock button on the forearm, release it at each pose) to several
different poses - include different rear-arm (J2) and forearm (J3) angles. The tool reads the robot's
own pose at each one and fits: J3 convention, radial/height offset of the reported Cartesian point, r definition.
"""
import argparse
import sys
import time

import numpy as np

from dobot_twin_kinematics.params import load_magician

from .fake_dobot import FakeDobot
from .frame_map import DEFAULT_CAL_PATH, fit_calibration
from .link import DobotLink
from .transport import open_serial


def read_avg(link, n=8, dt=0.03):
    ps = []
    for _ in range(n):
        p = link.get_pose()
        if p is not None:
            ps.append(p)
        time.sleep(dt)
    if len(ps) < n // 2:
        return None
    a = np.array([[p.x, p.y, p.z, p.r, p.j1, p.j2, p.j3, p.j4] for p in ps])
    spread = a.max(0) - a.min(0)
    return a.mean(0), float(spread[4:].max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--poses', type=int, default=6)
    ap.add_argument('--out', default=DEFAULT_CAL_PATH)
    ap.add_argument('--limits-profile', default=None)
    ap.add_argument('--probe', action='store_true', help='just print the live pose for 20 s (read-only connection test)')
    ap.add_argument('--fake', action='store_true', help='use the simulated robot (self-test; nothing is saved unless --save-fake)')
    ap.add_argument('--save-fake', action='store_true')
    a = ap.parse_args(argv)

    robot = load_magician(profile=a.limits_profile)
    fake = FakeDobot(robot) if a.fake else None
    link = DobotLink(fake if fake else open_serial(a.port or None, a.baud), timeout=0.3)
    name = link.get_device_name()
    if link.get_pose() is None:
        print('ERROR: no answer from the robot (check USB cable, power, port, and that you are in the "dialout" group).')
        return 2
    print(f'Connected{" to " + name if name else ""}. This tool will NOT move the robot.\n')
    if a.probe:
        print('Move the arm by hand (hold the unlock button) and watch the numbers change. Ctrl+C to quit.')
        t_end = time.time() + 20
        while time.time() < t_end:
            p = link.get_pose()
            if p:
                print(f'x={p.x:7.1f} y={p.y:7.1f} z={p.z:7.1f} r={p.r:6.1f} | J1 {p.j1:6.1f} J2 {p.j2:6.1f} J3 {p.j3:6.1f} J4 {p.j4:6.1f}', end='\r', flush=True)
            time.sleep(0.2)
        print()
        return 0

    auto = [[0, 10, 5, 0], [30, 40, 30, 20], [-40, 60, 20, -30], [10, 25, 60, 45], [50, 70, 10, 0], [-20, 15, 45, 10]]
    samples = []
    for k in range(a.poses):
        if fake:
            fake.drag_to(auto[k % len(auto)])
        else:
            input(f'Pose {k + 1}/{a.poses}: move the arm by hand to a NEW pose (vary J2 and J3), let go, '
                  f'keep it still, then press Enter... ')
        r = read_avg(link)
        if r is None:
            print('  could not read the pose, skipping')
            continue
        v, jitter = r
        if jitter > 0.3:
            print(f'  arm still moving (jitter {jitter:.2f} deg) - pose not used, hold it still')
            continue
        samples.append(((v[4], v[5], v[6], v[7]), (v[0], v[1], v[2]), v[3]))
        print(f'  read  x={v[0]:7.1f} y={v[1]:7.1f} z={v[2]:7.1f} r={v[3]:6.1f}   J=({v[4]:6.1f} {v[5]:6.1f} {v[6]:6.1f} {v[7]:6.1f}) deg')

    rep = fit_calibration(samples, robot)
    print('\nRESULT:', rep.message)
    if rep.details:
        for conv, d in rep.details.get('per_convention', {}).items():
            print(f'   J3 {conv:8s}: fit error {d["err"]:.2f} mm  (radial offset {d["dr"]:.1f}, height offset {d["dz"]:.1f})')
    if not rep.ok:
        print('Not saved. Motion from the twin stays disabled (read-only).')
        return 1
    if fake and not a.save_fake:
        print('(fake robot: not saved)')
        return 0
    rep.cal.save(a.out)
    print(f'Saved to {a.out}. Restart the bridge to use it.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
