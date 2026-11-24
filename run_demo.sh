#!/usr/bin/env bash
# One-command launcher.   alias dobot='bash ~/Desktop/dobot_twin_ws/run_demo.sh'
#   dobot                  Phase 2 demo: simulated twin + control panel (no robot needed)
#   dobot fake             full system (sim twin + physical twin) with a built-in SIMULATED Dobot
#   dobot probe            read-only connection test to the real Dobot (prints live pose)
#   dobot calibrate        hands-only calibration against the real Dobot (needed before any motion)
#   dobot full             full system with the REAL Dobot (port auto-detected; port:=/dev/ttyUSB0)
#   dobot --rebuild [mode] rebuild first
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
cd "$WS"
if [ ! -f install/setup.bash ] || [ "$1" = "--rebuild" ]; then
  rm -rf build install log
  colcon build --symlink-install
  shift || true
fi
source install/setup.bash
MODE="$1"
case "$MODE" in
  full)      shift; exec ros2 launch dobot_twin_driver twin_full.launch.py "$@" ;;
  fake)      shift; exec ros2 launch dobot_twin_driver twin_full.launch.py fake:=true "$@" ;;
  probe)     shift; exec ros2 run dobot_twin_driver twin_calibrate --probe "$@" ;;
  calibrate) shift; exec ros2 run dobot_twin_driver twin_calibrate "$@" ;;
  *)         exec ros2 launch dobot_twin_kinematics twin_demo.launch.py "$@" ;;
esac
