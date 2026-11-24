# Dobot Magician Digital Twin (ROS 2 Humble + RViz2)

Workspace layout (`src/`):

| Package | Role | Status |
|---|---|---|
| `dobot_description` | **Your original package, byte-identical** (sha256 manifest: `ORIGINAL_SHA256.txt`). Single source of truth for geometry. | unmodified |
| `dobot_twin_description` | Twin presentation layer: sim/real instances, RViz2 config, labels, model verification | **Phase 1 done** |
| `dobot_twin_kinematics` | FK / IK, Cartesian X-Y-Z, limit profiles, simulated motion control, control panel | **Phase 2 done** |
| `dobot_twin_driver` | Physical Dobot bridge: serial protocol, real-time joint sync, hands-only calibration, guarded commands, fake robot | **Phase 3 done** (hardware-untested) |

## Sim vs physical are separated by construction
| | simulated twin | physical twin |
|---|---|---|
| namespace | `/sim` | `/real` |
| TF frames | `sim/<link>` | `real/<link>` |
| joint states | `/sim/joint_states` | `/real/joint_states` (driver only) |
| RViz | translucent (alpha 0.6) + blue label | opaque + orange label |

## Quick demo (shortest way)
```bash
bash ~/Desktop/dobot_twin_ws/run_demo.sh            # builds on first run, then RViz + control panel
bash ~/Desktop/dobot_twin_ws/run_demo.sh --rebuild  # after updating files
```
Make it a one-word command (once): `echo "alias dobot='bash ~/Desktop/dobot_twin_ws/run_demo.sh'" >> ~/.bashrc`
then open a new terminal and type `dobot`.
Needs once: `sudo apt install python3-tk`.

## Phase 3: physical robot (see `src/dobot_twin_driver/docs/PHASE3_GUIDE.md`)
```bash
dobot fake        # full system with a built-in simulated Dobot (no hardware)
dobot probe       # read-only test of the real robot, prints live pose
dobot full        # sim twin (blue) + physical twin (orange) + panels, real robot
dobot calibrate   # hands-only calibration (required before any motion)
```

## Phase 2: kinematics + simulated control (no robot needed)
Topics (all under `/sim`, simulation only): `cmd/goal_xyz` (joint move), `cmd/line_xyz` (straight line),
`cmd/jog` (Twist, dead-man), `cmd/joints`, `cmd/gripper` (0..0.0135 m), `cmd/home`, `cmd/stop`;
outputs `joint_states`, `tool_pose`, `status`, `markers`.
```bash
ros2 topic pub --once /sim/cmd/goal_xyz geometry_msgs/msg/PointStamped "{point: {x: 0.22, y: 0.05, z: 0.06}}"
```
Joint limits: `config/limits.yaml` (profiles `conservative` default, `urdf`, `datasheet`);
choose with `limits_profile:=datasheet`. Verification: `docs/PHASE2_VERIFICATION.md`.

## Run (Phase 1 only)
```bash
cd dobot_twin_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch dobot_twin_description twin_model.launch.py                 # sim + sliders
ros2 launch dobot_twin_description twin_model.launch.py instance:=both  # side by side
ros2 launch dobot_twin_description twin_model.launch.py instance:=both overlay:=true
```
Sliders drive `joint_1..4` and `joint_prismatic_l`; mimic joints are resolved by robot_state_publisher.
If only `sim` is launched, the "Physical twin" display shows a harmless "no description" warning.

## Verify the model (no ROS needed)
```bash
cd src/dobot_twin_description/tools
python3 verify_model.py --pkg ../../dobot_description --checksums ../../../ORIGINAL_SHA256.txt \
        --report ../docs/PHASE1_VERIFICATION.md
python3 render_model.py --pkg ../../dobot_description --out ../docs/images   # offline renders
colcon test --packages-select dobot_twin_description
```
Requires numpy, scipy, matplotlib. On a ROS machine `verify_model.py` also cross-checks the real `xacro`.

## Known issues in the ORIGINAL model (left untouched on purpose)
1. `joint_prismatic_r` limit has lower (0) > upper (-0.0135).
2. `joint_mimic_1/2` declare [0, 360 deg] but need [-90, 5] / [-70, 15] deg. Harmless in robot_state_publisher
   (no clamping); would break the level wrist in loaders that clamp.
3. URDF limits differ from the datasheet (see `docs/PHASE1_VERIFICATION.md`).
The twin never relies on these declarations: it publishes only the 5 actuated joints.
