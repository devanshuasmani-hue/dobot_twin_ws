# Phase 2 notes

* Cartesian position = gripper fingertip centre (TCP), metres, robot root frame (origin on the J1 axis at table
  level, X forward, Y left, Z up). `tcp_dz` (0.1048 m) was measured from the gripper meshes. The physical Dobot's own
  X/Y/Z origin is different; a fixed offset is calibrated in Phase 3 by reading the robot's pose at a known joint state.
* Joint convention = original URDF (J3 relative to the rear arm). `j3_convention` in limits.yaml selects how the
  physical robot's J3 is interpreted in Phase 3 (relative / absolute).
* Elbow solution: the one inside the limits closest to the current pose (continuity).
* Every command is validated (reach, limits, floor) before motion; a straight-line move is checked along its
  whole path (2 mm samples) and rejected up front if any point fails.
* Not modelled: collisions with the base/table beyond the floor plane, dynamics, payload.
