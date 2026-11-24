# Phase 3 verification (against the built-in simulated Dobot)

All checks below ran against `FakeDobot`, a byte-level simulated robot with a hidden J3 convention and hidden Cartesian offsets.
**Nothing here has been run on the real robot** - see PHASE3_GUIDE.md for the read-only first-contact procedure.

| Result | Test |
|---|---|
| PASSED | test_connect_and_poll_is_strictly_read_only |
| PASSED | test_all_commands_rejected_when_not_armed_or_not_calibrated |
| PASSED | test_no_data_state_and_no_arm_without_fresh_pose |
| PASSED | test_armed_xyz_move_arrives_and_twin_matches_command |
| PASSED | test_armed_line_and_joint_and_home |
| PASSED | test_unsafe_goals_rejected_and_nothing_sent |
| PASSED | test_speed_is_capped |
| PASSED | test_stop_always_works_and_halts_motion |
| PASSED | test_arm_timeout_stops_robot |
| PASSED | test_lost_pose_while_armed_stops_and_disarms |
| PASSED | test_alarm_blocks_arm_and_motion |
| PASSED | test_gripper_command_and_inversion[False] |
| PASSED | test_gripper_command_and_inversion[True] |
| PASSED | test_known_getpose_frame_bytes |
| PASSED | test_roundtrip_and_resync |
| PASSED | test_get_pose_matches_hidden_model |
| PASSED | test_timeout_when_disconnected |
| PASSED | test_alarms_read_and_clear |
| PASSED | test_queued_motion_only_runs_after_queue_start_and_uses_speed_ratio |
| PASSED | test_queue_stop_and_clear_halt_pending_moves |
| PASSED | test_calibration_recovers_hidden_truth[j1+j4-absolute] |
| PASSED | test_calibration_recovers_hidden_truth[j1+j4-relative] |
| PASSED | test_calibration_recovers_hidden_truth[j4-absolute] |
| PASSED | test_calibration_recovers_hidden_truth[j4-relative] |
| PASSED | test_calibration_refuses_similar_poses |
| PASSED | test_calibration_refuses_too_few_poses |
| PASSED | test_calibration_refuses_model_mismatch |
| PASSED | test_frame_roundtrip_with_offsets |

Key invariants covered: connecting/polling sends **zero write frames**; every command is rejected when not calibrated / not armed / stale / alarmed; unsafe goals send nothing; STOP works unarmed; ARM timeout and lost data stop the robot; speed is capped; calibration recovers the hidden J3 convention and offsets exactly and refuses similar, ambiguous or mismatching poses.
Also run: stubbed-ROS smoke test of the bridge node (connect, publish, arm, goal, reject, stop, hand-drag).
