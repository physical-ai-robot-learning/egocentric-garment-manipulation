# Replay-track stage map — calibration, tag pose, jaw aperture, robotization, comparison

*Moved from `scripts/README.md` in the 11-01 layout refactor; paths updated to the 11-05 layout
([refactor map](refactor_2026-11.md)). Paths below are relative to the repo root; modules run as `python -m <module>`.*

The 10-26/27 training-data pipeline is the packages `laundry_tracking`, `laundry_dataset`, `laundry_training` and
`laundry_deploy` ([codebase guide](codebase.md), [commands](pipeline.md#commands-on-real-data-stages-07)); it reuses the
calibration, tag pose and jaw modules below. Rows marked *replay track* are only used for open-loop replay. ~~Struck~~
rows moved to `archive/`.

Labels D0–D7 are the data stages of the archived Technical Report v0.1 (`report/archive/report_v0.1_en.tex`, Fig. 2); README steps 1–6 group them.

| stage | code |
|---|---|
| one-time setup: camera ↔ robot | `laundry_tracking.calib.robot_cam_solve` (the robot poses came from `visual_teleop.tools.robot_cam_poses`, which is not in this tree; see git history before 11-05) |
| one-time setup: UMI tag rig → jaw tip | `laundry_tracking.calib`: `tag_rig_solve`, `tag_rig_pivot`, `tag_rig_pivot_joint`, `tag_rig_touch_tcp`; `src/laundry_shared/rebot_prep/parity/`: `tag_rig_corners.py`, `tag_rig_imu.py`; `scripts/laundry/retarget/umi_align.py` |
| D0 → D1 → D2 → D3 tags → jaw-tip path in the robot base | `src/laundry_shared/rebot_prep/parity/tag_rig_pose.py`, `src/laundry_shared/rebot_prep/parity/tag_rig_fuse.py`, `laundry_tracking.calib.umi_tcp_traj` (also the tag input of `laundry_tracking`) |
| ~~D3 → D4 gap filling (MASt3R)~~ | ~~`wrist_m3_prep.py`, `umi_joint_handeye.py`, `umi_fuse_segments.py`, `umi_slam_eval.py`, `umi_fuse_slam.py`, `slam_vs_tag.py`, `umi_pipeline.sh`~~ → `archive/scripts/mast3r_gapfill/` (10-27; replaced by CoTracker fusion) |
| jaw aperture (keypoints) | `laundry_jaw`: `kpt_label` (draft + browser labelling), `kpt_train` (YOLO-pose per hand), `aperture` (tips → aperture g + state, writes umi_jaw.npz) |
| D4 → D5 robotization (*replay track*; training labels use `laundry_dataset.robotize_ik`) | `scripts/laundry/retarget/umi_robotize_bi.py` |
| D5 → D6 planner regression test (*replay track*) | `scripts/laundry/eval/regress_replay_plan.py` |
| D7 comparison (*replay track*) | `scripts/laundry/eval/replay_parity.py`, `scripts/laundry/eval/replay_side_by_side.py` |
| camera ↔ robot re-solve (10-26) | `laundry_tracking.calib.robot_cam_from_anchors`: camera ↔ robot re-solve after a camera move from the two fixed table tags (known robot coordinates) + the table depth plane; writes `robot_cam_v00N.json` (`configs/laundry/calibration/` in this repository) |

Recording (D0), jaw state, planning and replay (D5 → D7) run from `src/laundry_shared/visual_teleop/tools/`
(`tag_rig_record`, `umi_jaw`, `umi_replay_bi`); the robot bridge is `laundry_deploy.robot_service.mit_bridge`. The
bridge-start and ready-pose helpers of the replay era (`bridge_start`, `ready_pose`) are not in this tree.
