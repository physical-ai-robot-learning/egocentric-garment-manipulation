# Replay-track stage map — calibration, tag pose, jaw aperture, robotization, comparison

*Moved from `scripts/README.md` in the 11-01 layout refactor. Script paths below are relative to the repo root.*

The 10-26/27 training-data pipeline lives in [`scripts/processing/`](../scripts/processing/README.md); it reuses the calibration, tag pose and
jaw scripts below. Rows marked *replay track* are only used for open-loop replay. ~~Struck~~ rows moved to `legacy/`.

Labels D0–D7 are the data stages of the technical report (Fig. 2); README steps 1–6 group them.

| stage | script |
|---|---|
| one-time setup: camera ↔ robot | `scripts/pose/robot_cam_solve.py` (poses from `visual_teleop.tools.robot_cam_poses`) |
| one-time setup: UMI tag rig → jaw tip | `scripts/pose/`: `tag_rig_solve.py`, `tag_rig_pivot.py`, `tag_rig_pivot_joint.py`, `tag_rig_touch_tcp.py`; `src/rebot_prep/parity/`: `tag_rig_corners.py`, `tag_rig_imu.py`; `scripts/retarget/umi_align.py` |
| D0 → D1 → D2 → D3 tags → jaw-tip path in the robot base | `src/rebot_prep/parity/tag_rig_pose.py`, `src/rebot_prep/parity/tag_rig_fuse.py`, `scripts/pose/umi_tcp_traj.py` (also the tag input of `scripts/processing/`) |
| ~~D3 → D4 gap filling (MASt3R)~~ | ~~`wrist_m3_prep.py`, `umi_joint_handeye.py`, `umi_fuse_segments.py`, `umi_slam_eval.py`, `umi_fuse_slam.py`, `slam_vs_tag.py`, `umi_pipeline.sh`~~ → `legacy/scripts/mast3r_gapfill/` (10-27; replaced by CoTracker fusion) |
| jaw aperture (keypoints) | `scripts/pose/`: `jaw_kpt_label.py` (draft + browser labelling), `jaw_kpt_train.py` (YOLO-pose per hand), `jaw_aperture.py` (tips → aperture g + state, writes umi_jaw.npz) |
| D4 → D5 robotization (*replay track*; training labels use `scripts/processing/labels_dataset/robotize_ik.py`) | `scripts/retarget/umi_robotize_bi.py` |
| D5 → D6 planner regression test (*replay track*) | `scripts/evaluation/regress_replay_plan.py` |
| D7 comparison (*replay track*) | `scripts/evaluation/replay_parity.py`, `scripts/evaluation/replay_side_by_side.py` |
| camera ↔ robot re-solve (10-26) | `scripts/pose/robot_cam_from_anchors.py`: camera ↔ robot re-solve after a camera move from the two fixed table tags (known robot coordinates) + the table depth plane; writes `configs/calibration/robot_cam_v00N.json` |

Recording (D0), jaw state, planning and replay (D5 → D7) run from `src/visual_teleop/tools/` (`tag_rig_record`, `umi_jaw`, `umi_replay_bi`, `bridge_start`, `ready_pose`).

