# pose

Calibration and pose scripts: camera ↔ robot, UMI tag rig → jaw tip, AprilTag + depth + IMU jaw-tip paths in the robot
base, and the jaw aperture (the gripper DoF) — from the jaw servo for demos since 10-29, from the wrist video before.

**Status:** current (tag pose and jaw aperture feed `scripts/processing/`; the open-loop replay track uses them directly)

## Inputs → Outputs
Recorded takes `datasets/tag_rig/<take>` (Orbbec RGB-D + tag detections, wrist video, IMU, jaw servo `gripper_motor_{left,right}.npz` since 10-29) →
`configs/calibration/robot_cam_v00N.json`, `umi_tag_rig_v00N.json`, `jaw_kpt_*` models / calibration, and per take
`umi_traj_{right,left}.npz` (jaw-tip TCP in the robot base) and `umi_jaw.npz` (aperture g, state).

## Entry points
```bash
python scripts/pose/robot_cam_solve.py datasets/tag_rig/<take> datasets/robot_cam/<run> [--write]   # first camera ↔ robot solve
python scripts/pose/robot_cam_solve.py --synthetic                                                  # self-test
python scripts/pose/robot_cam_from_anchors.py datasets/tag_rig/<take> --frames 2400:2828:4           # re-solve after a camera move
python scripts/pose/tag_rig_solve.py datasets/tag_rig/<ts>                                          # tag rig geometry
python scripts/pose/tag_rig_pivot.py datasets/tag_rig/<pivot take> --rig datasets/tag_rig/<rig take>/tag_rig.json [--write]
python scripts/pose/tag_rig_pivot_joint.py datasets/tag_rig/<take1> <take2> ... [--write]
python scripts/pose/tag_rig_touch_tcp.py datasets/tag_rig/<corners take> [--write]
python scripts/pose/umi_tcp_traj.py datasets/tag_rig/<take> --hand <hand> --depth --imu-fill         # tags → jaw-tip path
python scripts/pose/jaw_kpt_label.py select datasets/tag_rig/20261022_105200                     # jaw-tip labelling
python scripts/pose/jaw_kpt_label.py serve datasets/jaw_kpt/20261022_105200
python scripts/pose/jaw_kpt_train.py datasets/jaw_kpt/20261022_105200 --hand left                 # YOLO-pose per hand
python scripts/pose/jaw_aperture.py datasets/tag_rig/<take>/wrist                                   # tips → aperture g
python scripts/pose/jaw_from_motor.py datasets/tag_rig/<take>                                       # servo → umi_jaw.npz (10-29 →)
python scripts/pose/jaw_motorlike.py eval <model_dir> --hand left <take> ...                         # video model trained on the servo (evaluation only)
python scripts/pose/umi_side_check.py                                                               # which camera / servo / IMU is on which hand
PYTHONPATH=src python -m rebot_prep.parity.tag_rig_corners datasets/tag_rig/<corners take> --board-tag 0.024
PYTHONPATH=src python -m rebot_prep.parity.tag_rig_imu datasets/tag_rig/<take> [--write]
```
Each script inserts `<repo>/src` into `sys.path` itself, so no install is needed for these.

## Layout
- `robot_cam_*.py` — camera ↔ robot base calibration
- `tag_rig_*.py`, `umi_tcp_traj.py` — tag rig calibration and the per-take jaw-tip path (shared code in `src/rebot_prep/parity/`)
- `jaw_kpt_label.py`, `jaw_kpt_train.py`, `jaw_aperture.py` — jaw aperture. Kept here (not in `dataset/`) because the
  aperture is a state estimate of the gripper DoF from the wrist camera, produced per take like the arm pose.
  `jaw_aperture.py` imports `crop` from its sibling `jaw_kpt_label.py`. The hand labels used for the 10-22 models are in
  [`docs/records/jaw_kpt_labels_20261022_105200/`](../../docs/records/jaw_kpt_labels_20261022_105200/).
- `jaw_from_motor.py` — **current jaw label for demos since 10-29**: the UMI jaw servo (Feetech STS3215, torque off,
  logged ≈100 Hz by `tag_rig_record`) → per-hand calibration → `g_cont`, hysteresis events, per-take re-centring;
  writes the same `umi_jaw.npz` as the video path (+ `motor_pos`).
- `jaw_motorlike.py`, `jaw_motorlike_overlay.py` — a wrist-video model trained to reproduce the servo, for relabelling
  older video-only demos; evaluated, **not applied** to any dataset.
- `umi_side_check.py` — checks which wrist camera, jaw servo and IMU belong to which hand before recording.

## Provenance
Moved from flat `scripts/` in the 11-01 layout refactor; servo jaw tools added from the 10-29 → 11-01 work; the old stage table is [docs/pipeline_replay_track.md](../../docs/pipeline_replay_track.md).
