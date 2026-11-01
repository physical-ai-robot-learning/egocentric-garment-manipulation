# processing — demonstration → training data (2026-10-26/27 flattening pipeline)

The current offline pipeline that turns fixed-camera + wrist + IMU recordings into fused jaw-tip trajectories, quality grades, Cartesian training labels and a LeRobot dataset.

**Status:** current (snapshot record — see Provenance)

## Inputs → Outputs
Recorded takes (`tag_rig_record`: Orbbec RGB-D, wrist fisheyes, IMUs) + tag poses from `scripts/pose/` → fused trajectories, GOLD / SILVER / REJECT grades, labels A / B, LeRobot v3 dataset, offline eval and deploy.

## Entry points
The stage table below (paths relative to this folder unless they start with `scripts/` or `src/`).

## Layout
- `fusion_stack/` — CoTracker + tag + IMU pose fusion, gates, trims, quality, QA renders
- `fgo_tools/` — data loaders and factor-graph helpers shared by every stage
- `labels_dataset/` — robotize/IK labels, IK QA, idle trim, LeRobot export
- `training/` — DiT trainer (jaw BCE head, masks, per-episode weights) and its configs (added 11-01)
- `eval_deploy/` — offline policy eval, receding-horizon deploy core + UI, X-VLA server, height calibration, camera tools

## Provenance
Moved whole from `processing/` in the 11-01 layout refactor (internal relative paths unchanged). Synced 11-01 with the 10-28 → 11-01 code (`training/` and the new `eval_deploy/` files added).

Snapshot of the offline pipeline that turned 300 successful T-shirt **flattening** demonstrations into robot training
labels. The code is copied as it runs (absolute home paths, same module layout); it is a record of the method, not a
packaged library.

| stage | where it runs | code |
|---|---|---|
| 1 · success judging | Mac, by eye | frame at **END − 0.5 s** of every demo (the operator re-crumples the shirt right after Space) |
| 2 · jaw state | GPU (RunPod) + Mac MPS | demos before 10-29: `src/visual_teleop/tools/umi_jaw.py infer` (ResNet18 v003 classifier) → `scripts/pose/jaw_aperture.py` (YOLO-pose jaw tips → aperture g); demos from 10-29: Feetech servo → `scripts/pose/jaw_from_motor.py` (same `umi_jaw.npz` format) |
| 3 · tag pose | Mac | `scripts/pose/umi_tcp_traj.py` (rig tags → TCP, IMU fills rotation), per-take camera ↔ robot calibration `fusion_stack/camcal.py` (`configs/calibration/robot_cam_v001…v005.json`, re-solved with `scripts/pose/robot_cam_from_anchors.py` = fixed tags 400/401 + table plane) |
| 4 · prep | Mac | arm assignment (rig y side; labels are swapped since 10-22), idle-trim rows + start seeds `fusion_stack/orbbec/v2b/new_take_prep.py`, tag rotation `fgo_tools/tag_rot_extract_v002.py`, Orbbec tracker `fusion_stack/orbbec/v2b/track_v31.py` |
| 5 · CoTracker | GPU (RunPod RTX 5090) | `fusion_stack/orbbec/v2b/cotrack/make_pack2.py` (40 query points per rig) → `run_cotracker.py` (CoTracker3 online) |
| 6 · 3-D + gates | CPU (Mac or RunPod) | `cotrack/solve2.py` (depth lift, rigid rig fit, hidden-tag hold-out) → `cotrack/gate.py` (G1 IMU-vs-track acceleration, G2 left/right collision < 8 cm, G3 tag disagreement > 10 cm) |
| 7 · fusion | CPU | `fusion/trajectory_fusion_v1.py` variant **Cg** (CoTracker + tag anchors + constant bias; IMU acceleration only inside gaps; rotation from tags + gyro) |
| 8 · trims | Mac | `orbbec/v2b/tighten_trim.py` (keep first fused move > 10 cm/s − 0.3 s … last + 0.3 s), `orbbec/v2b/edge_gap_trim.py` (cut a ≥ 0.8 s one-arm tracking gap touching the start / end) |
| 9 · quality | Mac | `cotrack/episode_quality.py` → GOLD / SILVER / REJECT; videos `fusion/render_fusion_qa.py`, source-coloured trajectory GIF `fusion/render_traj_sources.py` |
| 10 · labels | Mac | `labels_dataset/robotize_ik.py`: **A** = Cartesian target before IK (fusion + reBot EEF axes + 0.25 s smoothing + ~~table clearance~~) — superseded 10-29 by **F** (`--label fused`, measured human jaw tip, no clamp); **B** = IK → FK (QA / comparison); IK diagnostics per frame; `labels_dataset/ik_qa_stats.py` splits large IK residuals by cause |
| 11 · dataset | Mac | `labels_dataset/tag_rig_unsplit.py --label fused --mask-mode stitched [--speed 0.3]` (~~`--label rob`~~ before 10-29) → LeRobot v3, 20 Hz, 3 cameras, state 30-D, action 20-D absolute (relative `inv(T_t)·T_t+k` is built in the training loader) |
| 11b · training | GPU | `training/train_dit_tagrig.py` + `training/tagrig_common.py` (jaw BCE, masks, episode weights; `GRAD_CLIP=0`), configs `training/dit_tagrig_v1.yaml`, `act_tagrig_v1.yaml` |
| 12 · eval / deploy | GPU / robot | `eval_deploy/tagrig_offline_eval.py` (k16/k32, zero-motion baseline, grasp / free and episode-phase breakdown), `eval_deploy/act_tagrig_deploy.py` + `act_tagrig_core.py` + `act_tagrig_ui.py` (receding-horizon deploy and its UI; refuses GO unless the bridge limits the left jaw opening), `xvla_server.py`, `height_calib.py` (per-arm table plane from tag touches, not yet validated on the robot) |

`fusion_stack/` = the `_imu_consistency` tree, `fgo_tools/` = `_fgo_1023/tools` (data loaders shared by every stage).
