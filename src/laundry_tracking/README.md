# laundry_tracking

Pose stage: the 6-DoF jaw-tip pose of both hand-worn grippers in the robot base frame, from the fixed RGB-D camera
(point tracking anchored by the AprilTag markers, IMU only inside tracking gaps), plus camera ↔ robot calibration.

**Status:** current (fusion variant **Cg**).

## Inputs → Outputs
A recorded take (`EGO_DATA_ROOT/datasets/tag_rig/<take>`: Orbbec colour + depth, wrist cameras, IMUs, tag detections) →
per-demo fused trajectories, gates and grades, QA videos, and the derived take used by `laundry_dataset`.

## Layout (in pipeline order)
| path | what |
|---|---|
| `calib/` | camera ↔ robot (`robot_cam_solve`, `robot_cam_from_anchors` after a camera move), tag rig (`tag_rig_solve`, `tag_rig_pivot*`, `tag_rig_touch_tcp`), `umi_tcp_traj` (tag pose per take), `umi_side_check` |
| `new_take_prep` | arm assignment and trims of a new take |
| `cotrack/` | `make_pack` → `run_cotracker` (GPU) → `point_solver` (3-D) → `gate` → `episode_quality` (GOLD / SILVER / REJECT), `out_of_view` |
| `fusion/` | `trajectory_fusion` (Cg: CoTracker + markers, IMU acceleration only inside gaps), `make_derived`, `fusion_metrics`, `render_qa`, `render_sources` |
| `tighten_trim`, `edge_gap_trim`, `intervals`, `whole_demo_stitch` | trims and stitching |
| `orbbec_tracker`, `orbbec_candidate`, `arm_rotation`, `imu_motion`, `tag_rotations`, `slam_consistency`, `camera_calibration` | building blocks of the above |
| `fgo/` | factor-graph pose smoother (`arm_data`, `pose_smoother`) |
| `workdirs`, `constants` | work-folder layout under `EGO_FUSION_WORK`, shared constants |

## Entry points
```bash
python -m laundry_tracking.calib.robot_cam_solve --synthetic          # self-test (CI)
python -m laundry_tracking.new_take_prep <take>
python -m laundry_tracking.cotrack.make_pack <pack> <take>:<demo> ...
python -m laundry_tracking.fusion.trajectory_fusion <pack> Cg --ct solve3
```
Tests: `src/laundry_tracking/tests/` (synthetic world, golden outputs, `--help` of every CLI).

## Provenance
Was `src/garment_pipeline/fusion_stack/` + `fgo_tools/` + `scripts/pose/` (11-04 OA refactor; module map in the refactor log).
