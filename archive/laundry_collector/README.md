# laundry_collector v1 — bare-hand T-shirt folding collector

Fixed global RGB-D (Orbbec) + left/right wrist RGB (Arducam fisheye), **no IMU, no gripper**. Three decoupled stages:
`raw collector → preprocess/ (SLAM, world alignment, hand pose, virtual gripper) → export (P2)`. Raw episodes are never
modified by preprocessing (everything derived goes to `<episode>/processed/`), so algorithms can change after the pilot.

## Run
```
.venv/bin/python -m laundry_collector.collect                    # UI, legacy/configs/laundry/hardware_laundry_v1.yaml (sudo for the Orbbec on macOS)
.venv/bin/python -m laundry_collector.collect --mock             # UI on synthetic cameras
.venv/bin/python -m laundry_collector.collect --mock --headless-smoke 3
.venv/bin/python -m laundry_collector.qa <episode_dir>           # re-run the sync / quality report
.venv/bin/pytest tests/laundry -q
```
Keys: Space = Start/Stop, Esc = Cancel, Ctrl+S = Save, D = RGB/depth.

## Protocol (session.py)
IDLE → **Start** → INIT (hands on X_L / X_R, still `init_hold_s`) → MANIPULATION → **Stop** → END (hands off, still
`end_hold_s`) → REVIEW → **Save** (needs Success/Failure) or **Cancel** (moved to `_discarded/`, never deleted).
Phase boundaries (host clock) go to `events.jsonl` and `metadata.phases`; wrist motion during the holds is checked.

## Episode layout
```
<root>/<dataset>/episode_NNNNNN/
  global/{rgb/rgb.mp4, depth/NNNNNN.png (uint16, aligned to color), timestamps.csv}
  left_wrist/{rgb/rgb.mp4, timestamps.csv}   right_wrist/{...}
  calibration/{global_intrinsics, left_intrinsics, right_intrinsics, world_config, devices}.json
  metadata.json  events.jsonl  sync_report.{json,txt}  recorder_summary.json  .complete | .incomplete
```
`timestamps.csv`: video_frame, frame_index (device counter when the camera has one), camera_timestamp_ns (device clock),
host_timestamp_ns (monotonic, common clock) [+ depth_index, depth_timestamp_ns, depth_file]. Drops are counted from the
device frame number, else from frame intervals > 1.5 periods.

## Preprocess
| module | what |
|---|---|
| `preprocess/hand_pose.py` | `HandPoseProvider` interface, 21 landmarks in the wrist camera frame; WiLoR adapter = P2 |
| `preprocess/slam.py` + `preprocess_slam.py stage/submit/collect` | MASt3R-SLAM RGB-only on the 4090 (HRA runner), → `processed/<side>_camera_pose.npy` |
| `preprocess/world_align.py` | `WorldAligner` (init hold on X_L/X_R → table frame; scale from a `ScaleEstimator`), `HandToWorldTransformer`, `relative_actions` |
| `preprocess/gripper_frame.py` | pinch → `T_camera_gripper` + jaw (canonical frame below) |
| `preprocess/smoothing.py` | gap fill + zero-phase EMA (SLERP for rotation), never NaN |
| `preprocess/renderer.py` | primitive gripper mesh + `OpenCVRenderer` (fisheye/pinhole projection), debug overlay |
| `preprocess/virtual_gripper.py` | one wrist video → `processed/<side>/` overlay mp4 + `gripper_pose.{csv,npy}` + `T_camera_gripper.npy` + QA |

**Canonical gripper frame (both hands):** origin = thumb/index tip midpoint, +z = palm normal out of the palm, +y = wrist →
fingers, +x = y × z (jaw axis). The rendered gripper and the action labels read the same `T_camera_gripper` array.

## Before the pilot (open items)
- `legacy/configs/laundry/world_config.yaml`: measure X_L / X_R, `T_world_global`, `T_camera_tcp` per side.
- A metric `ScaleEstimator` for the SLAM trajectories (global RGB-D wrist tracking is the candidate).
- Wrist hand pose (WiLoR on undistorted wrist frames) → keypoints npz for `virtual_gripper.py`.
- `palm_normal_sign` in `gripper_frame.py` assumes MediaPipe/MANO landmark order; verify on the first real take (debug video).
