# Codebase guide: packages, interfaces, coordinate representation

*The component table, installation notes and "The split" were rewritten on 2026-11-05 for the current layout; the
interface tables and the coordinate representation were moved from the README with only path fixes. Old → new paths:
[refactor_2026-11.md](refactor_2026-11.md).*

## Packages and folders, by pipeline stage

Stages as in [pipeline.md](pipeline.md#commands-on-real-data-stages-07). Every module with a command line runs as
`python -m <package>.<module>` and has `main(argv) -> int`; every package has characterization tests in
`src/<package>/tests/`.

| package / folder | stage | what is in it | status |
|---|---|---|---|
| [`src/laundry_common/`](../src/laundry_common/) | all | `paths`: every machine-specific root (`EGO_DATA_ROOT`, `EGO_FUSION_WORK`, `EGO_TRAIN_ROOT`, `EGO_CKPT_ROOT`, `TAGRIG_STATE_DIR`, …) with environment-variable overrides; the defaults are the paths of the recording machine | current |
| [`src/laundry_tracking/`](../src/laundry_tracking/) | 0, 2–4 | `calib/` (camera ↔ robot `robot_cam_solve` / `robot_cam_from_anchors`, tag rig `tag_rig_solve` / `tag_rig_pivot` / `tag_rig_touch_tcp`, tags → jaw-tip path `umi_tcp_traj`); `new_take_prep`; `cotrack/` (`make_pack`, `run_cotracker`, `point_solver`, `gate`, `episode_quality`); `fusion/` (`trajectory_fusion` variant Cg, `make_derived`, `render_qa`, `render_sources`); trims `tighten_trim`, `edge_gap_trim`; `fgo/` (data loaders, pose smoother) | current (CoTracker needs a CUDA GPU) |
| [`src/laundry_jaw/`](../src/laundry_jaw/) | 2 | jaw state: `from_motor` (servo → `umi_jaw.npz`, the source since 10-29), `aperture` / `kpt_label` / `kpt_train` (jaw-tip keypoints in the wrist video), `motorlike` (+ `motorlike_overlay`, relabel attempt, not used for any reported model) | current |
| [`src/laundry_dataset/`](../src/laundry_dataset/) | 5 | labels + LeRobot export: `robotize_ik` (labels F / A / B + IK diagnostics; IK in a sub-process, `EGO_IK_PYTHON`), `tag_rig_unsplit` (the dataset builder), `tag_rig_to_lerobot`, `idle_trim`, `ik_qa_stats`; `builds/` (split / audit / upload / QA-render tools of the 11-02 → 11-05 flattening sets) | current |
| [`src/laundry_grasp/`](../src/laundry_grasp/README.md) | 5b | grasp held / empty labels from the wrist camera: `events` (jaw close events), `crops`, `model` (ResNet-18 recipe), `pgrasp` (causal window), `generate` (frozen `grasp-v2-W0.4`), `sidefile`, `gold_ui`, `cv`, `render` | current (frozen generator) |
| [`src/laundry_training/`](../src/laundry_training/) | 6 | DiT / ACT trainer on the LeRobot fork: `env_options` (all switches: `GRIP_STATE_MASK`, `GRIP_BCE`, `GRAD_CLIP`, `MASK_CROSS_GAP`, `AUG_*`, `PGRASP`), `datasets`, `loss_masks`, `augment`, `grip_head`, `train_dit_tagrig`, `train_dit_mix` + configs | current (needs the LeRobot fork of the training nodes) |
| [`src/laundry_deploy/`](../src/laundry_deploy/) | 7 | receding-horizon deploy: `core/` (policy / X-VLA / label runners, chunks, real-time chunking, kinematics, collision, height calibration), `session/` (run loop, streamer, cameras, CLI), `ui/` (Flask UI :8770), `tools/` (offline eval, height calibration, camera align / viewer, …), `robot_service/` (reBot MIT bridge + jaw / joint-zero tools), `xvla_server` | current (robot + reBot vendor plugin for anything that moves) |
| [`src/laundry_reference/`](../src/laundry_reference/README.md) | 3, 5, 9 | small standalone re-implementations of the trajectory stages (trajectory CSV, gap filling, relative SE(3), robotization, parity metrics, plot) that run on the synthetic data; not used by the real pipeline | current (reference) |
| [`src/laundry_shared/`](../src/laundry_shared/) | 1, 2, replay, 7 | copies of the modules the laundry code imports from the HandUMI / teleop work, in their original style: `visual_teleop` (recorder `tag_rig_record`, jaw classifier `umi_jaw`, reBot IK / limits, replay planner `umi_replay_bi`), `rebot_prep.parity` (tag-rig pose), `handumi_collector`, `ego_collector`, `ego_teleop`, `contracts`, `xvla_core`, `umi_to_lerobot` ([provenance](provenance.md)) | current (live teleoperation parked) |
| [`scripts/laundry/`](../scripts/laundry/) | 3, 6, replay | operational entry points: `eval/` (replay parity, side-by-side video, planner regression, per-epoch offline eval of cluster runs), `retarget/` (UMI ↔ reBot EEF alignment, two-hand robotization), `ray_cotracker/` and `ray_ditddp/` (cluster jobs), `runpod_convert/` (the whole conversion chain on a rented pod), `grasp/` (11-05 grasp held / empty labels and p_grasp side files; research scripts, not yet in OA style) | current |
| [`configs/laundry/`](../configs/laundry/README.md) | 0 | calibration results, robot settings and overrides, jaw models, regression baseline, HandUMI profiles, pinned environments (`envs/`) | current |
| [`tests/laundry/`](../tests/laundry/) | – | layout / import smoke tests, the scripts' synthetic self-tests, deploy UI routes, reference regression (`expected_synthetic.json`) | current |
| [`assets/laundry/examples/`](../assets/laundry/examples/README.md) | – | SYNTHETIC bimanual demo + simulated replay (CSV / NPZ) and their generator | current |
| [`hardware/umihand/`](../hardware/umihand/README.md) | 1 | HandUMI build copy + our modified parts, AprilTag sheets, IMU firmware (upstream design by BrikHMP18, Apache-2.0) | current |
| [`analysis/`](../analysis/verify_claims.py) | – | `verify_claims.py`: every number of the technical report against the document it was taken from | current |
| [`report/`](../report/README.md) | – | technical report, English + Korean (PDF + LaTeX + figures); earlier versions in `report/archive/` | current |
| [`docs/`](../docs/) | – | this guide, [pipeline](pipeline.md), [history](history.md), [project details](details.md), [results](results.md), [replay-track stage map](pipeline_replay_track.md), [environments](environments.md), [provenance](provenance.md), dated status notes, [records](records/README.md) | current |
| [`media/`](../media/README.md) | – | GIFs, videos and photos of the real setup | current |
| [`archive/`](../archive/README.md) | – | MASt3R gap filling and runner (moved 10-27), ORB-SLAM3, UMIalign, stereo VIO, red-tape jaw, bare-hand collector; documents of the old layout | ~~superseded~~ |

## Installation details

```bash
uv sync                               # Python 3.12; base deps (numpy, scipy, pandas, pyyaml, matplotlib, OpenCV < 5) + pytest
uv sync --group vision                # + torch, torchvision, ultralytics (CoTracker stage, jaw models)
uv sync --group robot                 # + pin, pin-pink, requests, flask (IK labels, deploy, robot bridge)
uv sync --group train                 # + torch, diffusers, accelerate, safetensors (the trainer also needs the LeRobot fork)
uv sync --group collector             # + pyserial, feetech-servo-sdk, av, mcap, tqdm, pupil-apriltags (recorder)
# without uv:
python -m pip install -e ".[test]"    # base deps + pytest
python -m pytest -q                   # needs torch and Pink IK as well (uv: --group vision --group robot; see the README)
```

The training, deploy and robot-bridge environments (pinned versions recorded from the machines that produced the
results, and what is not reproducible from this repository) are listed in [environments.md](environments.md).

Without installing, put `src/` on the path: `PYTHONPATH=src python -m laundry_<stage>.<module>`. The scripts under
`scripts/laundry/` are run as files (`python scripts/laundry/eval/replay_parity.py …`) with the package installed or
`PYTHONPATH=src`.

Notes:
- The original runs used several interpreters: the repo `.venv` for the recorder (`sudo` for the Orbbec on macOS), an
  X-VLA environment with Pink / pinocchio for IK, bridge start and replay, and a separate ultralytics environment for the
  jaw keypoint scripts. The helper modules the code imported from the project's other working folders are copied into
  `src/laundry_shared/` ([provenance + md5](provenance.md)); still outside: the vendor reBot LeRobot plugin, the X-VLA
  training patch.
- The packages read and write the data folders of the original machine by default; set `EGO_DATA_ROOT` (recordings,
  calibration), `EGO_FUSION_WORK` (fusion work tree), `EGO_TRAIN_ROOT` (derived trajectories, reports, datasets) and
  `EGO_CKPT_ROOT` (checkpoint archive) to use other locations ([`src/laundry_common/paths.py`](../src/laundry_common/paths.py)).
  Training and the X-VLA server need the LeRobot / X-VLA environments (torch, lerobot, diffusers), not the base install.
- Not included: raw data, trained checkpoints, the reBot vendor plugin and URDF, ORB-SLAM3 source.

## The split

**Three layers.** Which code is which, at a glance:

| layer | where | what it is |
|---|---|---|
| production code | `src/laundry_{common,tracking,jaw,dataset,training,deploy}/`, `src/laundry_shared/`, `scripts/laundry/` | the code that produced every reported result and runs the robot (same behaviour as the code that ran, checked by characterization tests) |
| portable reference implementation | [`src/laundry_reference/`](../src/laundry_reference/README.md) | small re-implementations of the core ideas that run on synthetic data without the lab set-up; not used by the real pipeline |
| superseded | [`archive/`](../archive/README.md) | earlier approaches kept for the record (see [history](history.md#current-vs-superseded)) |

* **`src/`** holds all importable code, one package per pipeline stage (table above). `laundry_shared` holds the modules
  copied from the HandUMI / teleop work so that imports resolve from a clean clone; they keep their original style.
* **`scripts/laundry/`** holds thin operational entry points: cluster and pod jobs, and the replay-track scripts.
* **`configs/laundry/`** holds calibration and robot files that the code addresses as `<repo root>/configs/laundry/...`
  (through `laundry_common.paths.CONFIGS` or `Path(__file__).parents[n]`).
* **`assets/laundry/examples/`** — the synthetic data and its deterministic generator; **`tests/laundry/`** — import /
  path smoke tests, the scripts' synthetic self-tests and the reference regression.
* **`docs/`, `report/`, `media/`** — details, results, technical reports, figures and clips of the real setup;
  **`hardware/`** — HandUMI build copy; **`archive/`** — superseded code.

## The interfaces of the real pipeline

From the code that defines them:

| interface | definition | where |
|---|---|---|
| world frame | reBot base frame B; camera → base `T_B_C` from `configs/laundry/calibration/robot_cam_v00N.json` (per take by date) | `src/laundry_tracking/camera_calibration.py`, `src/laundry_tracking/calib/umi_tcp_traj.py` |
| jaw-tip path per take | `umi_traj_{right,left}.npz`: jaw-tip TCP `p_B`, rig rotation `R_B_tcp`, `ntags` | `src/laundry_tracking/calib/umi_tcp_traj.py` |
| jaw state | `umi_jaw.npz`: `g_cont` 0 = closed … 1 = open, `valid`, `ev_t`; binary `state` 1 = open, 0 = closed; servo-derived files also keep `motor_pos` and set `model = feetech_sts3215_motor` | video: `src/laundry_shared/visual_teleop/tools/umi_jaw.py`, `src/laundry_jaw/aperture.py`; servo (Feetech STS3215, torque-off encoder, 10-29 →): `src/laundry_jaw/from_motor.py` |
| training label | **F** (10-29 →, `--label fused`) = fused human jaw-tip pose in reBot gripper axes (`R_B_tcp·R_U_E`), 0.25 s smoothing, before IK; no table lift, no clamp; IK changes no value and masks no frame. ~~**A** (`--label rob`) = F + table clearance (tilt toward vertical ≤ 30°, then corner lift)~~. **B** (`--label ikfk`) = IK → FK (QA only) | `src/laundry_dataset/robotize_ik.py` (`--label rob` / `ikfk` / `fused` in `laundry_dataset.tag_rig_unsplit`) |
| dataset | LeRobot v3, 20 Hz, 3 cameras; `observation.state` 30-D = per-arm pose relative to the episode start (pose9 each), g_L, g_R, left-right relative pose9 + valid flag (jaw entries zeroed when `GRIP_STATE_MASK=1`); `action` 20-D absolute `[L xyz, L rot6d, R xyz, R rot6d, g_L, g_R]` (rot6d = first two rows of R; gripper target = binary jaw state with a 0.1 s ramp, continuous aperture stored separately as `gripper_cont`), with a per-dim `action_mask[20]` | `src/laundry_dataset/tag_rig_unsplit.py`, `src/laundry_dataset/tag_rig_to_lerobot.py`, `src/laundry_shared/contracts/` |
| chunk | relative motion `inv(T_t)·T_{t+k}`, k = 1…32 at 20 Hz, built in the training loader; deploy commits the first steps of each plan, then re-plans (DiT 4 by default, ACT 1, X-VLA 8; selectable in the UI; at speed > 1 the DiT commit is capped at 32/speed) | [details.md §3.2–3.3](details.md#33-running-a-trained-policy-on-the-robot) |
| robot model corrections | TCP offset, per-arm joint zero offsets, left base pose: `configs/laundry/robot/{tcp_override,joint_offset,base_override}_v001.json` | applied in `src/laundry_shared/visual_teleop/robot/ik_pink.py` |
| units | code stores metres and seconds (e.g. tag size 0.048 m); README / report quote mm | — |

## The reference interface: trajectory CSV

All `laundry_reference` tools exchange one wide CSV, defined in [`src/laundry_reference/trajio.py`](../src/laundry_reference/trajio.py)
and documented with the example data in [`assets/laundry/examples/README.md`](../assets/laundry/examples/README.md). One row per frame, `t` in seconds,
then per hand (`R_` right, `L_` left):

| column | unit | meaning |
|---|---|---|
| `x, y, z` | m | jaw-tip position in the world (robot base) frame: +x forward, +y left, +z up; table plane z = 0 |
| `qx, qy, qz, qw` | – | jaw-tip orientation, unit quaternion |
| `jaw` | – | 1 = open, 0 = closed |
| `visible` | – | 1 = pose measured (markers seen), 0 = not measured (pose is NaN) |

Columns appended by the tools and carried along unchanged by later ones:

| column | written by | meaning |
|---|---|---|
| `filled` | `gap_fill` | 0 measured / 1 interpolated / 2 unfilled (too long or open-ended) |
| `human_x/y/z` | `robotize` | the path before robotization |
| `disp_mm` | `robotize` | per-frame robotization displacement, mm |
| `quality` | `robotize` | 0 / 1 / 2 (default boundaries 20 / 50 mm) |

## Coordinate representation

* **Storage: absolute poses in the world frame.** Each frame stores the jaw-tip pose `T_B(t) ∈ SE(3)` of each hand in
  the robot base frame `B` (+x forward, +y left, +z up), the jaw state, and flags for whether the pose was measured or
  filled. The jaw-tip TCP is defined the same way for the human gripper and the robot gripper (midpoint between the jaw
  tips), so no tool offset has to be guessed later.
* **Learning contract: relative SE(3).** Learning targets are expressed relative to a reference pose `T_0` (e.g. the
  pose at the start of an action chunk):

  `T_rel(t) = T_0^{-1} T(t)`

  so the target does not depend on where in the workspace the motion happens, and the same contract applies to human
  data and to robot data. Absolute storage keeps the option of re-deriving any relative form, and is what the replay
  uses. A reference implementation of the conversion and its round trip is in
  [`src/laundry_reference/relative_se3.py`](../src/laundry_reference/relative_se3.py) (chunk horizon `--chunk`, default 1.0 s).
* **Per-frame quality.** Robotization stores, for every frame, how far the robot target was moved from the human path,
  and a coarse quality class derived from it; filled frames are flagged separately. These are intended as label weights.
