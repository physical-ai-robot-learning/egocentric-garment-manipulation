# Refactor 2026-11: OA code layout and documentation restructure

This note maps the paths of this repository before and after the November 2026 restructure. Dated records
(`docs/status_2026-11-0*.md`, `docs/records/`, `report/archive/`) keep the paths of their date; use the tables below to
find the current location.

Two steps:

1. **Code layout (commit `3fbf473`, 2026-11-05).** The code tree was replaced by the codebase that is maintained in the
   project's company repository, where it was restructured on 2026-11-04 to the "OA" layout and style of the company's
   OpenArm work. The behaviour was kept: every package got characterization tests recorded on the code *before* the
   change, and they pass unchanged after it (529 passed, 3 skipped in this repository on 11-05; 596 / 3 after `laundry_grasp` was added the same day).
2. **Documentation (2026-11-05).** The README was shortened to a front page; its long parts moved to
   [pipeline.md](pipeline.md), [codebase.md](codebase.md) and [history.md](history.md) (Korean:
   [pipeline_ko.md](pipeline_ko.md), [codebase_ko.md](codebase_ko.md), [history_ko.md](history_ko.md)) with only path
   fixes; current documents and both reports now name the new paths.

## Why

* **One package per pipeline stage.** The old tree mixed an installable package (`src/garment_pipeline/`), stage scripts
  that put `src/` on `sys.path` (`scripts/pose`, `scripts/retarget`, …), full copies of the HandUMI / teleop packages and
  thin wrappers. A reader had to know the history to find the code of a stage. Now each stage is one package
  (`laundry_jaw`, `laundry_tracking`, `laundry_dataset`, `laundry_training`, `laundry_deploy`), run as
  `python -m laundry_<stage>.<module>`.
* **Same layout as the maintained code.** The company repository is where the code is developed further; keeping the
  public tree identical in layout means a change there maps one to one onto this repository.
* **Names without version suffixes.** Modules carry what they do (`cotrack/point_solver`) instead of their iteration
  (`solve2`); the old name is kept in each module docstring ("was …").
* **Machine paths in one place.** Every data, checkpoint and work root comes from `laundry_common.paths`
  (environment variables `EGO_DATA_ROOT`, `EGO_FUSION_WORK`, `EGO_TRAIN_ROOT`, `EGO_CKPT_ROOT`, `TAGRIG_STATE_DIR`, …;
  the defaults are the paths of the recording machine).
* **Namespaced paths** (`laundry_*`, `…/laundry/`), so the company repository can later merge its OpenArm code without
  path conflicts; this repository keeps the same names.

Code style ("OA"): `mField` instance members, `tLocal` local variables, camelCase functions, methods and parameters,
PascalCase classes, frozen dataclasses for options and records, `…Error` exception classes, named UPPER_SNAKE
constants, `main(argv) -> int` entry points, no work at import time, no `sys.path` edits. Names fixed from outside
(argparse flags, JSON / CSV / npz keys, environment variables, LeRobot / torch hooks, checkpoint keys) are unchanged.

## Structure (old → new)

| old | new |
|---|---|
| `src/garment_pipeline/fusion_stack/` | `src/laundry_tracking/` |
| `src/garment_pipeline/fgo_tools/` | `src/laundry_tracking/fgo/` |
| `scripts/pose/{robot_cam_*,tag_rig_*,umi_side_check,umi_tcp_traj}.py` | `src/laundry_tracking/calib/` |
| `scripts/pose/jaw_*.py` | `src/laundry_jaw/` (`jaw_` prefix dropped: `jaw_from_motor` → `from_motor`, `jaw_aperture` → `aperture`, `jaw_kpt_label` → `kpt_label`, `jaw_kpt_train` → `kpt_train`, `jaw_motorlike` → `motorlike`) |
| `src/garment_pipeline/labels_dataset/` | `src/laundry_dataset/` |
| `scripts/flatten_builds/*` | `src/laundry_dataset/builds/` (module names below) |
| `src/garment_pipeline/training/` | `src/laundry_training/` |
| `src/garment_pipeline/eval_deploy/` | `src/laundry_deploy/` |
| `scripts/deploy/robot_bridge/` | `src/laundry_deploy/robot_service/` |
| `src/egogarment/` (+ the wrappers `scripts/{gap_fill,robotize,parity_metrics,relative_se3,plot_example,trajio}.py`) | `src/laundry_reference/` (wrappers removed: `python -m laundry_reference.<module>` or the `laundry-reference-*` console commands) |
| `src/egogarment/paths.py` | `src/laundry_common/paths.py` |
| `src/{visual_teleop,handumi_collector,ego_collector,ego_teleop,rebot_prep,contracts}`, `src/xvla_core.py`, `src/umi_to_lerobot.py` | `src/laundry_shared/<same name>` — only the modules the laundry code imports (90 Python files), in their original style |
| `src/umi/` (our stand-in for UMI's `pose_util`) | removed (not imported by the laundry code; still in git history and in egocentric-cube-stacking) |
| `scripts/evaluation/` | `scripts/laundry/eval/` (with `epoch_eval_gateway.sh`) |
| `scripts/retarget/` | `scripts/laundry/retarget/` |
| `scripts/grasp/` | `src/laundry_grasp/` (OA package, 11-05: `events`, `crops`, `model`, `pgrasp`, `generate`, `sidefile`, `gold_ui`, `cv`, `render`, `rule_v0`; every real-data output re-made and compared byte for byte with the original scripts) + `archive/laundry/grasp_exploration/` (exploratory scripts, verbatim) |
| `scripts/ray_jobs/cotracker/` | `scripts/laundry/ray_cotracker/` |
| `scripts/ray_jobs/ditddp/` | `scripts/laundry/ray_ditddp/` |
| `scripts/runpod_convert/` | `scripts/laundry/runpod_convert/` |
| `configs/*` | `configs/laundry/*` (`calibration/`, `handumi/`, `regression/`, `robot/`) |
| `envs/requirements-{train,deploy}.txt` | `configs/laundry/envs/` |
| `examples/` | `assets/laundry/examples/` |
| `tests/` | `tests/laundry/` + one `tests/` folder per package (`src/laundry_*/tests/`) |
| `legacy/` | `archive/` (same content) |
| `requirements.txt`, extras `[pipeline]`, `[vision]`, `[robot]`, `[jaw]`, … | `pyproject.toml` with hatchling and dependency groups (`vision`, `robot`, `train`, `collector`, `dev`); extra `[test]` for pip |
| `docs/shared_packages_MANIFEST.md`, `docs/refactor_candidates.md` | `archive/docs/` (they described the old layout) |

The working files that the fusion stage read and wrote next to its modules (`out/`, `stitch/qa/`, start seeds) are now
under `laundry_common.paths.FUSION_WORK` (`EGO_FUSION_WORK`), same sub-paths. The deploy state files (UI settings,
camera map, start pose, height calibration) default to `EGO_TRAIN_ROOT/deploy`; `TAGRIG_STATE_DIR` moves them.

## Modules (old → new)

**laundry_tracking** (was `garment_pipeline.fusion_stack`)

| old | new |
|---|---|
| `core` | `slam_consistency` |
| `camcal` | `camera_calibration` |
| `stitch/wdo` | `whole_demo_stitch` |
| `fgo_tools/fgo_data`, `fgo_tools/fgo_core` | `fgo/arm_data`, `fgo/pose_smoother` |
| `fgo_tools/tag_rot_extract_v002` | `tag_rotations` |
| `orbbec/v2b/track_v31` | `orbbec_tracker` |
| `orbbec/v2b/build_candidate_v0` | `orbbec_candidate` |
| `orbbec/v2b/v32_common` | `arm_rotation` |
| `orbbec/v2b/{imu_motion,new_take_prep,edge_gap_trim,tighten_trim}` | same names, package top level |
| `orbbec/v2b/cotrack/{make_pack2,run_cotracker,solve2,gate,episode_quality,out_of_view}` | `cotrack/{make_pack,run_cotracker,point_solver,gate,episode_quality,out_of_view}` |
| `orbbec/v2b/fusion/{trajectory_fusion_v1,fusion_metrics,make_derived_cgg,render_fusion_qa,render_traj_sources}` | `fusion/{trajectory_fusion,fusion_metrics,make_derived,render_qa,render_sources}` |
| new | `workdirs`, `constants`, `intervals` |

**laundry_dataset** (was `garment_pipeline.labels_dataset` + `scripts/flatten_builds`)

| old | new |
|---|---|
| `robotize_ik.Arm` | `robot_arm.Arm` (also `from laundry_dataset import Arm`) |
| `tag_rig_to_lerobot` | `takes`, `calibration`, `arm_assignment`, `export_common`, `tag_rig_to_lerobot` |
| `tag_rig_unsplit` | `unsplit_build` (`UnsplitOptions`), `unsplit_render`, CLI `tag_rig_unsplit` |
| `flatten_builds/flat_gsr_build` | `builds/split_gold_silver_reject` |
| `flatten_builds/flat1103n_build` | `builds/per_take_holdout_build` |
| `flatten_builds/flat_build_upload` | `builds/build_and_upload` |
| `flatten_builds/ctgap_trimmed` | `builds/ct_gap_audit` |
| `flatten_builds/idle_audit_1103` | `builds/idle_audit` |
| `flatten_builds/render_1103` | `builds/render_qa_videos` |

**laundry_training** (was `garment_pipeline.training`)

| old | new |
|---|---|
| `tagrig_common` | `env_options` (all environment switches, e.g. `GRIP_STATE_MASK`, `GRIP_BCE`, `GRAD_CLIP`, `MASK_CROSS_GAP`, `PGRASP`), `layout`, `stats`, `loss_masks`, `augment`, `datasets`, `grip_head`, `metrics` |
| `tagrig_common.PGRASP` | `laundry_training.env_options.PGRASP` |
| `install_grip_bce` | `installGripBce` (lazy export at the package root, with `ActRelDataset`, `DitRelDataset`) |
| `train_dit_tagrig`, `train_dit_mix`, `*.yaml` | same names |

**laundry_deploy** (was `garment_pipeline.eval_deploy` + `scripts/deploy/robot_bridge`)

| old | new |
|---|---|
| `act_tagrig_core`, `tagrig_core/{policy,geometry}` | `core/{contract,images,chunk,policy_runner,xvla_runner,label_runner,kinematics,collision,camera_model,plot,checkpoint,errors}` |
| `height_calib` | `core/height_calib` + CLI `tools/height_calib` |
| `act_tagrig_deploy`, `tagrig_deploy/*` | `session/*` (`run_loop`, `streamer`, `cameras`, `ring`, `camera_commands`, …), CLI `session/cli` |
| `act_tagrig_ui`, `tagrig_ui/*` | `ui/{server,app,run_control,ready_move,settings,page.html}` |
| `tagrig_offline_eval`, `act_tagrig_domain_gap`, `cam_align`, `cam_viewer`, `return_to_start`, `run_n_steps` | `tools/{offline_eval,domain_gap,cam_align,cam_viewer,return_to_start,run_n_steps}` |
| `xvla_server` | same name |
| `robot_bridge/robot_service_mit.py` | `robot_service/mit_bridge` + `config`, `jaw`, `profile`, `arm_state`, `hardware` |
| `robot_bridge/zero_left_{j6,gripper}_only.py` | `robot_service/zero_left_{j6,gripper}` |
| `robot_bridge/{check_grippers,read_left_gripper}.py` | `robot_service/{check_grippers,read_left_gripper}` |

**laundry_reference** (was `egogarment`): `io/trajio`, `pose/gap_fill`, `pose/relative_se3`, `retarget/robotize`,
`evaluation/parity_metrics`, `viz/plot_example` → top-level `trajio`, `gap_fill`, `relative_se3`, `robotize`,
`parity_metrics`, `plot_example`. The console commands are now `laundry-reference-*` (were `egogarment-*`).

**Commands (examples)**

| old | new |
|---|---|
| `python scripts/pose/robot_cam_solve.py …` | `python -m laundry_tracking.calib.robot_cam_solve …` |
| `python scripts/pose/umi_tcp_traj.py <take> …` | `python -m laundry_tracking.calib.umi_tcp_traj <take> …` |
| `python scripts/pose/jaw_from_motor.py <take>` | `python -m laundry_jaw.from_motor <take>` |
| `python -m garment_pipeline.fusion_stack.orbbec.v2b.cotrack.solve2 <pack>` | `python -m laundry_tracking.cotrack.point_solver <pack>` |
| `python -m garment_pipeline.fusion_stack.orbbec.v2b.fusion.trajectory_fusion_v1 <pack> Cg --ct solve3` | `python -m laundry_tracking.fusion.trajectory_fusion <pack> Cg --ct solve3` |
| `python -m garment_pipeline.labels_dataset.tag_rig_unsplit …` | `python -m laundry_dataset.tag_rig_unsplit …` |
| `python -m garment_pipeline.eval_deploy.act_tagrig_deploy run …` | `python -m laundry_deploy.session.cli run …` |
| `python -m garment_pipeline.eval_deploy.act_tagrig_ui` | `python -m laundry_deploy.ui.server` |
| `python -m garment_pipeline.eval_deploy.tagrig_offline_eval …` | `python -m laundry_deploy.tools.offline_eval …` |
| `python scripts/deploy/robot_bridge/robot_service_mit.py` | `python -m laundry_deploy.robot_service.mit_bridge` |
| `python -m visual_teleop.tools.<tool>` | `python -m laundry_shared.visual_teleop.tools.<tool>` |
| `python -m visual_teleop.tools.bridge_start …` | not in this tree any more; the bridge is started with `python -m laundry_deploy.robot_service.mit_bridge` |
| `python scripts/evaluation/replay_parity.py …` | `python scripts/laundry/eval/replay_parity.py …` |
| `python scripts/gap_fill.py …` | `python -m laundry_reference.gap_fill …` |

## Verification

| check | result |
|---|---|
| characterization tests per package, recorded on the old code and run unchanged on the new one (synthetic recording world through every tracking CLI; deploy goldens byte-identical; synthetic takes through every dataset / build mode; jaw and reference regression values) | pass |
| `PYTHONPATH=src python -m pytest -q` | 596 passed, 3 skipped (skips: lerobot not installed, reBot URDF not in the repository); base install 425 passed, 41 skipped |
| `python analysis/verify_claims.py` | 227 claims checked, 0 failed |
| `python scripts/laundry/eval/replay_parity.py --synthetic`, `python -m laundry_tracking.calib.robot_cam_solve --synthetic` | PASS |
| `--help` of the pipeline entry points (CI step) | parse |
| relative links in the Markdown files | no broken link outside `archive/` |

Not checkable without the lab set-up, so the rewrite was kept mechanical there: real robot (`--go`, cameras, MIT
bridge), CUDA CoTracker and the cluster / RunPod jobs, the LeRobot write and training, the real Pink QP IK, YOLO
training, real takes.

## Suspected bugs (recorded during the code refactor, not fixed, so that behaviour stays the same)

* Deploy UI reuses one run-argument object across runs: after a GO, `no_safety`, `no_floor`, `max_ik_stop_mm=1e9` and
  the speed caps stay set for later DRY runs; `temporal_ensemble` once 0 never comes back.
* MIT bridge: `/plan/rows` returns 500 on a missing row field, `/torque arm=both` returns 500 with one arm connected,
  `/set_speed` applies flags before validating the speeds.
* `pose_smoother.solveRotation` caps the solver at 30 evaluations; the configured 200 is never read.
* `tag_rig_unsplit` always passes `--ik-fail-mm` (default 30) while the library default is 10;
  `per_take_holdout_build` loops forever if no take has more than two keys left.
* `replay_parity.synthetic` has a branch `(1 if leak < 0.1 else 1)` with the same value on both sides.

Operational note: the robot computer runs the deploy from a separate checkout of the code of 11-02/03 with the old
layout; this repository has the same behaviour in the new layout (tests above) but is not the code the robot runs today.
