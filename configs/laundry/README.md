# configs/laundry

Calibration results, robot settings / overrides, regression baselines, HandUMI collector profiles and pinned
environments.

**Status:** current (`calibration/`, `robot/`, `regression/`, `envs/`); `handumi/` = HandUMI collector profiles (the
collector modules this code uses are in `src/laundry_shared/handumi_collector/`)

## Inputs → Outputs
Written by the calibration modules in `laundry_tracking.calib` and by `scripts/laundry/retarget/umi_align.py`; read by
those, by `laundry_shared.visual_teleop` (IK model corrections) and by the pipeline packages (`laundry_tracking`,
`laundry_dataset`, `laundry_deploy`).

## Entry points
None. Code addresses these files as `<repo root>/configs/laundry/...` (`laundry_common.paths.CONFIGS` or
`Path(__file__).parents[n]`).

## Layout
- `calibration/` — camera ↔ robot (`robot_cam_v00N`), tag rig, start anchors, UMI ↔ EEF alignment, jaw models / calibration, IMU
- `robot/` — MIT bridge deploy settings, TCP / joint-offset / base overrides applied in `src/laundry_shared/visual_teleop/robot/ik_pink.py`
- `regression/` — replay-planner baseline for `scripts/laundry/eval/regress_replay_plan.py`
- `handumi/` — HandUMI collector hardware / task / pose profiles
- `envs/` — pinned requirement files of the training and deploy environments ([docs/environments.md](../../docs/environments.md))

## Provenance
Was `configs/` + `envs/` until the 2026-11-05 layout change ([docs/refactor_2026-11.md](../../docs/refactor_2026-11.md)).
The configs of the parked bare-hand collector are in `archive/configs/laundry/`. Some JSON / YAML files name, in comments or
metadata fields, the paths of their date (`configs/calibration/...`, `configs/robot/...`, or the calibration folder of
the recording machine); they are records and were not edited, so that the files the code reads stay byte-identical.
