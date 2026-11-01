# configs

Calibration results, robot settings / overrides, regression baselines and HandUMI collector profiles.

**Status:** current (`calibration/`, `robot/`, `regression/`); `handumi/` = HandUMI collector profiles (the collector code, `handumi_collector`, comes from the shared [`robot-learning`](https://github.com/physical-ai-vla/robot-learning) package)

## Inputs → Outputs
Written by the calibration scripts in `scripts/pose/` and `scripts/retarget/`; read by those scripts,
`src/visual_teleop/` and `src/garment_pipeline/`.

## Entry points
None. Code addresses these files as `<repo root>/configs/...`, so this folder stays at the top level.

## Layout
- `calibration/` — camera ↔ robot (`robot_cam_v00N`), tag rig, start anchors, UMI ↔ EEF alignment, jaw models / calibration, IMU
- `robot/` — MIT bridge deploy settings, TCP / joint-offset / base overrides applied in `src/visual_teleop/robot/ik_pink.py`
- `regression/` — replay-planner baseline for `scripts/evaluation/regress_replay_plan.py`
- `handumi/` — HandUMI collector hardware / task / pose profiles

## Provenance
Unchanged location. `configs/laundry/` (bare-hand collector) moved to `legacy/configs/laundry/`.
