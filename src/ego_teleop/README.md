# ego_teleop

Wearable teleoperation research stack: wrist fisheye + IMU (tag-free VIO) → relative SE(3) for the reBot arm, head RGB-D hand pose → DexPilot → Aero Hand, with calibration tooling.

**Status:** experimental (research, not deployed; live wrist VIO is the blocker — see the docs)

## Inputs → Outputs

- In: HandUMI wrist camera + IMU streams, head RGB-D; configs `configs/ego_teleop/`, calibrations `configs/calibration/`.
- Out: robot commands through the reBot client (with safety gates), episode logs, calibration bundles.

## Entry points

```bash
python -m ego_teleop.tools.m1_vio ...          # runbooks: docs/experimental/ego_teleop/
python -m ego_teleop.tools.f5_teleop_hud ...
python -m ego_teleop.tools.kalibr_export ... / kalibr_import ... / imu_allan ...
```

## Layout

- `calibration/` — wrist bundle (Kalibr / Allan)
- `tracking/` — VIO backends (OpenVINS, MASt3R live), fused wrist provider
- `hand3d/`, `retarget/` — RGB-D hand pose and Aero retargeting
- `robot/` — reBot client, coordinator, safety
- `recorder/`, `transforms/`, `tools/`, `config.py`

## Provenance

Shared with [egocentric-cube-stacking](https://github.com/physical-ai-robot-learning/egocentric-cube-stacking) (`src/ego_teleop/`): both repositories hold identical copies of the Python files of this package (see [`docs/provenance.md`](../../docs/provenance.md)). The guides below are in egocentric-cube-stacking.
See [`docs/experimental/README.md`](https://github.com/physical-ai-robot-learning/egocentric-cube-stacking/blob/main/docs/experimental/README.md) and [`docs/provenance/experimental_MANIFEST.md`](https://github.com/physical-ai-robot-learning/egocentric-cube-stacking/blob/main/docs/provenance/experimental_MANIFEST.md). Tests: `tests/experimental/teleop/`.
