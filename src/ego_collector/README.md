# ego_collector

V0 egocentric collector: head C922 + AprilTag gloves and world tags → offline pseudo-action pipeline (`record → process-tags → process-hands[3d] → generate-actions → qa`).

**Status:** superseded (by `handumi_collector`). Kept because `ego_collector.camera` (UVC/Orbbec capture, ChArUco helpers) is imported by `handumi_collector` and the calibration scripts.

## Inputs → Outputs

- In: head-camera recordings with tagged gloves; configs `configs/{world_tag_map,wrist_extrinsics,qa,aperture_calibration,collection_plan}.yaml`, `configs/camera/`.
- Out: per-episode tag tracks, hand poses and pseudo actions with QA reports.

## Entry points

```bash
ego-collector --help          # console script declared in pyproject.toml (ego_collector.cli:main)
python -m ego_collector --help
```

## Layout

- `camera/` — capture backends and calibration helpers (live dependency)
- `commands/`, `cli.py` — CLI
- `tracking/`, `hands/`, `hands3d/` — tag and hand tracking (incl. WiLoR/HaWoR and Aero-hand adapters)
- `actions/`, `filtering/`, `engine/` — pseudo-action generation
- `qa/`, `io/`, `recording/`, `visualization/`

## Provenance

Shared with [egocentric-cube-stacking](https://github.com/physical-ai-vla/egocentric-cube-stacking) (`src/ego_collector/`): both repositories hold identical copies of the Python files of this package (see [`docs/provenance.md`](../../docs/provenance.md)). The guides below are in egocentric-cube-stacking.
See [`docs/experimental/README.md`](https://github.com/physical-ai-vla/egocentric-cube-stacking/blob/main/docs/experimental/README.md) and [`docs/provenance/experimental_MANIFEST.md`](https://github.com/physical-ai-vla/egocentric-cube-stacking/blob/main/docs/provenance/experimental_MANIFEST.md). Tests: `tests/experimental/v0/`.
