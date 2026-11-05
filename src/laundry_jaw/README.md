# laundry_jaw

Jaw opening labels of the hand-worn grippers (HandUMI).

**Status:** current: `from_motor` (since 10-29 the jaw label source). The video models (`aperture`, `kpt_*`, `motorlike*`)
are kept for takes recorded before the servo was logged and for comparison.

## Inputs → Outputs
| module | input | output |
|---|---|---|
| `from_motor` | `<take>/gripper_motor_{left,right}.npz` (Feetech STS3215 read with torque off, ~100 Hz) | `<take>/<wrist dir>/umi_jaw.npz` (`g_cont` 0 closed … 1 open, hysteresis `state`, open/close events); per-take servo-offset shift and jaw-direction check |
| `aperture` | wrist-camera frames + jaw-tip keypoint model | jaw aperture 0..1 from the two tips |
| `kpt_label`, `kpt_train` | wrist frames | jaw-tip keypoint labels / model training |
| `motorlike`, `motorlike_overlay` | wrist crops + servo-labelled takes | crop → servo-like jaw model and its review video |
| `filters` | jaw series | median / hysteresis helpers |

## Entry points
```bash
python -m laundry_jaw.from_motor <take> [--hi 0.55 --lo 0.35 --dwell 0.1]
python -m laundry_jaw.aperture --help
```
Tests: `src/laundry_jaw/tests/` (characterization snapshots + jaw-direction check).

## Provenance
Was `scripts/pose/jaw_*.py` (11-04 OA refactor; `jaw_` prefix dropped).
