# laundry_common

Machine-specific locations of the pipeline, in one module.

**Status:** current.

## Inputs → Outputs
Environment variables → `pathlib.Path` constants used by every other package. Defaults are the layout of the recording /
training Mac, so nothing needs setting there; on another machine set the variables, no code changes.

| variable | constant | what |
|---|---|---|
| `EGO_DATA_ROOT` | `DATA_ROOT`, `TAG_RIG`, `DATA_CALIB`, `M3_RUNS` | recording workspace (takes, calibration of the recording machine) |
| `EGO_FUSION_WORK` | `FUSION_WORK` | working tree of the fusion stage (CoTracker packs, `out/`, `stitch/qa`) |
| `EGO_TRAIN_ROOT` | `TRAIN_ROOT`, `DERIVED`, `REPORTS`, `IDLE_TRIM_CSV` | training workspace (derived trajectories, LeRobot sets, reports) |
| `EGO_CKPT_ROOT` | `CKPT_ROOT` | checkpoint / raw-take archive |
| `TAGRIG_STATE_DIR` | `TAGRIG_STATE_DIR` | working state of the deploy tools (UI settings, camera map, start pose, height calibration) |
| `EGO_GRASP_WORK`, `EGO_GRASP_ROUNDS`, `EGO_GRASP_SPLITS` | `GRASP_*` | grasp-label work folders (`laundry_grasp`) |
| others | see the module docstring | cockpit tools, models, VIO builds, handumi-sw, IK interpreter, ffmpeg |

Files shipped in the repository resolve from `REPO_ROOT` / `CONFIGS` (`configs/laundry/`).

## Entry points
None (library). `from laundry_common import paths`.

## Provenance
Added in the 11-04 OA refactor (was `egogarment.paths` plus hard-coded paths in the scripts).
