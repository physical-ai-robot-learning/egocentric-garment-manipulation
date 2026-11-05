# laundry_dataset

Training labels and LeRobot datasets from the fused trajectories.

**Status:** current. Builder `tag_rig_unsplit` (label F = fused jaw tip before IK, no clamp or lift; time-stretch; gated
loss masks); split builds of the flattening sets in `builds/`.

## Inputs → Outputs
Derived takes (`EGO_TRAIN_ROOT/data/tag_rig_derived*`), raw takes, calibration → IK labels (`robotize_ik`: robot joint
labels and IK diagnostics) → LeRobot datasets `data/lerobot/<name>` (`tag_rig_unsplit`; per-frame masks, split json).

## Layout
| path | what |
|---|---|
| `robotize_ik`, `robot_arm` (`Arm`) | reBot IK of the fused trajectories (Pink / pinocchio sub-process `EGO_IK_PYTHON`), labels F / A / B |
| `tag_rig_unsplit`, `unsplit_build` (`UnsplitOptions`), `unsplit_render` | the current dataset builder (whole demos, never cut; `--label fused`, `--speed`, `--mask-mode gated`, `--grip-label cont`) |
| `tag_rig_to_lerobot`, `takes`, `calibration`, `arm_assignment`, `export_common` | export helpers shared by the builders |
| `idle_trim`, `ik_qa_stats` | idle start/end table, IK QA statistics |
| `builds/` | flattening sets: `idle_audit`, `ct_gap_audit`, `split_gold_silver_reject` (train = GOLD with gap < 0.25 s, …), `per_take_holdout_build`, `build_and_upload`, `render_qa_videos` |

## Entry points
```bash
python -m laundry_dataset.robotize_ik <raw take> <derived take> --demos ... --workers 8
FILL_MASK_S=0.3 python -m laundry_dataset.tag_rig_unsplit --label fused --speed 0.3 --mask-mode gated --grip-label cont --out <set> --takes ...
python -m laundry_dataset.builds.split_gold_silver_reject <prefix> <derived> <quality.csv> <ctgap.csv>
```
Tests: `src/laundry_dataset/tests/` (synthetic takes with a deterministic fake IK; golden values).

## Provenance
Was `src/garment_pipeline/labels_dataset/` and the flatten build scripts (11-04 OA refactor; split rules of the 11-04
evening ported 11-05).
