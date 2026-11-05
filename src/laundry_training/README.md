# laundry_training

DiT / ACT training on the LeRobot fork of the training nodes, with the project's dataset adapters and loss options.

**Status:** current (FLAT250, FLAT-PGRASP). Needs the LeRobot fork (`configs/laundry/envs/requirements-train.txt`); the
adapters and options are importable and tested without it.

## Inputs → Outputs
A LeRobot set (`data/lerobot/<name>`) + an initial checkpoint → checkpoints. Options are environment variables
(`env_options`): `GRIP_STATE_MASK` (hide the jaw state), `PGRASP=<version>` (jaw state slots ← p_grasp side file),
`MASK_CROSS_GAP`, `GRIP_BCE`, `GRAD_CLIP`, `AUG` + `AUG_*` (train-time image augmentation incl. saturation / hue).

## Layout
| module | what |
|---|---|
| `train_dit_tagrig`, `train_dit_mix` | entry points (wrap `lerobot_train` with the adapters below) |
| `datasets` (`DitRelDataset`, `ActRelDataset`) | relative-SE(3) action chunks, jaw-state masking / p_grasp, augmentation |
| `rel_chunk`, `layout` | action contract (33-dim), state layout |
| `loss_masks`, `grip_head`, `stats`, `metrics`, `augment`, `env_options` | loss masks across gaps, jaw BCE head, normalisation stats, logged metrics, image augmentation, options |

## Entry points
```bash
python -m laundry_training.train_dit_tagrig --policy.path=<init> --dataset.root=<set> ...     # on a training node
bash scripts/laundry/ray_ditddp/run.sh                                                         # 2x4090 DDP job (BH Ray)
```
Tests: `src/laundry_training/tests/` (stubbed LeRobot; characterization of masks, chunks, augmentation, stats).

## Provenance
Was `src/garment_pipeline/training/` (`tagrig_common` split into the modules above, 11-04 OA refactor).
