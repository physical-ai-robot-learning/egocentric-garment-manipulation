# ray_ditddp

DiT fine-tune of the flattening sets as a BH Ray gateway job: 2× RTX 4090 DDP on `gpu-4090`, the UMI recipe of FLAT149
(init H100 663012, continuous jaw, LR 3e-4 cosine, DDIM 50/16, AUG incl. saturation/hue, `GRAD_CLIP=0`, global batch
1024, `MASK_CROSS_GAP=1`). Used for FLAT250 (11-04) and FLAT-PGRASP (11-05).

**Status:** current (11-05). Follows the 11-04 BH cluster rules: gateway :8266, GPU by label, data and checkpoints under
`/mnt/shared/users/johann`, environment from this folder's `pyproject.toml` + `uv.lock` (`uv run --frozen`).

## Job folder
The job's working directory is this folder plus a copy of the training code under `src/`:
`src/laundry_common`, `src/laundry_training` and the `src/laundry_shared` modules (`contracts`, `xvla_core`,
`umi_to_lerobot`, and the packages they import). Write the repository commit the copy came from into `SOURCE_COMMIT`
(the FLAT250 job ran `7852b9c` + the 11-05 `PGRASP` lines that are now in `src/laundry_training`).

## Stages (`STAGE=…`)
- `agg`: aggregate the per-take `flat1103n_* / flat1104_* / flat1105_*` S03G sets into one `<AGG>_{train,val,test}`
  (lerobot `aggregate_datasets`) and write a merged `meta/split.json`.
- `smoke`: 20 updates into `ckpt/ldt/SMOKE_ddp`.
- `train`: `DS=<set>`, `EPOCHS` (default 100); a checkpoint every epoch, `training_state` kept on the newest two only;
  `RESUME=1` continues from the newest complete checkpoint, `FRESH=1` starts over.
- `PGRASP=<version>` exports the option for the trainer (and for a config-only resume) and refuses to start without
  `<set>/meta/pgrasp_<version>.npy` (made by `python -m laundry_grasp.sidefile`).

Per-epoch offline evaluation on the Mac: `scripts/laundry/eval/epoch_eval_gateway.sh`.
