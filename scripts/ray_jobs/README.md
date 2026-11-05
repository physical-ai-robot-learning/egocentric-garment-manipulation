# ray_jobs

GPU jobs submitted to a Ray cluster (gateway job API, GPU selected by a node label, data and checkpoints on the
cluster's shared storage, environment from each job folder's `pyproject.toml` + `uv.lock`, run with `uv run`).

**Status:** current (11-05). Cluster addresses and storage paths in the scripts are those of the cluster the project used.

## Layout
- `cotracker/`: CoTracker3 point tracking for one conversion pack (`submit.sh` / `submit_gateway.sh` on the cluster,
  `submit_runpod.sh` + `up_one.sh` on a rented pod, `CT_HOST` / `CT_PORT` required), `run_cotracker.py` (online),
  `run_cotracker_offline.py`.
- `ditddp/`: DiT fine-tune on 2× RTX 4090 with distributed data parallel (FLAT250, FLAT250-PGRASP). `run.sh` stages:
  `agg` (aggregate per-take sets into one train / val / test), `smoke` (20 updates), `train` (`DS`, `EPOCHS`, a
  checkpoint every epoch, `RESUME=1`, `FRESH=1`); `PGRASP=<version>` requires `<set>/meta/pgrasp_<version>.npy`. The job
  folder is this folder plus a copy of the repository `src/` as `src/`.

## Provenance
Added 11-04 (CoTracker on the cluster, then a rented pod while the cluster storage was read-only) and 11-05 (DDP job).
