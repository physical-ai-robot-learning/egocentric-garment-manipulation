# ray_xvla_umiflat — X-VLA fine-tune on the flattening data (2026-11-10 →)

Second policy family on the same converted demonstrations: X-VLA initialised from our SoftFold checkpoint (650k steps),
new domain id, relative 32-step action chunks, continuous jaw target, jaw state masked, no gradient clipping.

- `run.sh` — job entry (`STAGE=smoke|train`, `RESUME=1`; continual stages `cstable|cdecay|cum|ceval`).
- `src/train_umiflat.py` — wraps the lerobot X-VLA trainer (domain id, micro-batching, FP32 / TF32 switch).
- `src/continual.py` — warmup-stable-decay schedule and recent-day over-sampling for a dataset that grows every day.
- `src/eval_offline.py` — offline evaluation on held-out demonstrations.
- `src/rel16_*`, `src/xvla_cumulative.py` — per-sample loss-mask dataset and the cumulative-set builder it uses.

Data: [`../xvla_data/`](../xvla_data/README.md). Status: [docs/status_2026-11-12.md](../../../docs/status_2026-11-12.md) §3
(no offline or robot result yet).
