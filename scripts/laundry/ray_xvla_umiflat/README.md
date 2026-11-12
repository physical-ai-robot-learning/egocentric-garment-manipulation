# ray_xvla_umiflat — X-VLA SoftFold 650k → UMI flattening (2026-11-10 →)

Job package (uv project) for the BH Ray gateway and for RunPod pods. `run.sh STAGE=smoke|train` (fixed-epoch cosine run,
`RESUME=1`), `STAGE=cstable|cdecay|cum|ceval` (11-11 continual plan: warmup-stable-decay, cumulative daily sets, offline
eval). `src/train_umiflat.py` wraps the lerobot X-VLA trainer (domain 22, micro-batching, FP32 / `XVLA_TF32`),
`src/continual.py` (WSD schedule, recent-day over-sampling), `src/eval_offline.py`, `src/xvla_cumulative.py`;
`src/rel16_*` are the loss-mask dataset / trainer modules it builds on. `submit_eval.sh` submits an offline eval.

`runpod/`: `pod_train.sh` (the 11-12 H100 ×4 run on the cumulative set), `pod_bench.sh`, `wait_and_launch.sh` (wait for
GPU stock, create the pod), `pull_ckpts.sh` (pod → Mac → cluster). Pod addresses in these files are of that day's pod.

Status and numbers: `docs/status_2026-11-12.md` §3.
