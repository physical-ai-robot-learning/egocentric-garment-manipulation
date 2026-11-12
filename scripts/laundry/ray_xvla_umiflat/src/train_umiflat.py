"""[2026-11-10 user] X-VLA SoftFold 650k -> UMI flatten (S03G) full SFT: lerobot_train + the REL-only per-dim loss-mask plugin
(rel16_relonly_lossmask.py, copied unchanged from the 4090 umi_bridge/umi76 recipe).

Two additions on top of train_rel16_relonly_lossmask.py:
  * lerobot 0.5.2's lerobot_train imports make_pre_post_processors from lerobot.policies (not .factory), so the plugin's
    aux.loss_mask pass-through is re-bound there too -- without it the mask would be dropped silently (all ones)
  * strict FP32 (user spec): TF32 off for matmul AND cuDNN convolutions (PyTorch enables cuDNN TF32 by default)
Env: RELONLY_PREFLIGHT=N (plugin self-checks on the first N batches), XVLA_TF32=1 to allow TF32."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_relonly_lossmask  # noqa: E402

rel16_relonly_lossmask.install()
import lerobot.policies as LP  # noqa: E402
import lerobot.policies.factory as LF  # noqa: E402

LP.make_pre_post_processors = LF.make_pre_post_processors

# the dataset stores the whole (32, 32) chunk on each row: never let the loader stack chunk_size rows on top of it
# (the 4090 node's patched datasets/factory.py did the same; without it the action becomes (32, 32, 32))
import lerobot.datasets.factory as DF  # noqa: E402

_resolve = DF.resolve_delta_timestamps


def resolve_delta_timestamps(cfg, ds_meta):
    tOut = _resolve(cfg, ds_meta)
    tShape = list(ds_meta.features.get("action", {}).get("shape", []))
    if tOut and "action" in tOut and len(tShape) == 2 and tShape[0] == getattr(cfg, "chunk_size", -1):
        tOut.pop("action")
        print(f"[umiflat] action stored as {tShape}: delta-timestamp expansion of 'action' removed", flush=True)
    return tOut or None


DF.resolve_delta_timestamps = resolve_delta_timestamps
TF32 = os.environ.get("XVLA_TF32", "0") == "1"
torch.backends.cuda.matmul.allow_tf32 = TF32
torch.backends.cudnn.allow_tf32 = TF32
print(f"[umiflat] loss-mask processors re-bound in lerobot.policies; TF32 matmul/cudnn = {TF32}", flush=True)

import time  # noqa: E402
from contextlib import nullcontext  # noqa: E402

import lerobot.scripts.lerobot_train as LT  # noqa: E402

MICRO = int(os.environ.get("XVLA_MICRO", "0"))  # samples per forward/backward on one GPU; 0 = the whole batch at once


def _split(batch, n, i):
    """i-th of n equal slices along dim 0 of every per-sample entry (tensors / lists of the batch size)"""
    tB = batch["action"].shape[0]
    tLo, tHi = i * tB // n, (i + 1) * tB // n
    return {k: (v[tLo:tHi] if (torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == tB) or (isinstance(v, list) and len(v) == tB) else v)
            for k, v in batch.items()}


def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler=None, lock=None,
                  sample_weighter=None):
    """[11-10 user] gradient accumulation: --batch_size = samples per GPU per OPTIMIZER update, XVLA_MICRO = samples per GPU per
    forward/backward.  n = batch / MICRO micro-batches, loss / n each, DDP all-reduce only on the last (no_sync), then ONE
    optimizer + scheduler step -> global_step counts optimizer updates, so 4 GPU x 4 x 1 and 2 GPU x 8 (MICRO 4, n 2) are the same
    run (global batch 16).  Otherwise identical to lerobot_train.update_policy (no clip when grad_clip_norm == 0)."""
    assert sample_weighter is None, "[umiflat] sample weighting is not supported with accumulation"
    start_time = time.perf_counter()
    policy.train()
    tB = batch["action"].shape[0]
    n = max(1, tB // MICRO) if MICRO else 1
    assert tB % n == 0, f"[umiflat] batch {tB} not divisible into {n} micro-batches of {MICRO}"
    tLoss, output_dict = 0.0, None
    for i in range(n):
        with (accelerator.no_sync(policy) if i < n - 1 else nullcontext()):
            with accelerator.autocast():
                loss, output_dict = policy.forward(_split(batch, n, i) if n > 1 else batch)
            accelerator.backward(loss / n)
        tLoss += loss.item() / n
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if hasattr(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
    train_metrics.loss = tLoss
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if not _SAID.get("n"):
        _SAID["n"] = 1
        print(f"[umiflat] accumulation: batch {tB}/GPU -> {n} micro-batch(es) of {tB // n}, x {accelerator.num_processes} GPU = "
              f"global {tB * accelerator.num_processes} per optimizer step", flush=True)
    return train_metrics, output_dict


_SAID = {}
LT.update_policy = update_policy
import continual  # noqa: E402  [11-11] WSD + recent-day sampling, env-gated (continual.py)

continual.install(LT)
LT.main()
