"""Training-loop hooks of the tag-rig trainers: masked / weighted elementwise loss, gradient accumulation, metrics.jsonl
(training stage; was part of training/tagrig_common.py).

installMetrics(lt, policyModule, lossFnName) wraps lerobot_train.update_policy:
  - per batch: chunk mask (loss_masks.chunkMask, USE_ACTION_MASK=0 off), BCE jaw mask (GRIP_BCE), episode weights
  (USE_EPISODE_WEIGHT=0
    off) and padding into loss_masks.BATCH; the policy module's `F` is replaced by a proxy that applies them to the elementwise
    loss
  - GRAD_ACCUM=K > 1: the update runs as K micro-batches (accumUpdate)
  - METRICS_EVERY steps (default 1) -> <output_dir>/metrics.jsonl: loss, global grad-norm BEFORE clipping, clip threshold, clipped
  flag,
    lr, per-group loss on normalised targets (pos = xyz L/R, rot = rot6d L/R, grip = gL/gR). NaN/Inf loss or grad -> stop.
  - GRAD_SPLIT_EVERY=K (0 = off): every K steps an extra forward/backward logs gn_recon (L1 / diffusion MSE), gn_rest (ACT:
    kl_weight*KL), gn_total and gn_recon_{pos,rot,grip} -- the source of a large pre-clip grad-norm.
  - RESUME=1 continues the step count of an existing metrics.jsonl; DEADLINE_TS (unix s) forces one extra checkpoint.
The output directory and the train config come from lt._TAGRIG_OUT / lt._TAGRIG_CFG (set by the launcher).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import time

import numpy as np
import torch

from . import env_options
from .layout import ACTION_DIM, GRIP_A, GROUPS
from .loss_masks import BATCH, EPISODE_WEIGHTS, MASK_TABLES, chunkMask, maskedElementwise

ON = "1"
GRAD_SPLIT_TOL = 1e-9                # |loss - recon| below this: no separate "rest" term (gn_rest = 0)
METRICS_FILE = "metrics.jsonl"
FIRST_BATCH_PREVIEW = 8              # episode weights printed from the first batch


class _LossProxy:
    """stands in for `F` inside a policy module: masks / weights / records elementwise losses with a trailing dim of 20
    (was _FProxy)"""

    def __init__(self, names):
        self.mNames = names

    def __getattr__(self, key):
        tFn = getattr(torch.nn.functional, key)
        if key not in self.mNames:
            return tFn

        def wrapped(*a, **kw):
            tOut = tFn(*a, **kw)
            if kw.get("reduction") == "none" and tOut.shape[-1] == ACTION_DIM:
                tM = BATCH.get("M")
                if tM is not None and tM.shape == tOut.shape:
                    tOut = maskedElementwise(tOut, tM, BATCH.get("pad"), BATCH.get("pad_masking", True))
                tW = BATCH.get("w")                    # EPISODE weight (dataset column `weight`, 1.0 / 0.3) -> per-sample weight
                if tW is not None and tW.shape[0] == tOut.shape[0] and not env_options.WEIGHT_GRIP_ONLY:
                    tOut = tOut * tW.to(tOut.device, tOut.dtype).reshape(-1, *([1] * (tOut.dim() - 1)))
                BATCH["err_t"] = tOut
                tPer = tOut.detach().float().reshape(-1, ACTION_DIM).mean(0)
                BATCH["groups"] = {tG: float(tPer[tIx].mean()) for tG, tIx in GROUPS.items()}
            return tOut
        return wrapped


def _sliceBatch(value, sl, batchSize):
    return value[sl] if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batchSize else value


def accumUpdate(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator=None, lr_scheduler=None, lock=None,
                sample_weighter=None):
    """[2026-10-29] effective batch = --batch_size, computed as GRAD_ACCUM micro-batches (4090: a physical batch of 64 is OOM).
    Same body as lerobot_train.update_policy (3c15fd8) -- clip, ONE optimizer + scheduler step per call, so the step count =
    optimizer updates -- with the per-sample loss masks / episode weights / padding of BATCH sliced to each micro-batch. The
    group-normalised masked loss is normalised within each micro-batch (mean of the micro-batch losses). Keyword names are
    lerobot's update_policy signature."""
    tStart = time.time()
    policy.train()
    tB = int(batch["index"].shape[0])
    tN = min(env_options.GRAD_ACCUM, tB)
    tEdges = np.linspace(0, tB, tN + 1).astype(int)
    tFull = {tK: BATCH.get(tK) for tK in ("M", "Mg", "w", "pad")}
    tTotal = 0.0
    tOd = None
    for tJ in range(tN):
        tSl = slice(int(tEdges[tJ]), int(tEdges[tJ + 1]))
        for tKey, tVal in tFull.items():
            BATCH[tKey] = _sliceBatch(tVal, tSl, tB)
        tMicro = {tK: (tV[tSl] if (torch.is_tensor(tV) and tV.dim() > 0 and tV.shape[0] == tB)
                       or (isinstance(tV, list) and len(tV) == tB) else tV) for tK, tV in batch.items()}
        with accelerator.autocast():
            tLoss, tOd = policy.forward(tMicro)
        accelerator.backward(tLoss / tN)
        tTotal += float(tLoss.detach()) / tN
    BATCH.update(tFull)
    if grad_clip_norm > 0:
        tGn = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        tGn = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
    with (lock if lock is not None else contextlib.nullcontext()):
        optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    tUnwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if hasattr(tUnwrapped, "update") and callable(tUnwrapped.update):
        tUnwrapped.update()
    train_metrics.loss = tTotal
    train_metrics.grad_norm = float(tGn)
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.time() - tStart
    return train_metrics, tOd


def _gradSplit(policy, batch):
    """separate gradient norms of the reconstruction term (per action group) and of the rest (ACT: kl_weight * KL)"""
    policy = getattr(policy, "module", policy)       # unwrapped: autograd.grad never fires the DDP reducer hooks
    tParams = [tP for tP in policy.parameters() if tP.requires_grad]
    tNorm = lambda t: float(torch.sqrt(sum((tG.float() ** 2).sum()
                                           for tG in torch.autograd.grad(t, tParams, retain_graph=True, allow_unused=True)
                                           if tG is not None)))
    tLoss, _ = policy.forward(batch)
    tErr = BATCH["err_t"]
    tM = (~batch["action_is_pad"]).unsqueeze(-1).to(tErr.dtype) if "action_is_pad" in batch else torch.ones_like(tErr[..., :1])
    tRec = lambda ix: (tErr[..., ix] * tM).sum() / (tM.sum() * len(ix)).clamp_min(1)
    tR = tRec(list(range(ACTION_DIM)))
    tOut = dict(gn_recon=tNorm(tR), gn_rest=tNorm(tLoss - tR) if float((tLoss - tR).abs()) > GRAD_SPLIT_TOL else 0.0,
                gn_total=tNorm(tLoss))
    for tGroup, tIx in GROUPS.items():
        tOut[f"gn_recon_{tGroup}"] = tNorm(tRec(tIx))
    policy.zero_grad(set_to_none=True)
    return tOut


def installMetrics(lt, policyModule, lossFnName: str, every=None) -> None:
    """wrap lt.update_policy (and accelerate's clip_grad_norm_) with the masks, weights, accumulation and metrics.jsonl described
    above"""
    import accelerate
    policyModule.F = _LossProxy({lossFnName})
    every = int(every or os.environ.get("METRICS_EVERY", 1))
    tOrigClip = accelerate.Accelerator.clip_grad_norm_

    def clip(self, parameters, max_norm, *a, **kw):
        tNorm = tOrigClip(self, parameters, max_norm, *a, **kw)
        BATCH["grad_norm"] = float(tNorm)
        BATCH["max_norm"] = float(max_norm)
        return tNorm
    accelerate.Accelerator.clip_grad_norm_ = clip
    tOrigUpdate = lt.update_policy
    tState = dict(step=0, fh=None, n_clip=0)

    tSplitEvery = int(os.environ.get("GRAD_SPLIT_EVERY", 0))
    tRank = int(os.environ.get("RANK", 0))
    if tRank != 0:
        tSplitEvery = 0                     # DDP: grad split + metrics on rank 0 only (grad-norm is post all-reduce everywhere)

    def update(train_metrics, policy, batch, optimizer, grad_clip_norm, *a, **kw):
        # the preprocessor drops `weight`; `index` survives -> table lookup
        tIx = batch.get("index") if isinstance(batch, dict) else None
        tUseMask = MASK_TABLES["mask"] is not None and torch.is_tensor(tIx) and os.environ.get("USE_ACTION_MASK", "1") == ON
        BATCH["M"] = chunkMask(tIx) if tUseMask else None
        BATCH["pad"] = batch.get("action_is_pad") if isinstance(batch, dict) else None
        if env_options.GRIP_BCE and BATCH["M"] is not None:
            # 10-29: jaws trained by the BCE head only (grip_head), not by the diffusion MSE
            BATCH["Mg"] = BATCH["M"][..., GRIP_A].clone()
            tM = BATCH["M"].clone()
            tM[..., GRIP_A] = 0.0
            BATCH["M"] = tM
        tConfig = getattr(getattr(policy, "module", policy), "config", None)
        BATCH["pad_masking"] = True if MASK_TABLES["kind"] == "act" else bool(getattr(tConfig, "do_mask_loss_for_padding", False))
        tWeights = EPISODE_WEIGHTS["byIndex"]
        BATCH["w"] = (tWeights[tIx.reshape(-1).long().cpu()] if torch.is_tensor(tIx) and tWeights is not None
                      and os.environ.get("USE_EPISODE_WEIGHT", "1") == ON else None)
        if tState["step"] == 0 and tRank == 0:
            tPreview = None if BATCH["w"] is None else BATCH["w"][:FIRST_BATCH_PREVIEW].tolist()
            print(f"[tagrig] episode weight in batch: {tPreview} keys {sorted(batch)}", flush=True)
        tSplitNow = tSplitEvery and (tState["step"] == 0 or (tState["step"] + 1) % tSplitEvery == 0)
        tSplit = _gradSplit(policy, batch) if tSplitNow else {}
        if tState.get("init") is None:                       # resume: continue the step count of an existing metrics.jsonl
            tState["init"] = True
            tFile = pathlib.Path(lt._TAGRIG_OUT) / METRICS_FILE
            if tFile.exists() and os.environ.get("RESUME", "0") == ON:
                try:
                    tState["step"] = json.loads(tFile.read_text().strip().splitlines()[-1])["step"]
                except Exception:
                    pass
        BATCH["grad_norm"] = None
        tUpdate = accumUpdate if env_options.GRAD_ACCUM > 1 else tOrigUpdate
        tTm, tOd = tUpdate(train_metrics, policy, batch, optimizer, grad_clip_norm, *a, **kw)
        if BATCH.get("grad_norm") is None:
            # 10-29 GRAD_CLIP=0: accelerate's clip (our hook) is not called -> lerobot's own norm
            tGm = getattr(tTm, "grad_norm", None)
            BATCH["grad_norm"] = float(getattr(tGm, "val", tGm)) if tGm is not None else float("nan")
        _deadlineCheckpoint(lt, tState, tRank)
        tState["step"] += 1
        tGn = BATCH.get("grad_norm", float("nan"))
        tClipped = bool(tGn > BATCH.get("max_norm", math.inf))
        tState["n_clip"] += tClipped
        tLoss = float(tTm.loss.val) if hasattr(tTm.loss, "val") else float("nan")
        tFinite = math.isfinite(tGn) and math.isfinite(tLoss)
        if tRank == 0 and (tState["step"] % every == 0 or tSplit) or not tFinite:
            if tState["fh"] is None:
                tOut = pathlib.Path(lt._TAGRIG_OUT)
                tOut.mkdir(parents=True, exist_ok=True)
                tState["fh"] = open(tOut / METRICS_FILE, "a")
            tRec = dict(step=tState["step"], t=time.time(), loss=tLoss, grad_norm=tGn, clip=BATCH.get("max_norm"),
                        clipped=tClipped, clip_frac_so_far=tState["n_clip"] / tState["step"], lr=optimizer.param_groups[0]["lr"],
                        **{f"loss_{tK}": tV for tK, tV in BATCH.get("groups", {}).items()}, **tSplit,
                        **({tK: tV for tK, tV in (tOd or {}).items() if isinstance(tV, (int, float))}))
            tState["fh"].write(json.dumps(tRec) + "\n")
            tState["fh"].flush()
        if not tFinite:
            raise SystemExit(f"NaN/Inf at step {tState['step']}: loss {tLoss} grad_norm {tGn} -> stopped")
        return tTm, tOd
    lt.update_policy = update


def _deadlineCheckpoint(lt, state, rank):
    """10-29 RunPod budget (user: "잔액에 맞게 체크포인트"): at DEADLINE_TS (unix s) one extra full checkpoint (weights + training_state)
    is forced by setting the run's save_freq to 1 for exactly one step, then restored; training continues (a top-up keeps it
    going)."""
    tCfg = getattr(lt, "_TAGRIG_CFG", None)
    tDeadline = float(os.environ.get("DEADLINE_TS", "0") or 0)
    if tCfg is not None and tDeadline > 0:
        if state.get("dl") is None and time.time() >= tDeadline:
            state["dl"] = "armed"
            state["sf"] = tCfg.save_freq
            tCfg.save_freq = 1
            if rank == 0:
                print(f"[tagrig] DEADLINE reached at update {state['step']} -> extra checkpoint now", flush=True)
        elif state.get("dl") == "armed":
            tCfg.save_freq = state["sf"]
            state["dl"] = "done"
