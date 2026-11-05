"""Per-arm / per-step loss masks and per-sample episode weights of the tag-rig trainers (training stage; was part of
training/tagrig_common.py).

Action mask (2026-10-23, unsplit episodes): the dataset columns action_mask [20] (stored target valid) and anchor_ok [4]
(anchor pose valid, [pos_L, rot_L, pos_R, rot_R]) become lookup tables by global frame index (installMasks); per batch the chunk
mask is built from them (chunkMask: ACT rows t..t+31, DiT rows t-1..t+31 with row 0 = 1; arm dims also need the anchor's
pose_ok) and applied to the elementwise L1 (ACT) / MSE (DiT), renormalised per group by the mask sum (maskedElementwise).
USE_ACTION_MASK=0 off. MASK_CROSS_GAP=1 (11-04, read per batch): once a future step's xyz is masked, every later step of that
chunk is masked too.
Episode weight: the dataset column `weight` (1.0 TRAIN / 0.3 DOWNWEIGHT) by global frame index (installWeights); it multiplies the
per-sample elementwise loss before the policy's own masked mean (USE_EPISODE_WEIGHT=0 disables). Val eval is unweighted.

BATCH holds the per-batch state shared by the update hook (metrics), the loss proxy and the BCE jaw head: M (chunk mask), Mg (jaw
mask of the BCE head), w (episode weights), pad, pad_masking, groups (last per-group losses), err_t, grad_norm, max_norm.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .layout import ACTION_DIM, ANCHOR_POS_L, ANCHOR_POS_R, ANCHOR_ROT_L, ANCHOR_ROT_R, GROUPS, POS_L, POS_R, ROT_L, ROT_R
from .rel_chunk import H_ACT

ON = "1"
N_ANCHOR = 4                         # anchor_ok columns [pos_L, rot_L, pos_R, rot_R]
BATCH = {}                                                       # was tagrig_common._LAST
MASK_TABLES = dict(kind=None, mask=None, pose_ok=None, first=None, last=None)   # was tagrig_common._MASK
EPISODE_WEIGHTS = dict(byIndex=None)                             # was tagrig_common._WEIGHT_BY_INDEX


def maskedElementwise(out, mask, pad=None, padMasking=True, groups=None):
    """[2026-10-23] per-element loss WEIGHTS (0 = masked) with PER-GROUP normalisation: the policy's own reduction (mean over
    unpadded rows x dims, or the plain mean) becomes  sum_g (|D_g| / D) * sum(w_g * err_g) / sum(w_g)  for g in pos / rot / grip,
    i.e. each group is a weighted mean of its own supervised entries (NOT mean(w * err)); a group with zero total weight
    contributes 0. out, mask: (B, H, D); pad: (B, H) bool (True = padded row)."""
    groups = groups or GROUPS
    tD = out.shape[-1]
    tM = mask.to(out.device, out.dtype)
    if pad is not None:
        tM = tM * (~pad.to(out.device)).unsqueeze(-1).to(out.dtype)
    if pad is not None and padMasking:
        tDenom = ((~pad).sum() * tD).to(out.dtype)
    else:
        tDenom = torch.tensor(float(out.numel()), device=out.device, dtype=out.dtype)
    tScale = torch.zeros(tD, device=out.device, dtype=out.dtype)
    for tIx in groups.values():
        tSum = tM[..., tIx].sum()
        if tSum > 0:
            tScale[tIx] = tDenom * (len(tIx) / tD) / tSum
    return out * tM * tScale


def installMasks(ds, kind: str):
    """per-frame `action_mask` [20] and `anchor_ok` [4] -> lookup tables by global index (+ each frame's episode first / last
    index)"""
    tHf = ds.hf_dataset.with_format("numpy")
    if "action_mask" not in ds.features:
        print("[tagrig] dataset has no action_mask -> unmasked", flush=True)
        return ds
    tIdx = np.asarray(tHf["index"]).astype(np.int64)
    tEp = np.asarray(tHf["episode_index"])
    tN = int(tIdx.max()) + 1
    tMask = np.ones((tN, ACTION_DIM), np.float32)
    tPose = np.ones((tN, N_ANCHOR), np.float32)
    tFirst = np.zeros(tN, np.int64)
    tLast = np.zeros(tN, np.int64)
    tMask[tIdx] = np.asarray(tHf["action_mask"])
    tPose[tIdx] = np.asarray(tHf["anchor_ok"])                                  # [pos_L, rot_L, pos_R, rot_R]
    for tEpisode in np.unique(tEp):
        tSel = tIdx[tEp == tEpisode]
        tFirst[tSel] = tSel.min()
        tLast[tSel] = tSel.max()
    MASK_TABLES.update(kind=kind, mask=torch.from_numpy(tMask), pose_ok=torch.from_numpy(tPose), first=torch.from_numpy(tFirst),
                       last=torch.from_numpy(tLast))
    print(f"[tagrig] action_mask: {tMask[tIdx].mean(0).round(3).tolist()} (mean kept per dim), "
          f"anchor_ok {tPose[tIdx].mean(0).round(3).tolist()}", flush=True)
    return ds


def chunkMask(index):
    """(B,) global frame index -> (B, H, 20) chunk mask"""
    tT = index.reshape(-1).long().cpu()
    tTab = MASK_TABLES
    if tTab["kind"] == "act":
        tRows = torch.arange(0, H_ACT)[None] + tT[:, None]
    else:
        tRows = torch.arange(-1, H_ACT)[None] + tT[:, None]
    tRows = torch.maximum(torch.minimum(tRows, tTab["last"][tT][:, None]), tTab["first"][tT][:, None])
    tM = tTab["mask"][tRows].clone()                                           # (B, H, 20) target weights
    tPo = tTab["pose_ok"][tT]                                                   # anchor weights [pos_L, rot_L, pos_R, rot_R]
    # step weight = min(weight(anchor t), weight(t+k)); the relative translation R(t)^T (p(t+k) - p(t)) also needs the anchor
    # rotation
    tM[..., POS_L] = torch.minimum(tM[..., POS_L], tPo[:, ANCHOR_POS_L, None, None]) * (tPo[:, ANCHOR_ROT_L] > 0)[:, None, None]
    tM[..., ROT_L] = torch.minimum(tM[..., ROT_L], tPo[:, ANCHOR_ROT_L, None, None])
    tM[..., POS_R] = torch.minimum(tM[..., POS_R], tPo[:, ANCHOR_POS_R, None, None]) * (tPo[:, ANCHOR_ROT_R] > 0)[:, None, None]
    tM[..., ROT_R] = torch.minimum(tM[..., ROT_R], tPo[:, ANCHOR_ROT_R, None, None])
    if os.environ.get("MASK_CROSS_GAP", "0") == ON:
        # 11-04: once a future step's xyz is masked (gap >= FILL_MASK_S), every later step of that chunk is masked too
        # (the path across the gap is uncertain)
        tR0 = 0 if tTab["kind"] == "act" else 1
        for tSl in (POS_L, POS_R):
            tM[:, tR0:, tSl] = torch.cummin(tM[:, tR0:, tSl], dim=1).values
    if tTab["kind"] != "act":
        tM[:, 0] = 1.0                                                         # DiT row 0 = identity target
    return tM


def installWeights(ds):
    """global frame index -> episode weight (dataset column `weight`; absent -> all 1)"""
    tHf = ds.hf_dataset.with_format("numpy")
    tIdx = np.asarray(tHf["index"]).astype(np.int64)
    tW = np.asarray(tHf["weight"]).reshape(-1).astype(np.float32) if "weight" in ds.features else np.ones(len(tIdx), np.float32)
    tTab = np.ones(int(tIdx.max()) + 1, np.float32)
    tTab[tIdx] = tW
    EPISODE_WEIGHTS["byIndex"] = torch.from_numpy(tTab)
    print(f"[tagrig] episode weights: {np.unique(tW, return_counts=True)}", flush=True)
    return ds
