"""Train-only FIXED normalisation stats of the tag-rig trainers (training stage; was part of training/tagrig_common.py).

Computed from the TRAIN root only (never from val/test, never per episode):
  - state: recomputed from the train frames; action: from every relative window of the train episodes (LeRobot-style clamped
  padding)
  - std floor STD_FLOOR on every non-gripper dim (MEAN_STD), min-max range floor RANGE_FLOOR (MIN_MAX), symmetric around the
  centre
  - ACT action (MEAN_STD in LeRobot): mean/std set to (max+min)/2, (max-min)/2 -> the same [-1, 1] range map as the DiT's MIN_MAX
    (plain mean/std gave |x| up to 13.6 on the rot6d diagonal dims)
  - gripper dims (action 18/19, state 18/19) and T_LR_valid (state 29): mean 0 / std 1 and min 0 / max 1, i.e. NOT standardised:
    ACT (MEAN_STD) sees them in [0, 1]; DiT (MIN_MAX) maps [0, 1] -> [-1, 1]
  - 10-23: with an action_mask column only the loss-active targets of each dim enter the action stats (same chunk-mask logic)
ds.meta.stats is replaced before lerobot_train builds the processors, so these are exactly what the policy uses (verified from the
saved policy_preprocessor_*normalizer*.safetensors by training/tagrig_norm_audit.py --ckpt). Cached as
<root>/meta/tagrig_fixed_stats_<kind>_v4.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .layout import ACTION_DIM, GRIP_A, POS_L, POS_R, ROT_L, ROT_R, STAT_QUANTILES, STATE_UNSCALED
from .layout import ANCHOR_POS_L, ANCHOR_POS_R, ANCHOR_ROT_L, ANCHOR_ROT_R
from .rel_chunk import H_ACT, HORIZON, relativeWindows

STD_FLOOR = 1e-2                     # std floor of every standardised dim
RANGE_FLOOR = 2e-2                   # min-max range floor of every standardised dim
STD_FLAG = 1e-3                      # audit flag: raw std below this
CACHE_NAME = "tagrig_fixed_stats_{kind}_v4.json"
CACHE_NOTE = "train root only; floors + unscaled grippers, see training/tagrig_common.py"


def _dimStats(values, mask=None) -> dict:
    """per-dim stats; with mask (same shape, 0/1) only the loss-active entries of each dim are used
    (10-23: masked targets excluded)"""
    if mask is None:
        tStats = dict(min=values.min(0), max=values.max(0), mean=values.mean(0), std=values.std(0), count=np.array([len(values)]))
        for tQ in STAT_QUANTILES:
            tStats[f"q{tQ:02d}"] = np.percentile(values, tQ, axis=0)
        return tStats
    tCols = [values[mask[:, tD] > 0, tD] if (mask[:, tD] > 0).any() else values[:, tD] for tD in range(values.shape[1])]
    tFns = (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))
    tStats = {tK: np.array([tF(tC) for tC in tCols]) for tK, tF in tFns}
    tStats["count"] = np.array([len(values)])
    tStats["active_frac"] = (mask > 0).mean(0)
    for tQ in STAT_QUANTILES:
        tStats[f"q{tQ:02d}"] = np.array([np.percentile(tC, tQ) for tC in tCols])
    return tStats


def fixedStats(values, unscaled, mask=None):
    """raw per-dim stats (loss-active entries only when mask is given) + floors; unscaled dims get mean 0 / std 1 / min 0 / max 1
    -> (stats, audit rows)"""
    tRaw = _dimStats(values, mask)
    tStats = {tK: tV.copy() for tK, tV in tRaw.items() if tK != "active_frac"}
    tAudit = []
    for tD in range(values.shape[1]):
        tFix = []
        if tD in unscaled:
            tStats["mean"][tD], tStats["std"][tD], tStats["min"][tD], tStats["max"][tD] = 0.0, 1.0, 0.0, 1.0
            tFix.append("unscaled (0/1)")
        else:
            if tRaw["std"][tD] < STD_FLOOR:
                tStats["std"][tD] = STD_FLOOR
                tFix.append(f"std floor {STD_FLOOR}")
            if tRaw["max"][tD] - tRaw["min"][tD] < RANGE_FLOOR:
                tCentre = (tRaw["max"][tD] + tRaw["min"][tD]) / 2
                tStats["min"][tD], tStats["max"][tD] = tCentre - RANGE_FLOOR / 2, tCentre + RANGE_FLOOR / 2
                tFix.append(f"range floor {RANGE_FLOOR}")
        tAudit.append(dict(dim=tD, mean=tRaw["mean"][tD], std=tRaw["std"][tD], min=tRaw["min"][tD], max=tRaw["max"][tD],
                           p1=tRaw["q01"][tD], p99=tRaw["q99"][tD], flag_std_lt_1e3=bool(tRaw["std"][tD] < STD_FLAG),
                           fix=", ".join(tFix)))
    return {tK: (tV.astype(np.float32) if tK != "count" else tV) for tK, tV in tStats.items()}, tAudit


def _windowMasks(actionMask, anchorOk, episodeIndex, kind: str, stride: int) -> np.ndarray:
    """loss-active entries of every relative window (the chunk mask of loss_masks.chunkMask, on numpy) -> (N * H, 20)"""
    tOut = []
    for tEpisode in np.unique(episodeIndex):
        tSel = np.flatnonzero(episodeIndex == tEpisode)
        tN = len(tSel)
        for tT in range(0, tN, stride):
            tRows = np.clip(np.arange(tT, tT + H_ACT) if kind == "act" else np.arange(tT - 1, tT + HORIZON - 1), 0, tN - 1)
            tM = actionMask[tSel[tRows]].copy()
            tAo = anchorOk[tSel[tT]]
            tM[:, POS_L] = np.minimum(tM[:, POS_L], tAo[ANCHOR_POS_L]) * (tAo[ANCHOR_ROT_L] > 0)
            tM[:, ROT_L] = np.minimum(tM[:, ROT_L], tAo[ANCHOR_ROT_L])
            tM[:, POS_R] = np.minimum(tM[:, POS_R], tAo[ANCHOR_POS_R]) * (tAo[ANCHOR_ROT_R] > 0)
            tM[:, ROT_R] = np.minimum(tM[:, ROT_R], tAo[ANCHOR_ROT_R])
            if kind != "act":
                tM[0] = 1.0
            tOut.append(tM)
    return np.stack(tOut).reshape(-1, ACTION_DIM)


def _rangeMap(stats, audit, dims, note):
    """ACT: mean = (max + min) / 2, std = (max - min) / 2 on `dims` -> the DiT's [-1, 1] range map"""
    stats["mean"][dims] = (stats["max"][dims] + stats["min"][dims]) / 2
    stats["std"][dims] = (stats["max"][dims] - stats["min"][dims]) / 2
    for tD in dims:
        audit[tD]["fix"] = (audit[tD]["fix"] + ", " if audit[tD]["fix"] else "") + note


def computeAll(root, kind: str, stride: int = 1) -> dict:
    """train-root arrays -> dict(ds, A, P, S, epi, W, MW, action_stats, state_stats, action_audit, state_audit)
    (was compute_all)"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tDs = LeRobotDataset(repo_id="local/" + pathlib.Path(root).name, root=root)
    tHf = tDs.hf_dataset.with_format("numpy")
    tA, tP, tS, tEpi = (np.asarray(tHf[tK]) for tK in ("action", "pose_abs", "observation.state", "episode_index"))
    tW = relativeWindows(tA, tP, tEpi, kind, stride)
    tMw = None
    if "action_mask" in tDs.features:                                     # 10-23: stats from loss-active targets only
        tMw = _windowMasks(np.asarray(tHf["action_mask"]), np.asarray(tHf["anchor_ok"]), tEpi, kind, stride)
    tActionStats, tActionAudit = fixedStats(tW.reshape(-1, ACTION_DIM), GRIP_A, tMw)
    tStateStats, tStateAudit = fixedStats(tS.astype(np.float64), STATE_UNSCALED)
    if kind == "act":
        # audit 10-22: MEAN_STD put 472 values > 10 on the rot6d diagonal dims (mean ~1, long tail) -> ACT gets the DiT's
        # [-1, 1] range map: mean = (max + min) / 2, std = (max - min) / 2 (grippers stay 0/1)
        _rangeMap(tActionStats, tActionAudit, [tD for tD in range(ACTION_DIM) if tD not in GRIP_A],
                  "range map (mean=(max+min)/2, std=(max-min)/2)")
        # final-build gate 10-22: the anchor-state rot6d diagonal (Ranc_r4, ~1 with a long tail) reached |x| = 10.0 under mean/std
        # -> same range map for the ACT state (grippers and T_LR_valid stay unscaled)
        tStateDims = [tD for tD in range(tStateStats["mean"].shape[0]) if tD not in STATE_UNSCALED]
        _rangeMap(tStateStats, tStateAudit, tStateDims, "range map")
    return dict(ds=tDs, A=tA, P=tP, S=tS, epi=tEpi, W=tW, MW=tMw, action_stats=tActionStats, state_stats=tStateStats,
                action_audit=tActionAudit, state_audit=tStateAudit)


def installStats(ds, kind: str, stride: int = 1):
    """ds.meta.stats['action' / 'observation.state'] <- the fixed stats of the train root (computed once, then cached)"""
    tRoot = pathlib.Path(ds.root)
    tCache = tRoot / "meta" / CACHE_NAME.format(kind=kind)
    if tCache.exists():
        tCached = json.loads(tCache.read_text())
        tConv = lambda d: {tK: np.asarray(tV, np.float32 if tK != "count" else np.int64) for tK, tV in d.items()}
        tActionStats, tStateStats = tConv(tCached["action"]), tConv(tCached["state"])
    else:
        tAll = computeAll(tRoot, kind, stride)
        tActionStats, tStateStats = tAll["action_stats"], tAll["state_stats"]
        tCache.write_text(json.dumps({"action": {tK: tV.tolist() for tK, tV in tActionStats.items()},
                                      "state": {tK: tV.tolist() for tK, tV in tStateStats.items()}, "note": CACHE_NOTE}))
    for tKey, tStats in (("action", tActionStats), ("observation.state", tStateStats)):
        ds.meta.stats[tKey] = {**ds.meta.stats.get(tKey, {}), **tStats}
    return ds
