"""Relative action chunks of the contract: absolute stored targets -> rows inv(T(t)) @ T(t+k) (training stage).

The dataset stores per-frame ABSOLUTE next-frame targets. Two windows are built from them:

  DiT  n_obs_steps 2, horizon 33: the action window of sample t covers frames t-1 .. t+31, i.e. absolute poses T(t) .. T(t+32);
       toContract33 turns it into
           row 0      identity pose + gripper(t)          (the DiT's "previous step" row; dropped by the inference wrapper)
           row k      inv(T(t)) @ T(t+k), gripper(t+k)    k = 1..32  == contract chunk (contracts/action_contract.py)
       The policy's own executed slice starts at row n_obs_steps - 1 = 1, i.e. contract k = 1.
  ACT  n_obs_steps 1, chunk 32: anchor T(t) = the frame's pose_abs, actions t..t+31 = T(t+1..t+32) -> rows k = 1..32 (actRelative)

patchDataset (2026-10-14, Agent B; was rel_chunk.patch_dataset) also replaces dataset.meta.stats["action"] with statistics of the
RELATIVE rows (min-max normalisation must see the relative range, not absolute base-frame poses).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from laundry_shared.contracts.action_contract import T_to_pose9
from laundry_shared.xvla_core import relative_chunk_from_abs

from .layout import GRIPS, POSE9_L, POSE9_R, STAT_QUANTILES

HORIZON = 33                         # DiT window: row 0 identity + rows k = 1..32
H_ACT = 32                           # ACT chunk: rows k = 1..32
STATS_CACHE_NAME = f"rel_action_stats_h{HORIZON}.json"
_IDENTITY9 = T_to_pose9(np.eye(4)).astype(np.float32)


def toContract33(chunkAbs) -> np.ndarray:
    """(33, 20) absolute rows T(t) .. T(t+32) -> (33, 20) row 0 identity + g(t), rows k = inv(T(t)) @ T(t+k)"""
    tChunk = np.asarray(chunkAbs, np.float32)
    tOut = np.empty_like(tChunk)
    tOut[1:] = relative_chunk_from_abs(tChunk)
    tOut[0, POSE9_L] = _IDENTITY9
    tOut[0, POSE9_R] = _IDENTITY9
    tOut[0, GRIPS] = tChunk[0, GRIPS]
    return tOut


def actRelative(poseAbsT, actions) -> np.ndarray:
    """ACT: anchor pose_abs(t) (20,) + stored next-frame actions t..t+31 (32, 20) -> contract rows k = 1..32
    (was tagrig_common.act_rel)"""
    return relative_chunk_from_abs(np.vstack([poseAbsT[None], actions]))


def relativeWindows(actions, poseAbs, episodeIndex, kind: str, stride: int = 1) -> np.ndarray:
    """every relative window of every episode (indices clamped to the episode like LeRobot's padding) -> (N, H, 20) float64.
    kind "dit": rows of toContract33 over frames t-1 .. t+31; otherwise ACT rows anchored at pose_abs(t) (was
    tagrig_common.windows)"""
    tOut = []
    for tEpisode in np.unique(episodeIndex):
        tSel = np.flatnonzero(episodeIndex == tEpisode)
        tA, tP = actions[tSel], poseAbs[tSel]
        tN = len(tSel)
        for tT in range(0, tN, stride):
            if kind == "dit":
                tOut.append(toContract33(tA[np.clip(np.arange(tT - 1, tT - 1 + HORIZON), 0, tN - 1)]))
            else:
                tOut.append(actRelative(tP[tT], tA[np.clip(np.arange(tT, tT + H_ACT), 0, tN - 1)]))
    return np.stack(tOut).astype(np.float64)


def relativeActionStats(actions, episodeIndex, stride: int = 1) -> dict:
    """stats over every DiT window (all t, padded like LeRobot: indices clamped to the episode) -> LeRobot stats dict"""
    tRows = []
    for tEpisode in np.unique(episodeIndex):
        tA = actions[episodeIndex == tEpisode]
        tN = len(tA)
        for tT in range(0, tN, stride):
            tIdx = np.clip(np.arange(tT - 1, tT - 1 + HORIZON), 0, tN - 1)
            tRows.append(toContract33(tA[tIdx]))
    tR = np.concatenate(tRows, 0).astype(np.float64)
    tStats = dict(min=tR.min(0), max=tR.max(0), mean=tR.mean(0), std=tR.std(0), count=np.array([len(tR)]))
    for tQ in STAT_QUANTILES:
        tStats[f"q{tQ:02d}"] = np.percentile(tR, tQ, axis=0)
    return {tK: tV.astype(np.float32) if tK != "count" else tV for tK, tV in tStats.items()}


def patchDataset(ds, stride: int = 2):
    """in place: action windows -> contract rows; meta.stats['action'] -> relative stats (cached next to the dataset)"""
    tCache = pathlib.Path(ds.root) / "meta" / STATS_CACHE_NAME
    if tCache.exists():
        tCached = json.loads(tCache.read_text())
        tStats = {tK: np.asarray(tV, np.float32 if tK != "count" else np.int64) for tK, tV in tCached.items()}
    else:
        tHf = ds.hf_dataset.with_format("numpy")
        tStats = relativeActionStats(np.asarray(tHf["action"]), np.asarray(tHf["episode_index"]), stride=stride)
        tCache.write_text(json.dumps({tK: tV.tolist() for tK, tV in tStats.items()}))
    ds.meta.stats["action"] = tStats
    tBase = ds.__class__

    class RelChunkDataset(tBase):
        def __getitem__(self, idx):
            tItem = super().__getitem__(idx)
            tAction = tItem["action"]
            if tAction.ndim == 2 and tAction.shape[0] == HORIZON:
                tItem["action"] = torch.from_numpy(toContract33(tAction.numpy()))
            return tItem

    RelChunkDataset.__name__ = tBase.__name__ + "RelChunk"
    ds.__class__ = RelChunkDataset
    return ds
