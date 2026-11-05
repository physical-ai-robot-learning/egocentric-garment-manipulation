"""Exp 2 ([2026-10-28], was training/train_dit_mix.py): Multi-Task DiT pretrain on the Exp 3 X-VLA mixture (Soft-Fold + iPhUMI +
PrimeUMI, 1/3 share each), reading the Exp 3 LeRobot sets (schema softfold_cart20_h32_bin_lerobot/v1, already QA'd) WITHOUT
re-encoding any video (training stage).

Those sets store, per 15 Hz row t:
  observation.state  RELCART20 task anchor [L pose9 | R pose9 | gL | gR]  (per arm inv(T(t_task_start)) @ T(t), rot6d = first two
  ROWS) action (32 x 32)   row k-1 = [L xyz3 rot6d6 g | R xyz3 rot6d6 g | AUX12 = 0],  L/R pose = inv(T(t)) @ T(t + k * 0.05005
  s), k = 1..32
which is exactly the DiT contract chunk (rel_chunk: row k = inv(T(t)) @ T(t+k) at 20 Hz, same rot6d rows convention).
Per sample this wrapper builds
  observation.state (2, 30)  contract proprio = state20 + T_LR identity pose9 + T_LR_valid 0 (no shared L/R frame in these sets)
  action (33, 20)            row 0 = identity poses + grippers(t) (as rel_chunk.toContract33), rows 1..32 = the stored chunk
                             reordered to [L pose9 | R pose9 | gL | gR]; action_is_pad all False (rows already exclude the chunk
                             tail)
  images (2, ...)            the stored 320x240 global / left_wrist / right_wrist streams at [t - 1 row, t] (global = black for
  UMI)
Known difference to the GOLD fine-tune data: the observation history step is one 15 Hz row (66.7 ms) instead of one 20 Hz frame
(50 ms).
Sampling: weighted (share / rows of that source), replacing lerobot's EpisodeAwareSampler. Action / state stats: exact min / max
over all rows, mean / std / quantiles over every 10th row, cached in $MIX_STATS. Camera stats = ImageNet (use_imagenet_stats).

    MIX_SPEC='[{"name":..,"root":..,"share":..},..]' MIX_STATS=<json> python -m laundry_training.train_dit_mix <stock
    lerobot-train flags>
"""
from __future__ import annotations

import bisect
import copy
import json
import os
import pathlib
import sys

import numpy as np
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import IMAGENET_STATS

from laundry_shared.contracts.action_contract import ACTION_NAMES, T_to_pose9
from laundry_shared.contracts.observation_contract import PROPRIO_NAMES

from .layout import ACTION_DIM, GRIPS, STAT_QUANTILES
from .rel_chunk import HORIZON

N_OBS_STEPS = 2
STATE_DIM = 30
ID9 = T_to_pose9(np.eye(4)).astype(np.float32)
REORDER = list(range(0, 9)) + list(range(10, 19)) + [9, 19]      # X-VLA [L9 gL R9 gR] -> contract [L9 R9 gL gR]
TLR_PAD = np.concatenate([ID9, [0.0]]).astype(np.float32)        # T_LR identity + valid 0
STATS_SUBSAMPLE = 10                 # mean / std / quantiles over every 10th row
STATS_BLOCK = 20000                  # rows read per block while computing the stats
SAMPLER_SEED = 1000
_MIX = dict(ds=None)                 # the dataset of this run, for weightedSampler (was MIX_DS)


def state30(state20) -> np.ndarray:
    tS = np.asarray(state20, np.float32)
    return np.concatenate([tS, np.broadcast_to(TLR_PAD, tS.shape[:-1] + (len(TLR_PAD),))], -1)


def chunk33(chunk32, gNow) -> np.ndarray:
    tC = np.asarray(chunk32, np.float32)[:, :ACTION_DIM][:, REORDER]
    tRow0 = np.concatenate([ID9, ID9, np.asarray(gNow, np.float32)])
    return np.concatenate([tRow0[None], tC], 0)


def _stats(allMin, allMax, sub) -> dict:
    tStats = dict(min=allMin, max=allMax, mean=sub.mean(0), std=sub.std(0), count=np.array([len(sub)]))
    for tQ in STAT_QUANTILES:
        tStats[f"q{tQ:02d}"] = np.percentile(sub, tQ, axis=0)
    return {tK: (tV.astype(np.float32) if tK != "count" else tV) for tK, tV in tStats.items()}


def mixStats(dsets, path, sub: int = STATS_SUBSAMPLE, blk: int = STATS_BLOCK) -> dict:
    """action / state stats over every source (exact min / max, the rest over every `sub`-th row), cached at `path`
    (was mix_stats)"""
    tPath = pathlib.Path(path)
    if tPath.exists():
        tCached = json.loads(tPath.read_text())
        return {tKey: {tK: np.asarray(tV, np.int64 if tK == "count" else np.float32) for tK, tV in tD.items()}
                for tKey, tD in tCached.items()}
    tAmin = tAmax = tSmin = tSmax = None
    tAsub, tSsub = [], []
    for tDs in dsets:
        tHf = tDs.hf_dataset.with_format("numpy")
        tN = len(tHf)
        for tA0 in range(0, tN, blk):
            tB = tHf[tA0:min(tA0 + blk, tN)]
            tS = state30(tB["observation.state"])
            tA = np.stack([chunk33(tChunk, tStateRow[GRIPS]) for tChunk, tStateRow in zip(tB["action"], tB["observation.state"])])
            tA2 = tA.reshape(-1, tA.shape[-1])
            tAmin = tA2.min(0) if tAmin is None else np.minimum(tAmin, tA2.min(0))
            tAmax = tA2.max(0) if tAmax is None else np.maximum(tAmax, tA2.max(0))
            tSmin = tS.min(0) if tSmin is None else np.minimum(tSmin, tS.min(0))
            tSmax = tS.max(0) if tSmax is None else np.maximum(tSmax, tS.max(0))
            tAsub.append(tA[::sub].reshape(-1, tA.shape[-1]))
            tSsub.append(tS[::sub])
        print(f"[mix stats] {tDs.root} rows {tN}", flush=True)
    tOut = {"action": _stats(tAmin, tAmax, np.concatenate(tAsub).astype(np.float64)),
            "observation.state": _stats(tSmin, tSmax, np.concatenate(tSsub).astype(np.float64))}
    tPath.write_text(json.dumps({tKey: {tK: tV.tolist() for tK, tV in tD.items()} for tKey, tD in tOut.items()}))
    return tOut


class MixDataset(torch.utils.data.Dataset):
    """the sources concatenated; per item the contract state30 / action33 of the module docstring (meta / episodes / num_frames /
    num_episodes are the LeRobotDataset names lerobot_train reads)"""
    KEEP = ("task", "index", "episode_index", "frame_index", "timestamp", "task_index")

    def __init__(self, dsets, shares, stats):
        self.mDsets, self.mShares = dsets, np.asarray(shares, np.float64) / np.sum(shares)
        self.mOffsets = np.cumsum([0] + [len(tD) for tD in dsets]).tolist()
        self.mCams = list(dsets[0].meta.camera_keys)
        tMeta = copy.copy(dsets[0].meta)
        tMeta.info = copy.deepcopy(tMeta.info)
        tFeatures = tMeta.info["features"]
        for tKey in [tK for tK in tFeatures if tK.startswith("aux.") or tK == "state_prev_valid"]:
            del tFeatures[tKey]
        tFeatures["observation.state"] = dict(dtype="float32", shape=[STATE_DIM], names=list(PROPRIO_NAMES))
        tFeatures["action"] = dict(dtype="float32", shape=[ACTION_DIM], names=list(ACTION_NAMES))
        tMeta.stats = {tK: copy.deepcopy(tV) for tK, tV in dsets[0].meta.stats.items() if tK in self.mCams}
        for tKey in self.mCams:
            for tStat, tVal in IMAGENET_STATS.items():
                tMeta.stats[tKey][tStat] = torch.tensor(tVal, dtype=torch.float32)
        tMeta.stats.update(stats)
        self.meta = tMeta
        self.episodes = None

    @property
    def num_frames(self):
        return self.mOffsets[-1]

    @property
    def num_episodes(self):
        return sum(tD.num_episodes for tD in self.mDsets)

    def __len__(self):
        return self.mOffsets[-1]

    def sampleWeights(self):
        return torch.cat([torch.full((len(tD),), tS / len(tD), dtype=torch.float64) for tD, tS in zip(self.mDsets, self.mShares)])

    def __getitem__(self, i):
        tSource = bisect.bisect_right(self.mOffsets, i) - 1
        tItem = self.mDsets[tSource][i - self.mOffsets[tSource]]
        tState = tItem["observation.state"].numpy()
        tOut = {tK: tItem[tK] for tK in self.KEEP if tK in tItem}
        for tKey in self.mCams:
            tOut[tKey] = tItem[tKey]
            tOut[f"{tKey}_is_pad"] = tItem[f"{tKey}_is_pad"]
        tOut["observation.state"] = torch.from_numpy(state30(tState))
        tOut["observation.state_is_pad"] = tItem["observation.state_is_pad"]
        tOut["action"] = torch.from_numpy(chunk33(tItem["action"].numpy(), tState[-1, GRIPS]))
        tOut["action_is_pad"] = torch.zeros(HORIZON, dtype=torch.bool)
        tOut["mix_source"] = torch.tensor(tSource)
        return tOut


def makeDatasetMix(cfg):
    """lerobot_train.make_dataset replacement: the MIX_SPEC sources as one MixDataset (was make_dataset_mix)"""
    if getattr(cfg.policy, "horizon", None) != HORIZON or getattr(cfg.policy, "n_obs_steps", None) != N_OBS_STEPS:
        raise SystemExit(f"contract chunk needs policy.horizon={HORIZON} and policy.n_obs_steps={N_OBS_STEPS}")
    tSpec = json.loads(os.environ["MIX_SPEC"])
    tTransforms = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    tDsets = []
    for tSrc in tSpec:
        tMeta = LeRobotDatasetMetadata(f"local/{tSrc['name']}", root=tSrc["root"])
        tDelta = {tK: [-1.0 / tMeta.fps, 0.0] for tK in tMeta.camera_keys}
        tDelta["observation.state"] = [-1.0 / tMeta.fps, 0.0]
        tDsets.append(LeRobotDataset(f"local/{tSrc['name']}", root=tSrc["root"], delta_timestamps=tDelta,
                                     image_transforms=tTransforms, video_backend=cfg.dataset.video_backend, return_uint8=True,
                                     tolerance_s=cfg.tolerance_s))
        print(f"[mix] {tSrc['name']} share {tSrc['share']} rows {len(tDsets[-1])} eps {tDsets[-1].num_episodes} fps {tMeta.fps}",
              flush=True)
    _MIX["ds"] = MixDataset(tDsets, [tS["share"] for tS in tSpec], mixStats(tDsets, os.environ["MIX_STATS"]))
    return _MIX["ds"]


def weightedSampler(*_, **__):
    """lerobot_train.EpisodeAwareSampler replacement: rows drawn with probability share / rows of their source
    (was weighted_sampler)"""
    tGen = torch.Generator()
    tGen.manual_seed(SAMPLER_SEED)
    return torch.utils.data.WeightedRandomSampler(_MIX["ds"].sampleWeights(), num_samples=len(_MIX["ds"]), replacement=True,
                                                  generator=tGen)


def installPatches():
    """lerobot_train with the mixture dataset and the weighted sampler -> lerobot_train"""
    import lerobot.scripts.lerobot_train as LT
    LT.make_dataset = makeDatasetMix
    LT.EpisodeAwareSampler = weightedSampler
    return LT


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]         # lerobot_train parses sys.argv itself
    installPatches().main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
