#!/usr/bin/env python3
"""[2026-10-28] Exp 2: Multi-Task DiT pretrain on the Exp 3 X-VLA mixture (Soft-Fold + iPhUMI + PrimeUMI, 1/3 share each),
reading the Exp 3 LeRobot sets (schema softfold_cart20_h32_bin_lerobot/v1, already QA'd) WITHOUT re-encoding any video.

Those sets store, per 15 Hz row t:
  observation.state  RELCART20 task anchor [L pose9 | R pose9 | gL | gR]  (per arm inv(T(t_task_start)) @ T(t), rot6d = first two ROWS)
  action (32 x 32)   row k-1 = [L xyz3 rot6d6 g | R xyz3 rot6d6 g | AUX12 = 0],  L/R pose = inv(T(t)) @ T(t + k * 0.05005 s), k = 1..32
which is exactly the DiT contract chunk (training/rel_chunk.py: row k = inv(T(t)) @ T(t+k) at 20 Hz, same rot6d rows convention).
Per sample this wrapper builds
  observation.state (2, 30)  contract proprio = state20 + T_LR identity pose9 + T_LR_valid 0 (no shared L/R frame in these sets)
  action (33, 20)            row 0 = identity poses + grippers(t) (as rel_chunk.to_contract33), rows 1..32 = the stored chunk
                             reordered to [L pose9 | R pose9 | gL | gR]; action_is_pad all False (rows already exclude the chunk tail)
  images (2, ...)            the stored 320x240 global / left_wrist / right_wrist streams at [t - 1 row, t] (global = black for UMI)
Known difference to the GOLD fine-tune data: the observation history step is one 15 Hz row (66.7 ms) instead of one 20 Hz frame (50 ms).
Sampling: weighted (share / rows of that source), replacing lerobot's EpisodeAwareSampler.  Action / state stats: exact min / max over
all rows, mean / std / quantiles over every 10th row, cached in $MIX_STATS.  Camera stats = ImageNet (use_imagenet_stats).
    MIX_SPEC='[{"name":..,"root":..,"share":..},..]' MIX_STATS=<json> python training/train_dit_mix.py <stock lerobot-train flags>"""
import copy, bisect, json, os, pathlib, sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
import lerobot.scripts.lerobot_train as LT  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.transforms import ImageTransforms  # noqa: E402
from lerobot.utils.constants import IMAGENET_STATS  # noqa: E402
from garment_pipeline.training.rel_chunk import HORIZON  # noqa: E402
from contracts.action_contract import ACTION_NAMES, T_to_pose9  # noqa: E402
from contracts.observation_contract import PROPRIO_NAMES  # noqa: E402

ID9 = T_to_pose9(np.eye(4)).astype(np.float32)
REORDER = list(range(0, 9)) + list(range(10, 19)) + [9, 19]      # X-VLA [L9 gL R9 gR] -> contract [L9 R9 gL gR]
TLR_PAD = np.concatenate([ID9, [0.0]]).astype(np.float32)        # T_LR identity + valid 0
MIX_DS = None


def state30(s20):
    s20 = np.asarray(s20, np.float32)
    return np.concatenate([s20, np.broadcast_to(TLR_PAD, s20.shape[:-1] + (10,))], -1)


def chunk33(chunk32, g_now):
    c = np.asarray(chunk32, np.float32)[:, :20][:, REORDER]
    row0 = np.concatenate([ID9, ID9, np.asarray(g_now, np.float32)])
    return np.concatenate([row0[None], c], 0)


def _stats(R_all_min, R_all_max, R_sub):
    st = dict(min=R_all_min, max=R_all_max, mean=R_sub.mean(0), std=R_sub.std(0), count=np.array([len(R_sub)]))
    for q in (1, 10, 50, 90, 99): st[f"q{q:02d}"] = np.percentile(R_sub, q, axis=0)
    return {k: (v.astype(np.float32) if k != "count" else v) for k, v in st.items()}


def mix_stats(dsets, path, sub=10, blk=20000):
    p = pathlib.Path(path)
    if p.exists():
        j = json.loads(p.read_text())
        return {key: {k: np.asarray(v, np.int64 if k == "count" else np.float32) for k, v in d.items()} for key, d in j.items()}
    amin = amax = smin = smax = None; asub, ssub = [], []
    for ds in dsets:
        hf = ds.hf_dataset.with_format("numpy"); n = len(hf)
        for a in range(0, n, blk):
            b = hf[a:min(a + blk, n)]; S = state30(b["observation.state"]); A = np.stack([chunk33(c, s[18:20]) for c, s in zip(b["action"], b["observation.state"])])
            A2 = A.reshape(-1, A.shape[-1])
            amin = A2.min(0) if amin is None else np.minimum(amin, A2.min(0)); amax = A2.max(0) if amax is None else np.maximum(amax, A2.max(0))
            smin = S.min(0) if smin is None else np.minimum(smin, S.min(0)); smax = S.max(0) if smax is None else np.maximum(smax, S.max(0))
            asub.append(A[::sub].reshape(-1, A.shape[-1])); ssub.append(S[::sub])
        print(f"[mix stats] {ds.root} rows {n}", flush=True)
    out = {"action": _stats(amin, amax, np.concatenate(asub).astype(np.float64)),
           "observation.state": _stats(smin, smax, np.concatenate(ssub).astype(np.float64))}
    p.write_text(json.dumps({key: {k: v.tolist() for k, v in d.items()} for key, d in out.items()}))
    return out


class MixDataset(torch.utils.data.Dataset):
    KEEP = ("task", "index", "episode_index", "frame_index", "timestamp", "task_index")

    def __init__(self, dsets, shares, stats):
        self.dsets, self.shares = dsets, np.asarray(shares, np.float64) / np.sum(shares)
        self.offsets = np.cumsum([0] + [len(d) for d in dsets]).tolist()
        self.cams = list(dsets[0].meta.camera_keys)
        meta = copy.copy(dsets[0].meta); meta.info = copy.deepcopy(meta.info); f = meta.info["features"]
        for k in [k for k in f if k.startswith("aux.") or k == "state_prev_valid"]: del f[k]
        f["observation.state"] = dict(dtype="float32", shape=[30], names=list(PROPRIO_NAMES))
        f["action"] = dict(dtype="float32", shape=[20], names=list(ACTION_NAMES))
        meta.stats = {k: copy.deepcopy(v) for k, v in dsets[0].meta.stats.items() if k in self.cams}
        for k in self.cams:
            for t, v in IMAGENET_STATS.items(): meta.stats[k][t] = torch.tensor(v, dtype=torch.float32)
        meta.stats.update(stats); self.meta = meta; self.episodes = None

    @property
    def num_frames(self): return self.offsets[-1]

    @property
    def num_episodes(self): return sum(d.num_episodes for d in self.dsets)

    def __len__(self): return self.offsets[-1]

    def sample_weights(self):
        return torch.cat([torch.full((len(d),), s / len(d), dtype=torch.float64) for d, s in zip(self.dsets, self.shares)])

    def __getitem__(self, i):
        s = bisect.bisect_right(self.offsets, i) - 1; it = self.dsets[s][i - self.offsets[s]]
        st = it["observation.state"].numpy()
        out = {k: it[k] for k in self.KEEP if k in it}
        for k in self.cams:
            out[k] = it[k]; out[f"{k}_is_pad"] = it[f"{k}_is_pad"]
        out["observation.state"] = torch.from_numpy(state30(st)); out["observation.state_is_pad"] = it["observation.state_is_pad"]
        out["action"] = torch.from_numpy(chunk33(it["action"].numpy(), st[-1, 18:20]))
        out["action_is_pad"] = torch.zeros(HORIZON, dtype=torch.bool)
        out["mix_source"] = torch.tensor(s)
        return out


def make_dataset_mix(cfg):
    global MIX_DS
    if getattr(cfg.policy, "horizon", None) != HORIZON or getattr(cfg.policy, "n_obs_steps", None) != 2:
        raise SystemExit(f"contract chunk needs policy.horizon={HORIZON} and policy.n_obs_steps=2")
    spec = json.loads(os.environ["MIX_SPEC"])
    tf = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    dsets = []
    for s in spec:
        m = LeRobotDatasetMetadata(f"local/{s['name']}", root=s["root"])
        dt = {k: [-1.0 / m.fps, 0.0] for k in m.camera_keys}; dt["observation.state"] = [-1.0 / m.fps, 0.0]
        dsets.append(LeRobotDataset(f"local/{s['name']}", root=s["root"], delta_timestamps=dt, image_transforms=tf,
                                    video_backend=cfg.dataset.video_backend, return_uint8=True, tolerance_s=cfg.tolerance_s))
        print(f"[mix] {s['name']} share {s['share']} rows {len(dsets[-1])} eps {dsets[-1].num_episodes} fps {m.fps}", flush=True)
    MIX_DS = MixDataset(dsets, [s["share"] for s in spec], mix_stats(dsets, os.environ["MIX_STATS"]))
    return MIX_DS


def weighted_sampler(*_, **__):
    g = torch.Generator(); g.manual_seed(1000)
    return torch.utils.data.WeightedRandomSampler(MIX_DS.sample_weights(), num_samples=len(MIX_DS), replacement=True, generator=g)


LT.make_dataset = make_dataset_mix
LT.EpisodeAwareSampler = weighted_sampler

if __name__ == "__main__":
    LT.main()
