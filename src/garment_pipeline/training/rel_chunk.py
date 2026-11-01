"""[2026-10-14 Agent B] Training-time contract chunk for LeRobot Multi-Task DiT.

The dataset stores per-frame ABSOLUTE next-frame targets (datasets/xvla_softfold_to_lerobot.py).  With n_obs_steps = 2 and
horizon = 33 the action window of sample t covers frames t-1 .. t+31, i.e. absolute poses T(t) .. T(t+32).  This module turns it into

    row 0      identity pose + gripper(t)          (the DiT's "previous step" row; dropped by the inference wrapper)
    row k      inv(T(t)) @ T(t+k), gripper(t+k)    k = 1..32  == Agent A contract chunk (contracts/action_contract.py)

and replaces dataset.meta.stats["action"] with statistics of these RELATIVE rows (min-max normalisation must see the relative
range, not absolute base-frame poses).  The policy's own executed slice starts at row n_obs_steps - 1 = 1, i.e. contract k = 1.
"""
import json, pathlib, sys

import numpy as np
import torch

from xvla_core import relative_chunk_from_abs  # noqa: E402
from contracts.action_contract import T_to_pose9  # noqa: E402

HORIZON = 33
_IDENTITY9 = T_to_pose9(np.eye(4)).astype(np.float32)


def to_contract33(chunk_abs):
    c = np.asarray(chunk_abs, np.float32); out = np.empty_like(c)
    out[1:] = relative_chunk_from_abs(c)
    out[0, 0:9] = _IDENTITY9; out[0, 9:18] = _IDENTITY9; out[0, 18:20] = c[0, 18:20]
    return out


def relative_action_stats(actions, episode_index, stride=1):
    """stats over every window (all t, padded like LeRobot: indices clamped to the episode) -> LeRobot stats dict"""
    rows = []
    for e in np.unique(episode_index):
        A = actions[episode_index == e]; n = len(A)
        for t in range(0, n, stride):
            idx = np.clip(np.arange(t - 1, t - 1 + HORIZON), 0, n - 1); rows.append(to_contract33(A[idx]))
    R = np.concatenate(rows, 0).astype(np.float64)
    st = dict(min=R.min(0), max=R.max(0), mean=R.mean(0), std=R.std(0), count=np.array([len(R)]))
    for q in (1, 10, 50, 90, 99): st[f"q{q:02d}"] = np.percentile(R, q, axis=0)
    return {k: v.astype(np.float32) if k != "count" else v for k, v in st.items()}


def patch_dataset(ds, stride=2):
    """in place: action windows -> contract rows; meta.stats['action'] -> relative stats (cached next to the dataset)"""
    root = pathlib.Path(ds.root); cache = root / "meta" / f"rel_action_stats_h{HORIZON}.json"
    if cache.exists():
        st = {k: np.asarray(v, np.float32 if k != "count" else np.int64) for k, v in json.loads(cache.read_text()).items()}
    else:
        hf = ds.hf_dataset.with_format("numpy")
        st = relative_action_stats(np.asarray(hf["action"]), np.asarray(hf["episode_index"]), stride=stride)
        cache.write_text(json.dumps({k: v.tolist() for k, v in st.items()}))
    ds.meta.stats["action"] = st
    base = ds.__class__

    class RelChunkDataset(base):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            a = item["action"]
            if a.ndim == 2 and a.shape[0] == HORIZON:
                item["action"] = torch.from_numpy(to_contract33(a.numpy()))
            return item

    RelChunkDataset.__name__ = base.__name__ + "RelChunk"; ds.__class__ = RelChunkDataset
    return ds
