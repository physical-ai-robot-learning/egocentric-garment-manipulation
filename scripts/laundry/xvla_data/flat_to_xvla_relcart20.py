#!/usr/bin/env python3
"""[2026-11-10 user] Tag-rig flatten set (S03G, DiT layout) -> X-VLA RELCART20 REL32 set for the SoftFold 650k fine-tune
("X-VLA SoftFold -> UMI300 T-shirt flattening SFT"; user: continuous gripper, jaw state 0 (GSM), no grad clip).

Same rows, same videos, same episodes as the source split (only data parquet + meta are rewritten; videos are linked):
  axes     X-VLA dataset-TCP axes T_x = T_A @ Ry(-90) (laundry_deploy.core.xvla_runner.XvlaRunner.RY, the Exp 3 convention)
  action   (32, 32) per row t: k = 1..32 -> [L xyz rot6d g | R xyz rot6d g | 0 x 12]
           pose = inv(T_x(t)) @ T_x(t+k) with T(t) = pose_abs[t], T(t+k) = action[t+k-1] (the ACT window of rel_chunk; the
           episode tail repeats its last target), g = action gripper at t+k-1 (continuous, 0 = closed, 1 = open), rot6d = rows
  state    (20,) RELCART20 task anchor [L pose9 | R pose9 | gL gR] = observation.state[0:18] (episode-anchor poses) in X axes,
           grippers 0 (GSM: the DiT/Exp 3 jaw-state mask, the deployed xvla_runner zeroes them for -GSM checkpoints)
  mask     aux.loss_mask (20,) in the action layout, per sample (rel16_relonly_lossmask): xyz = anchor pos_ok & rot_ok and
           every target pos_ok of k = 1..32; rot6d = anchor rot_ok and every target rot_ok; gripper = every target jaw ok
  task     "Flatten and align the T-shirt."
Stats: action / state mean, std, min, max over the TRAIN split (val/test get the train stats).
usage: python datasets/flat_to_xvla_relcart20.py SRC_ROOT DST_ROOT [--stats-from TRAIN_DST_ROOT]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

from laundry_shared.contracts.action_contract import T_to_pose9, inv_T, make_T, pose9_to_T

TASK = "Flatten and align the T-shirt."
H = 32
RY = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
RH = make_T(RY, np.zeros(3))
SRC_POSE = {"left": slice(0, 9), "right": slice(9, 18)}
SRC_POS = {"left": slice(0, 3), "right": slice(9, 12)}
SRC_ROT = {"left": slice(3, 9), "right": slice(12, 18)}
SRC_GRIP = {"left": 18, "right": 19}
ANCHOR = {"left": (0, 1), "right": (2, 3)}  # anchor_ok [pos_ok_L, rot_ok_L, pos_ok_R, rot_ok_R]
X_OFF = {"left": 0, "right": 10}  # RELCART20 arm block [xyz 3 | rot6d 6 | g]
KEEP = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def toX(T):
    return inv_T(RH) @ T @ RH


def convertEpisode(d: pd.DataFrame):
    tN = len(d)
    tPose = np.stack(d["pose_abs"].values).astype(np.float64)
    tAct = np.stack(d["action"].values).astype(np.float64)
    tMask = np.stack(d["action_mask"].values) > 0.5
    tAnc = np.stack(d["anchor_ok"].values) > 0.5
    tS30 = np.stack(d["observation.state"].values).astype(np.float64)
    tIdx = np.minimum(np.arange(tN)[:, None] + np.arange(H)[None], tN - 1)  # (N, 32): frame of target k = t + k - 1
    tOut = np.zeros((tN, H, 32), np.float32)
    tLoss = np.zeros((tN, 20), np.float32)
    tState = np.zeros((tN, 20), np.float32)
    for tSide in ("left", "right"):
        tTt = toX(pose9_to_T(tPose[:, SRC_POSE[tSide]]))  # (N, 4, 4)
        tTf = toX(pose9_to_T(tAct[:, SRC_POSE[tSide]]))[tIdx]  # (N, 32, 4, 4)
        o = X_OFF[tSide]
        tOut[:, :, o:o + 9] = T_to_pose9(inv_T(tTt)[:, None] @ tTf)
        tOut[:, :, o + 9] = tAct[tIdx, SRC_GRIP[tSide]]
        tPosOk, tRotOk = tAnc[:, ANCHOR[tSide][0]], tAnc[:, ANCHOR[tSide][1]]
        tLoss[:, o:o + 3] = (tPosOk & tRotOk & tMask[tIdx][:, :, SRC_POS[tSide]].all((1, 2)))[:, None]
        tLoss[:, o + 3:o + 9] = (tRotOk & tMask[tIdx][:, :, SRC_ROT[tSide]].all((1, 2)))[:, None]
        tLoss[:, o + 9] = tMask[tIdx][:, :, SRC_GRIP[tSide]].all(1)
        s = 0 if tSide == "left" else 9
        tState[:, s:s + 9] = T_to_pose9(toX(pose9_to_T(tS30[:, SRC_POSE[tSide]])))
    return tOut, tState, tLoss


def stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(), min=x.min(0).tolist(), max=x.max(0).tolist(),
                count=[int(len(x))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--stats-from", default=None)
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    assert not dst.exists(), f"{dst} exists"
    (dst / "meta").mkdir(parents=True)
    tActs, tStates, tLosses = [], [], []
    for tF in sorted(glob.glob(str(src / "data/*/*.parquet"))):
        d = pd.read_parquet(tF)
        tParts = []
        for tEp, tD in d.groupby("episode_index", sort=False):
            tD = tD.sort_values("frame_index")
            assert (np.diff(tD["frame_index"].values) == 1).all(), f"episode {tEp}: frames not consecutive"
            tA, tS, tL = convertEpisode(tD)
            tOut = tD[KEEP].copy()
            tOut["task_index"] = 0
            tOut["observation.state"] = list(tS)
            tOut["action"] = [r.tolist() for r in tA]
            tOut["aux.loss_mask"] = list(tL)
            tParts.append(tOut)
            tActs.append(tA); tStates.append(tS); tLosses.append(tL)
        tNew = pd.concat(tParts).sort_values("index")
        assert (tNew["index"].values == d.sort_values("index")["index"].values).all()
        tDst = dst / os.path.relpath(tF, src); tDst.parent.mkdir(parents=True, exist_ok=True)
        tNew.to_parquet(tDst, index=False)
        print(f"{tF} -> {tDst} ({len(tNew)} rows)", flush=True)
    info = json.loads((src / "meta/info.json").read_text())
    tFeat = {k: v for k, v in info["features"].items() if k.startswith("observation.images.") or k in KEEP}
    tFeat["observation.state"] = dict(dtype="float32", shape=[20], names=None)
    tFeat["action"] = dict(dtype="float32", shape=[H, 32], names=None)
    tFeat["aux.loss_mask"] = dict(dtype="float32", shape=[20], names=None)
    info["features"] = tFeat; info["total_tasks"] = 1
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    for tF in sorted(glob.glob(str(src / "meta/episodes/*/*.parquet"))):
        e = pd.read_parquet(tF)
        e = e[[c for c in e.columns if not c.startswith("stats/") or c.startswith("stats/observation.images.")]]
        e["tasks"] = [[TASK] for _ in range(len(e))]
        tDst = dst / os.path.relpath(tF, src); tDst.parent.mkdir(parents=True, exist_ok=True); e.to_parquet(tDst, index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(dst / "meta/tasks.parquet")
    for tName in ("split.json",):
        if (src / "meta" / tName).exists(): shutil.copy(src / "meta" / tName, dst / "meta" / tName)
    tA = np.concatenate(tActs); tS = np.concatenate(tStates); tL = np.concatenate(tLosses)
    srcStats = json.loads((src / "meta/stats.json").read_text())
    st = {k: v for k, v in srcStats.items() if k.startswith("observation.images.") or k in KEEP}
    if a.stats_from:
        tTrain = json.loads((pathlib.Path(a.stats_from) / "meta/stats.json").read_text())
        st["action"], st["observation.state"], st["aux.loss_mask"] = tTrain["action"], tTrain["observation.state"], tTrain["aux.loss_mask"]
    else:
        st["action"], st["observation.state"], st["aux.loss_mask"] = stats(tA), stats(tS), stats(tL)
    (dst / "meta/stats.json").write_text(json.dumps(st))
    if (src / "videos").exists():
        os.system(f"cp -al '{src}/videos' '{dst}/videos'")
    k = tA[:, :, :20]
    print(json.dumps(dict(rows=int(len(tA)), loss_mask_mean=tL.mean(0).round(3).tolist(),
                          k32_trans_mm_p50=[float(np.median(np.linalg.norm(k[:, 31, o:o + 3], axis=1)) * 1e3) for o in (0, 10)],
                          grip_open_frac=[float((k[:, 0, 9] > 0.5).mean()), float((k[:, 0, 19] > 0.5).mean())])), flush=True)


if __name__ == "__main__":
    main()
