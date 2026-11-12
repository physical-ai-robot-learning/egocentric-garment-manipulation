"""[2026-11-11 user] X-VLA UMI-flatten offline evaluator (the daily "did it get better" gate of the continual schedule).

Per evaluated row (every --stride-th row of the split, fixed seed per row) the policy predicts the (32, 32) REL32 chunk; it is
compared with the stored label chunk in the X-VLA dataset frame, per arm, only where aux.loss_mask is 1 for that group:
  pos_mm@k   |p_pred - p_gt| at horizon k = 1, 8, 16, 32 (k = 1..32 is the chunk index; rows are 0.05 s of the 0.3x timeline)
  rot_deg@k  geodesic angle between the predicted and label rot6d (rows convention of laundry_shared.contracts.action_contract)
  grip_mae   mean |g_pred - g_gt| over k = 1..32 (continuous, 0 closed .. 1 open)
plus the same numbers for the "hold" baseline (identity relative pose, gripper = label at k=1), so a model that learned nothing
scores ~baseline.  Output: one json (--out) {ckpt, split, rows, metrics{arm: {...}}, baseline{...}, score}; score = mean over
arms of pos_mm@16 (lower is better; the promotion gate compares it day to day).
    python eval_offline.py --ckpt <.../checkpoints/NNNNNN> --root <xvla split root> [--stride 15] [--out eval.json]"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

K_EVAL = (1, 8, 16, 32)
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}
ARMS = {"left": 0, "right": 10}


def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def angle_deg(Ra, Rb):
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0          # trace(Ra Rb^T)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def chunk_errors(pred, gt, mask):
    """pred / gt (32, 32) one row; mask (20,) -> {arm: dict of per-k errors or None}"""
    out = {}
    for a, o in ARMS.items():
        d = {}
        if mask[o] > 0.5:
            e = np.linalg.norm(pred[:, o:o + 3] - gt[:, o:o + 3], axis=1) * 1000
            d.update({f"pos_mm@{k}": float(e[k - 1]) for k in K_EVAL})
        if mask[o + 3] > 0.5:
            e = angle_deg(rot6d_to_matrix(pred[:, o + 3:o + 9]), rot6d_to_matrix(gt[:, o + 3:o + 9]))
            d.update({f"rot_deg@{k}": float(e[k - 1]) for k in K_EVAL})
        if mask[o + 9] > 0.5:
            d["grip_mae"] = float(np.mean(np.abs(pred[:, o + 9] - gt[:, o + 9])))
        out[a] = d
    return out


def summarize(rows):
    tOut = {}
    for a in ARMS:
        tKeys = sorted({k for r in rows for k in r[a]})
        tOut[a] = {k: dict(mean=float(np.mean(v)), p50=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v))
                   for k in tKeys for v in [[r[a][k] for r in rows if k in r[a]]]}
    return tOut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--root", required=True); ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    tCk = pathlib.Path(a.ckpt); tPm = tCk / "pretrained_model" if (tCk / "pretrained_model").exists() else tCk
    policy = XVLAPolicy.from_pretrained(str(tPm)).to(a.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(tPm),
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
    ds = LeRobotDataset(repo_id="local/" + pathlib.Path(a.root).name, root=a.root, video_backend="pyav", tolerance_s=0.001)
    tIdx = list(range(0, len(ds), a.stride))
    if a.max_rows: tIdx = tIdx[:a.max_rows]
    rows, base = [], []
    t0 = time.time()
    for n, i in enumerate(tIdx):
        it = ds[i]
        gt = it["action"].numpy().astype(np.float64); mask = it["aux.loss_mask"].numpy()
        b = {RENAME.get(k, k): (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in it.items() if k not in ("action", "aux.loss_mask")}
        b = pre(b)
        torch.manual_seed(i)
        with torch.no_grad():
            p = policy.predict_action_chunk(b)
        p = post(p)
        pred = (p if torch.is_tensor(p) else p["action"]).detach().float().cpu().numpy().reshape(-1, 32, 32)[0].astype(np.float64)
        rows.append(chunk_errors(pred, gt, mask))
        hold = np.zeros_like(gt); hold[:, 3:9] = [1, 0, 0, 0, 1, 0]; hold[:, 13:19] = [1, 0, 0, 0, 1, 0]
        hold[:, 9], hold[:, 19] = gt[0, 9], gt[0, 19]
        base.append(chunk_errors(hold, gt, mask))
        if n % 50 == 0:
            print(f"[eval] {n}/{len(tIdx)} rows, {(time.time() - t0) / max(n, 1):.2f} s/row", flush=True)
    M, B = summarize(rows), summarize(base)
    score = float(np.mean([M[x]["pos_mm@16"]["mean"] for x in ARMS if "pos_mm@16" in M[x]]))
    tRes = dict(ckpt=str(tCk), split=a.root, stride=a.stride, rows=len(tIdx), score_pos_mm_at16=score,
                baseline_score=float(np.mean([B[x]["pos_mm@16"]["mean"] for x in ARMS if "pos_mm@16" in B[x]])), metrics=M, baseline=B)
    for x in ARMS:
        m = M[x]; bb = B[x]
        print(f"[eval] {x:5s} pos@1/8/16/32 " + " ".join(f"{m.get(f'pos_mm@{k}', {}).get('mean', np.nan):6.1f}" for k in K_EVAL)
              + " mm | rot@16 " + f"{m.get('rot_deg@16', {}).get('mean', np.nan):5.1f} deg | grip {m.get('grip_mae', {}).get('mean', np.nan):.3f}"
              + f" || hold pos@16 {bb.get('pos_mm@16', {}).get('mean', np.nan):6.1f} mm", flush=True)
    print(f"[eval] score (pos_mm@16, lower better) {score:.2f}  hold baseline {tRes['baseline_score']:.2f}", flush=True)
    if a.out: pathlib.Path(a.out).write_text(json.dumps(tRes, indent=1))


if __name__ == "__main__":
    main()
