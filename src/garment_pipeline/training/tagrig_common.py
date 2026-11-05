"""[2026-10-22] Shared training glue for the tag-rig laundry dataset (datasets/tag_rig_to_lerobot.py), used by train_act.py and
train_dit_tagrig.py so ACT and Multi-Task DiT see the SAME relative-chunk targets, the SAME stats and the SAME logging.

Action target (both models): contract rows k = inv(T(t)) @ T(t+k), k = 1..32 @ 20 Hz, grippers absolute (binary + 0.1 s ramp).
  DiT : n_obs 2, horizon 33 -> rows 0..32 (row 0 = identity + g(t)), exactly training/rel_chunk.py
  ACT : n_obs 1, chunk 32   -> rows 1..32; anchor T(t) = the frame's pose_abs, actions t..t+31 = T(t+1..t+32)

Stats (computed here from the TRAIN root only; never from val/test, never per-episode):
  - state: recomputed from the train frames; action: from every relative window of the train episodes (LeRobot-style clamped padding)
  - std floor STD_FLOOR on every non-gripper dim (MEAN_STD), min-max range floor RANGE_FLOOR (MIN_MAX), symmetric around the centre
  - ACT action (MEAN_STD in LeRobot): mean/std set to (max+min)/2, (max-min)/2 -> the same [-1, 1] range map as the DiT's MIN_MAX
    (plain mean/std gave |x| up to 13.6 on the rot6d diagonal dims)
  - gripper dims (action 18/19, state 18/19) and T_LR_valid (state 29): mean 0 / std 1 and min 0 / max 1, i.e. NOT standardised:
    ACT (MEAN_STD) sees them in [0, 1]; DiT (MIN_MAX) maps [0, 1] -> [-1, 1]
  ds.meta.stats is replaced before lerobot_train builds the processors, so these are exactly what the policy uses (verified from the
  saved policy_preprocessor_*normalizer*.safetensors by training/tagrig_norm_audit.py --ckpt).

GRAD_SPLIT_EVERY=K (0 = off): every K steps an extra forward/backward logs gn_recon (L1 / diffusion MSE), gn_rest (ACT: kl_weight*KL),
gn_total and gn_recon_{pos,rot,grip} -- the source of a large pre-clip grad-norm.
Action mask (2026-10-23, unsplit episodes): per-arm / per-step loss mask from the dataset columns action_mask + pose_ok (install_masks,
chunk_mask), applied to the elementwise L1 (ACT) / MSE (DiT) and renormalised by the mask sum (masked_elementwise).  USE_ACTION_MASK=0 off.
Episode weight: the dataset column `weight` (1.0 TRAIN / 0.3 DOWNWEIGHT, datasets/tag_rig_to_lerobot.py) multiplies the per-sample
elementwise loss before the policy's own masked mean (USE_EPISODE_WEIGHT=0 disables).  Val eval is unweighted.
Metrics (METRICS_EVERY steps, default 1) -> <output_dir>/metrics.jsonl: loss, global grad-norm BEFORE clipping, clip threshold,
clipped flag, lr, per-group loss on normalised targets (pos = xyz L/R, rot = rot6d L/R, grip = gL/gR).  NaN/Inf loss or grad -> stop.
"""
import json, math, os, pathlib, sys, time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
from xvla_core import relative_chunk_from_abs  # noqa: E402
from garment_pipeline.training.rel_chunk import to_contract33  # noqa: E402

H_ACT = 32
STD_FLOOR, RANGE_FLOOR = 1e-2, 2e-2
GRIP_A = [18, 19]
STATE_UNSCALED = [18, 19, 29]
GROUPS = {"pos": [0, 1, 2, 9, 10, 11], "rot": list(range(3, 9)) + list(range(12, 18)), "grip": [18, 19]}


# ---------------------------------------------------------------- relative windows
def act_rel(pose_abs_t, actions):
    """ACT: anchor pose_abs(t) (20,) + stored next-frame actions t..t+31 (32, 20) -> contract rows k = 1..32"""
    return relative_chunk_from_abs(np.vstack([pose_abs_t[None], actions]))


def windows(A, P, epi, kind, stride=1):
    out = []
    for e in np.unique(epi):
        sel = np.flatnonzero(epi == e); a, p = A[sel], P[sel]; n = len(sel)
        for t in range(0, n, stride):
            if kind == "dit":
                out.append(to_contract33(a[np.clip(np.arange(t - 1, t - 1 + 33), 0, n - 1)]))
            else:
                out.append(act_rel(p[t], a[np.clip(np.arange(t, t + H_ACT), 0, n - 1)]))
    return np.stack(out).astype(np.float64)          # (N, H, 20)


def _stats(X, M=None):
    """per-dim stats; with M (same shape, 0/1) only the loss-active entries of each dim are used (10-23: masked targets excluded)"""
    if M is None:
        st = dict(min=X.min(0), max=X.max(0), mean=X.mean(0), std=X.std(0), count=np.array([len(X)]))
        for q in (1, 10, 50, 90, 99): st[f"q{q:02d}"] = np.percentile(X, q, axis=0)
        return st
    cols = [X[M[:, d] > 0, d] if (M[:, d] > 0).any() else X[:, d] for d in range(X.shape[1])]
    st = {k: np.array([f(c) for c in cols]) for k, f in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))}
    st["count"] = np.array([len(X)]); st["active_frac"] = (M > 0).mean(0)
    for q in (1, 10, 50, 90, 99): st[f"q{q:02d}"] = np.array([np.percentile(c, q) for c in cols])
    return st


def fixed_stats(X, unscaled, M=None):
    """raw per-dim stats (loss-active entries only when M is given) + floors; unscaled dims get mean 0 / std 1 / min 0 / max 1."""
    raw = _stats(X, M); st = {k: v.copy() for k, v in raw.items() if k != "active_frac"}; audit = []
    for d in range(X.shape[1]):
        fix = []
        if d in unscaled:
            st["mean"][d], st["std"][d], st["min"][d], st["max"][d] = 0.0, 1.0, 0.0, 1.0; fix.append("unscaled (0/1)")
        else:
            if raw["std"][d] < STD_FLOOR: st["std"][d] = STD_FLOOR; fix.append(f"std floor {STD_FLOOR}")
            if raw["max"][d] - raw["min"][d] < RANGE_FLOOR:
                c = (raw["max"][d] + raw["min"][d]) / 2; st["min"][d], st["max"][d] = c - RANGE_FLOOR / 2, c + RANGE_FLOOR / 2; fix.append(f"range floor {RANGE_FLOOR}")
        audit.append(dict(dim=d, mean=raw["mean"][d], std=raw["std"][d], min=raw["min"][d], max=raw["max"][d], p1=raw["q01"][d], p99=raw["q99"][d],
                          flag_std_lt_1e3=bool(raw["std"][d] < 1e-3), fix=", ".join(fix)))
    return {k: (v.astype(np.float32) if k != "count" else v) for k, v in st.items()}, audit


def compute_all(root, kind, stride=1):
    """train-root arrays -> dict(state_stats, action_stats, audits, windows)"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/" + pathlib.Path(root).name, root=root)
    hf = ds.hf_dataset.with_format("numpy")
    A, P, S, epi = (np.asarray(hf[k]) for k in ("action", "pose_abs", "observation.state", "episode_index"))
    W = windows(A, P, epi, kind, stride)
    MW = None
    if "action_mask" in ds.features:                                      # 10-23: stats from loss-active targets only (same chunk-mask logic)
        AM = np.asarray(hf["action_mask"]); AO = np.asarray(hf["anchor_ok"]); MW = []
        for e in np.unique(epi):
            sel = np.flatnonzero(epi == e); n = len(sel)
            for t in range(0, n, stride):
                rows = np.clip(np.arange(t, t + H_ACT) if kind == "act" else np.arange(t - 1, t + 32), 0, n - 1); m = AM[sel[rows]].copy(); ao = AO[sel[t]]
                m[:, 0:3] = np.minimum(m[:, 0:3], ao[0]) * (ao[1] > 0); m[:, 3:9] = np.minimum(m[:, 3:9], ao[1])
                m[:, 9:12] = np.minimum(m[:, 9:12], ao[2]) * (ao[3] > 0); m[:, 12:18] = np.minimum(m[:, 12:18], ao[3])
                if kind != "act": m[0] = 1.0
                MW.append(m)
        MW = np.stack(MW).reshape(-1, 20)
    ast, aaud = fixed_stats(W.reshape(-1, 20), GRIP_A, MW); sst, saud = fixed_stats(S.astype(np.float64), STATE_UNSCALED)
    if kind == "act":   # audit 10-22: MEAN_STD put 472 values > 10 on the rot6d diagonal dims (mean ~1, long tail) -> ACT gets the DiT's
        nz = [d for d in range(20) if d not in GRIP_A]     # [-1, 1] range map: mean = (max + min) / 2, std = (max - min) / 2 (grippers stay 0/1)
        ast["mean"][nz] = (ast["max"][nz] + ast["min"][nz]) / 2; ast["std"][nz] = (ast["max"][nz] - ast["min"][nz]) / 2
        for d in nz: aaud[d]["fix"] = (aaud[d]["fix"] + ", " if aaud[d]["fix"] else "") + "range map (mean=(max+min)/2, std=(max-min)/2)"
        # final-build gate 10-22: the anchor-state rot6d diagonal (Ranc_r4, ~1 with a long tail) reached |x| = 10.0 under mean/std ->
        # same range map for the ACT state (grippers and T_LR_valid stay unscaled)
        ns = [d for d in range(sst["mean"].shape[0]) if d not in STATE_UNSCALED]
        sst["mean"][ns] = (sst["max"][ns] + sst["min"][ns]) / 2; sst["std"][ns] = (sst["max"][ns] - sst["min"][ns]) / 2
        for d in ns: saud[d]["fix"] = (saud[d]["fix"] + ", " if saud[d]["fix"] else "") + "range map"
    return dict(ds=ds, A=A, P=P, S=S, epi=epi, W=W, MW=MW, action_stats=ast, state_stats=sst, action_audit=aaud, state_audit=saud)


_WEIGHT_BY_INDEX = None
_MASK = dict(kind=None, mask=None, pose_ok=None, first=None, last=None)


def masked_elementwise(out, M, pad=None, pad_masking=True, groups=None):
    """[2026-10-23] per-element loss WEIGHTS (0 = masked) with PER-GROUP normalisation: the policy's own reduction (mean over unpadded
    rows x dims, or the plain mean) becomes  sum_g (|D_g| / D) * sum(w_g * err_g) / sum(w_g)  for g in pos / rot / grip, i.e. each group
    is a weighted mean of its own supervised entries (NOT mean(w * err)); a group with zero total weight contributes 0.
    out, M: (B, H, D); pad: (B, H) bool (True = padded row)."""
    groups = groups or GROUPS; D = out.shape[-1]
    M = M.to(out.device, out.dtype)
    if pad is not None: M = M * (~pad.to(out.device)).unsqueeze(-1).to(out.dtype)
    denom = ((~pad).sum() * D).to(out.dtype) if (pad is not None and pad_masking) else torch.tensor(float(out.numel()), device=out.device, dtype=out.dtype)
    scale = torch.zeros(D, device=out.device, dtype=out.dtype)
    for ix in groups.values():
        sw = M[..., ix].sum()
        if sw > 0: scale[ix] = denom * (len(ix) / D) / sw
    return out * M * scale


def install_masks(ds, kind):
    """per-frame `action_mask` [20] (stored target valid) and `pose_ok` [2] (anchor pose valid) -> tables by global index; chunk mask built
    per batch in the update wrapper (ACT rows t..t+31, DiT rows t-1..t+31 with row 0 = 1; arm dims also need pose_ok of the anchor t)"""
    hf = ds.hf_dataset.with_format("numpy")
    if "action_mask" not in ds.features: print("[tagrig] dataset has no action_mask -> unmasked", flush=True); return ds
    idx = np.asarray(hf["index"]).astype(np.int64); ep = np.asarray(hf["episode_index"]); n = int(idx.max()) + 1
    M = np.ones((n, 20), np.float32); P = np.ones((n, 4), np.float32); F = np.zeros(n, np.int64); L = np.zeros(n, np.int64)
    M[idx] = np.asarray(hf["action_mask"]); P[idx] = np.asarray(hf["anchor_ok"])         # [pos_L, rot_L, pos_R, rot_R]
    for e in np.unique(ep):
        sel = idx[ep == e]; F[sel] = sel.min(); L[sel] = sel.max()
    _MASK.update(kind=kind, mask=torch.from_numpy(M), pose_ok=torch.from_numpy(P), first=torch.from_numpy(F), last=torch.from_numpy(L))
    print(f"[tagrig] action_mask: {M[idx].mean(0).round(3).tolist()} (mean kept per dim), anchor_ok {P[idx].mean(0).round(3).tolist()}", flush=True)
    return ds


def chunk_mask(index):
    """(B,) global frame index -> (B, H, 20) chunk mask"""
    t = index.reshape(-1).long().cpu(); m = _MASK
    if m["kind"] == "act": rows = torch.arange(0, 32)[None] + t[:, None]
    else: rows = torch.arange(-1, 32)[None] + t[:, None]
    rows = torch.maximum(torch.minimum(rows, m["last"][t][:, None]), m["first"][t][:, None])
    M = m["mask"][rows].clone()                                            # (B, H, 20) target weights
    po = m["pose_ok"][t]                                                   # anchor weights [pos_L, rot_L, pos_R, rot_R]
    # step weight = min(weight(anchor t), weight(t+k)); the relative translation R(t)^T (p(t+k) - p(t)) also needs the anchor rotation
    M[..., 0:3] = torch.minimum(M[..., 0:3], po[:, 0, None, None]) * (po[:, 1] > 0)[:, None, None]
    M[..., 3:9] = torch.minimum(M[..., 3:9], po[:, 1, None, None])
    M[..., 9:12] = torch.minimum(M[..., 9:12], po[:, 2, None, None]) * (po[:, 3] > 0)[:, None, None]
    M[..., 12:18] = torch.minimum(M[..., 12:18], po[:, 3, None, None])
    if os.environ.get("MASK_CROSS_GAP", "0") == "1":                       # 11-04: once a future step's xyz is masked (gap >= FILL_MASK_S), every later
        r0 = 0 if m["kind"] == "act" else 1                                # step of that chunk is masked too (the path across the gap is uncertain)
        for sl in (slice(0, 3), slice(9, 12)):
            M[:, r0:, sl] = torch.cummin(M[:, r0:, sl], dim=1).values
    if m["kind"] != "act": M[:, 0] = 1.0                                   # DiT row 0 = identity target
    return M


def install_weights(ds):
    """global frame index -> episode weight (dataset column `weight`; absent -> all 1)"""
    global _WEIGHT_BY_INDEX
    hf = ds.hf_dataset.with_format("numpy"); idx = np.asarray(hf["index"]).astype(np.int64)
    w = np.asarray(hf["weight"]).reshape(-1).astype(np.float32) if "weight" in ds.features else np.ones(len(idx), np.float32)
    tab = np.ones(int(idx.max()) + 1, np.float32); tab[idx] = w; _WEIGHT_BY_INDEX = torch.from_numpy(tab)
    print(f"[tagrig] episode weights: {np.unique(w, return_counts=True)}", flush=True)
    return ds


def install_stats(ds, kind, stride=1):
    root = pathlib.Path(ds.root); cache = root / "meta" / f"tagrig_fixed_stats_{kind}_v4.json"
    if cache.exists():
        c = json.loads(cache.read_text()); conv = lambda d: {k: np.asarray(v, np.float32 if k != "count" else np.int64) for k, v in d.items()}
        ast, sst = conv(c["action"]), conv(c["state"])
    else:
        r = compute_all(root, kind, stride); ast, sst = r["action_stats"], r["state_stats"]
        cache.write_text(json.dumps({"action": {k: v.tolist() for k, v in ast.items()}, "state": {k: v.tolist() for k, v in sst.items()},
                                     "note": "train root only; floors + unscaled grippers, see training/tagrig_common.py"}))
    for key, st in (("action", ast), ("observation.state", sst)):
        ds.meta.stats[key] = {**ds.meta.stats.get(key, {}), **st}
    return ds


from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


class ActRelDataset(LeRobotDataset):
    """module-level (picklable under spawn workers): action (32, 20) abs -> contract rows k = 1..32 anchored at pose_abs(t)"""
    def __getitem__(self, idx):
        item = super().__getitem__(idx); a = item["action"]
        if a.ndim == 2 and a.shape[0] == H_ACT:
            item["action"] = torch.from_numpy(act_rel(item["pose_abs"].numpy().astype(np.float64), a.numpy().astype(np.float64)).astype(np.float32))
        return item


GRIP_STATE_MASK = __import__("os").environ.get("GRIP_STATE_MASK", "0") == "1"
GRIP_STATE_DIMS = (18, 19)          # proprio30 [L pose9 | R pose9 | gL gR | T_LR pose9 | valid]
PGRASP = __import__("os").environ.get("PGRASP", "")   # 11-05 FLAT250-PGRASP: jaw slots <- <root>/meta/pgrasp_<PGRASP>.npy (twin of laundry_training.datasets)


# 11-01 user "이미지 증강 넣어보자": TRAIN-time image augmentation (AUG=1), applied per sample in the dataset (CPU workers, images in [0, 1],
# before the policy normaliser; the evaluator / deploy never see it).  Sized to the measured robot-vs-UMI gaps: the Orbbec moved 5-13 px /
# 0.5-1 deg between days -> global: shift +-AUG_G_SHIFT of the size, rot +-AUG_G_ROT deg; robot wrist cams mount 11-19 deg / 25-44 mm off the
# UMI -> wrists: shift +-AUG_W_SHIFT, rot +-AUG_W_ROT, zoom +-AUG_W_ZOOM; every camera: brightness / contrast +-AUG_BC, per-channel gain +-AUG_CH.
# Labels are untouched (the image changes, the action does not).
import os as _os
AUG = _os.environ.get("AUG", "0") == "1"
AUG_P = {k: float(_os.environ.get(k, v)) for k, v in dict(AUG_G_SHIFT=0.045, AUG_G_ROT=2.0, AUG_G_ZOOM=0.02, AUG_W_SHIFT=0.08, AUG_W_ROT=5.0, AUG_W_ZOOM=0.05,
                                                            AUG_BC=0.15, AUG_CH=0.05, AUG_SAT=0.0, AUG_HUE=0.0).items()}
# 11-02 user (UMI appendix "같게"): UMI ColorJitter = brightness 0.3 / contrast 0.4 / saturation 0.5 / hue 0.08 -> AUG_B / AUG_C (default AUG_BC),
# AUG_SAT, AUG_HUE (default 0 = the 11-01 behaviour); UMI RandomCrop 0.95 ~ AUG_*_ZOOM 0.05 + shift 0.025, RandomRotation +-5 deg ~ AUG_*_ROT 5.
AUG_P["AUG_B"] = float(_os.environ.get("AUG_B", AUG_P["AUG_BC"])); AUG_P["AUG_C"] = float(_os.environ.get("AUG_C", AUG_P["AUG_BC"]))


# 11-02: TF.adjust_hue (RGB->HSV->RGB) cost 62 ms per camera sample on the CPU workers -> the 4xH100 run was data-bound at ~3 s/update.
# Hue shift as a rotation of the chroma plane in YIQ (one 3x3 matrix, ~1 ms); hue_factor in turns like adjust_hue (0.08 = +-28.8 deg).
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], dtype=torch.float32)
_YIQ_INV = torch.linalg.inv(_YIQ)


def hue_rotate(y, hue_factor):
    """y (N, 3, H, W) float in [0, 1]"""
    a = 2.0 * math.pi * hue_factor; R = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    M = (_YIQ_INV @ R @ _YIQ).to(y.dtype)
    return torch.einsum("ij,njhw->nihw", M, y).clamp(0.0, 1.0)


def aug_image(x, shift, rot_deg, zoom):
    """x (..., C, H, W) float in [0, 1]: one random similarity (shift, rotation, zoom; border padding) + brightness / contrast / channel
    gain, the SAME for all frames of this sample (n_obs_steps)"""
    # 11-01: the 4090 video backend yields uint8 0..255 (the Mac pyav path float 0..1) -> work in float, return the input dtype/range
    dt = x.dtype; isint = not torch.is_floating_point(x); x = x.float() / 255.0 if isint else x
    shp = x.shape; y = x.reshape(-1, *shp[-3:]); u = lambda r: (torch.rand(()) * 2 - 1) * r
    th = math.radians(float(u(rot_deg))); z = 1.0 + float(u(zoom)); tx, ty = float(u(2 * shift)), float(u(2 * shift))   # affine_grid units: [-1, 1]
    A = torch.tensor([[math.cos(th) / z, -math.sin(th) / z, tx], [math.sin(th) / z, math.cos(th) / z, ty]], dtype=y.dtype).expand(y.shape[0], 2, 3)
    g = torch.nn.functional.affine_grid(A, list(y.shape), align_corners=False)
    y = torch.nn.functional.grid_sample(y, g, mode="bilinear", padding_mode="border", align_corners=False)
    b, c = 1.0 + float(u(AUG_P["AUG_B"])), 1.0 + float(u(AUG_P["AUG_C"])); ch = 1.0 + (torch.rand(3, dtype=y.dtype) * 2 - 1) * AUG_P["AUG_CH"]
    m = y.mean(dim=(-3, -2, -1), keepdim=True); y = ((y - m) * c + m) * b * ch.view(1, 3, 1, 1)
    if AUG_P["AUG_SAT"] > 0 or AUG_P["AUG_HUE"] > 0:
        import torchvision.transforms.functional as TF
        y = y.clamp(0.0, 1.0)
        if AUG_P["AUG_SAT"] > 0: y = TF.adjust_saturation(y, 1.0 + float(u(AUG_P["AUG_SAT"])))
        if AUG_P["AUG_HUE"] > 0: y = hue_rotate(y, float(u(AUG_P["AUG_HUE"])))
    y = y.clamp(0.0, 1.0).reshape(shp)
    return (y * 255.0).round().to(dt) if isint else y.to(dt)


class DitRelDataset(LeRobotDataset):
    """module-level twin of rel_chunk.patch_dataset's class: window (33, 20) -> rows 0..32 (row 0 identity).
    [10-29] GRIP_STATE_MASK=1: the jaw state in observation.state is fixed to 0 (both obs steps).  Exp 1/2 learned to COPY the input jaw
    state (deploy: closed in -> closed out forever, the jaw never opened); with the state hidden the jaw timing must come from vision."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx); a = item["action"]
        if a.ndim == 2 and a.shape[0] == 33:
            item["action"] = torch.from_numpy(to_contract33(a.numpy()))
        if GRIP_STATE_MASK and "observation.state" in item:
            st = item["observation.state"].clone(); st[..., list(GRIP_STATE_DIMS)] = 0.0
            if PGRASP:
                if getattr(self, "_pg", None) is None: self._pg = np.load(self.root / "meta" / f"pgrasp_{PGRASP}.npy", mmap_mode="r")
                i_, f_ = int(item["index"]), int(item["frame_index"]); rows = [i_ - 1 if f_ > 0 else i_, i_][-st.shape[0]:] if st.dim() == 2 else [i_]
                st[..., list(GRIP_STATE_DIMS)] = torch.from_numpy(np.asarray(self._pg[rows], np.float32)).reshape(st.shape[:-1] + (2,))
            item["observation.state"] = st
        if AUG:
            for k in [k for k in item if k.startswith("observation.images.") and not k.endswith("_is_pad") and torch.is_tensor(item[k]) and item[k].dim() >= 3]:   # 11-01: skip the <cam>_is_pad masks
                g_ = k.endswith(".global"); P_ = AUG_P
                item[k] = aug_image(item[k], P_["AUG_G_SHIFT"] if g_ else P_["AUG_W_SHIFT"], P_["AUG_G_ROT"] if g_ else P_["AUG_W_ROT"], P_["AUG_G_ZOOM"] if g_ else P_["AUG_W_ZOOM"])
        return item


GRIP_BCE = __import__("os").environ.get("GRIP_BCE", "0") == "1"
GRIP_BCE_W = float(__import__("os").environ.get("GRIP_BCE_W", "1.0"))
# 10-30 user "그리퍼만 weight를 더 주게": WEIGHT_GRIP_ONLY=1 -> the per-sample dataset `weight` column scales ONLY the BCE jaw loss
# (the motor-jaw flattening demos are built with EP_WEIGHT=3); the diffusion MSE (pose) stays unweighted.
WEIGHT_GRIP_ONLY = __import__("os").environ.get("WEIGHT_GRIP_ONLY", "0") == "1"
# 10-30 user plan A (gripper relabel): GRIP_HEAD_ONLY=1 -> everything except grip_head frozen, observation encoder run under no_grad,
# diffusion loss not computed (loss = BCE only).  GRIP_HEAD_RESET=1 -> grip_head re-initialised at the first forward (after the
# checkpoint weights are loaded), so old-head vs new-head comparisons start from the same arm weights and a fresh head.
GRIP_HEAD_ONLY = __import__("os").environ.get("GRIP_HEAD_ONLY", "0") == "1"
GRIP_HEAD_RESET = __import__("os").environ.get("GRIP_HEAD_RESET", "0") == "1"


def install_grip_bce(MD):
    """[2026-10-29] UMI-style recipe (user plan "gripper loss: BCE 별도 head/loss"): Multi-Task DiT gets `grip_head`, an MLP on the
    observation conditioning vector -> (horizon, 2) jaw logits, trained with BCEWithLogits against the jaw rows of the contract chunk
    (action dims 18/19, MIN_MAX-normalised [-1, 1] -> target (x + 1) / 2 in [0, 1]), masked like the jaw dims (chunk mask + padding).
    The diffusion MSE no longer sees the jaw dims (update() zeroes their mask); at inference the denoised jaw rows are REPLACED by
    2 sigmoid(logit) - 1 (deploy core.PolicyRunner / evaluation tagrig_offline_eval).  Always installed when GRIP_BCE=1 at training;
    loading checkpoints: core / eval install it when the safetensors contain grip_head.* (strict=False loading otherwise drops it)."""
    import torch.nn as nn
    P = MD.MultiTaskDiTPolicy
    if getattr(P, "_grip_bce", False): return
    init0 = P.__init__

    def __init__(self, config, **kw):
        init0(self, config, **kw)
        d = self.observation_encoder.conditioning_dim
        self.grip_head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, config.horizon * 2))

    def grip_logits(self, cond):
        return self.grip_head(cond).view(cond.shape[0], self.config.horizon, 2)

    def forward(self, batch):
        if GRIP_HEAD_ONLY and not getattr(self, "_head_only_set", False):
            for n_, p_ in self.named_parameters(): p_.requires_grad_(n_.startswith("grip_head."))
            if GRIP_HEAD_RESET:
                for m_ in self.grip_head:
                    if hasattr(m_, "reset_parameters"): m_.reset_parameters()
            self._head_only_set = True
            print(f"[tagrig] GRIP_HEAD_ONLY: trainable {sum(p_.numel() for p_ in self.parameters() if p_.requires_grad)} params (grip_head), reset {GRIP_HEAD_RESET}", flush=True)
        batch = self._prepare_batch(batch)
        if GRIP_HEAD_ONLY:
            with torch.no_grad(): cond = self.observation_encoder.encode(batch)
            loss = None
        else:
            cond = self.observation_encoder.encode(batch); loss = self.objective.compute_loss(self.noise_predictor, batch, cond)
        lg = grip_logits(self, cond).float(); tgt = ((batch["action"][..., GRIP_A].float() + 1.0) / 2.0).clamp(0.0, 1.0)
        m = _LAST.get("Mg"); m = torch.ones_like(tgt) if m is None or m.shape != tgt.shape else m.to(tgt.device, tgt.dtype)
        pad = batch.get("action_is_pad")
        if pad is not None: m = m * (~pad.to(tgt.device)).unsqueeze(-1).to(tgt.dtype)
        w_ = _LAST.get("w")
        if WEIGHT_GRIP_ONLY and w_ is not None and w_.shape[0] == m.shape[0]:
            m_w = m * w_.to(m.device, m.dtype).reshape(-1, *([1] * (m.dim() - 1)))
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * m_w).sum() / m.sum().clamp_min(1.0)
        else:
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * m).sum() / m.sum().clamp_min(1.0)
        _LAST.setdefault("groups", {})["grip_bce"] = float(bce.detach())
        acc = (((lg > 0).float() == (tgt > 0.5).float()).float() * m).sum() / m.sum().clamp_min(1.0); _LAST["groups"]["grip_acc"] = float(acc)
        if loss is None: return GRIP_BCE_W * bce, None
        return loss + GRIP_BCE_W * bce.to(loss.dtype), None

    P.__init__ = __init__; P.forward = forward; P.grip_logits = grip_logits; P._grip_bce = True
    print(f"[tagrig] GRIP_BCE: grip_head (cond -> horizon x 2 logits), BCE weight {GRIP_BCE_W}, jaw dims removed from the diffusion MSE", flush=True)


def patch_act_dataset(ds):
    assert type(ds) is LeRobotDataset, type(ds); ds.__class__ = ActRelDataset; return ds


def patch_dit_dataset(ds):
    assert type(ds) is LeRobotDataset, type(ds); ds.__class__ = DitRelDataset; return ds


# ---------------------------------------------------------------- metrics logging
_LAST = {}


class _FProxy:
    """stands in for `F` inside a policy module: records per-dim elementwise losses with a trailing dim of 20"""
    def __init__(self, names): self._names = names

    def __getattr__(self, k):
        f = getattr(torch.nn.functional, k)
        if k not in self._names: return f

        def wrapped(*a, **kw):
            out = f(*a, **kw)
            if kw.get("reduction") == "none" and out.shape[-1] == 20:
                M = _LAST.get("M")
                if M is not None and M.shape == out.shape:
                    out = masked_elementwise(out, M, _LAST.get("pad"), _LAST.get("pad_masking", True))
                w = _LAST.get("w")                     # EPISODE weight (dataset column `weight`, 1.0 / 0.3) -> per-sample loss weight
                if w is not None and w.shape[0] == out.shape[0] and not WEIGHT_GRIP_ONLY:
                    out = out * w.to(out.device, out.dtype).reshape(-1, *([1] * (out.dim() - 1)))
                _LAST["err_t"] = out
                per = out.detach().float().reshape(-1, 20).mean(0)
                _LAST["groups"] = {g: float(per[ix].mean()) for g, ix in GROUPS.items()}
            return out
        return wrapped


GRAD_ACCUM = int(__import__("os").environ.get("GRAD_ACCUM", "1"))


def accum_update(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator=None, lr_scheduler=None, lock=None, sample_weighter=None):
    """[2026-10-29] effective batch = --batch_size, computed as GRAD_ACCUM micro-batches (4090: a physical batch of 64 is OOM).  Same body
    as lerobot_train.update_policy (3c15fd8) -- clip, ONE optimizer + scheduler step per call, so the step count = optimizer updates --
    with the per-sample loss masks / episode weights / padding of _LAST sliced to each micro-batch.  The group-normalised masked loss is
    normalised within each micro-batch (mean of the micro-batch losses)."""
    import contextlib
    t0 = time.time(); policy.train(); B = int(batch["index"].shape[0]); n = min(GRAD_ACCUM, B); edges = np.linspace(0, B, n + 1).astype(int)
    full = {k: _LAST.get(k) for k in ("M", "Mg", "w", "pad")}; tot = 0.0; od = None
    for j in range(n):
        sl = slice(int(edges[j]), int(edges[j + 1]))
        for k, v in full.items(): _LAST[k] = v[sl] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v
        mb = {k: (v[sl] if (torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B) or (isinstance(v, list) and len(v) == B) else v) for k, v in batch.items()}
        with accelerator.autocast():
            loss, od = policy.forward(mb)
        accelerator.backward(loss / n); tot += float(loss.detach()) / n
    _LAST.update(full)
    gn = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm) if grad_clip_norm > 0 else torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
    with (lock if lock is not None else contextlib.nullcontext()):
        optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None: lr_scheduler.step()
    up = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if hasattr(up, "update") and callable(up.update): up.update()
    train_metrics.loss = tot; train_metrics.grad_norm = float(gn); train_metrics.lr = optimizer.param_groups[0]["lr"]; train_metrics.update_s = time.time() - t0
    return train_metrics, od


def install_metrics(LT, policy_module, loss_fn_name, every=None):
    import os
    import accelerate
    policy_module.F = _FProxy({loss_fn_name})
    every = int(every or os.environ.get("METRICS_EVERY", 1))
    orig_clip = accelerate.Accelerator.clip_grad_norm_

    def clip(self, parameters, max_norm, *a, **kw):
        n = orig_clip(self, parameters, max_norm, *a, **kw); _LAST["grad_norm"] = float(n); _LAST["max_norm"] = float(max_norm); return n
    accelerate.Accelerator.clip_grad_norm_ = clip
    orig_update = LT.update_policy; state = dict(step=0, fh=None, n_clip=0)

    split_every = int(os.environ.get("GRAD_SPLIT_EVERY", 0)); rank = int(os.environ.get("RANK", 0))
    if rank != 0: split_every = 0                     # DDP: grad split + metrics on rank 0 only (grad-norm is post all-reduce on every rank)

    def grad_split(policy, batch):
        """separate gradient norms of the reconstruction term (per action group) and of the rest (ACT: kl_weight * KL)"""
        policy = getattr(policy, "module", policy)       # unwrapped: autograd.grad never fires the DDP reducer hooks
        params = [p for p in policy.parameters() if p.requires_grad]
        gn = lambda t: float(torch.sqrt(sum((g.float() ** 2).sum() for g in torch.autograd.grad(t, params, retain_graph=True, allow_unused=True) if g is not None)))
        loss, _ = policy.forward(batch); err = _LAST["err_t"]
        m = (~batch["action_is_pad"]).unsqueeze(-1).to(err.dtype) if "action_is_pad" in batch else torch.ones_like(err[..., :1])
        rec = lambda ix: (err[..., ix] * m).sum() / (m.sum() * len(ix)).clamp_min(1)
        r = rec(list(range(20))); out = dict(gn_recon=gn(r), gn_rest=gn(loss - r) if float((loss - r).abs()) > 1e-9 else 0.0, gn_total=gn(loss))
        for g, ix in GROUPS.items(): out[f"gn_recon_{g}"] = gn(rec(ix))
        policy.zero_grad(set_to_none=True); return out

    def update(train_metrics, policy, batch, optimizer, grad_clip_norm, *a, **kw):
        ix = batch.get("index") if isinstance(batch, dict) else None      # the preprocessor drops `weight`; `index` survives -> table lookup
        _LAST["M"] = chunk_mask(ix) if (_MASK["mask"] is not None and torch.is_tensor(ix) and os.environ.get("USE_ACTION_MASK", "1") == "1") else None
        _LAST["pad"] = batch.get("action_is_pad") if isinstance(batch, dict) else None
        if GRIP_BCE and _LAST["M"] is not None:            # 10-29: jaws trained by the BCE head only (install_grip_bce), not by the diffusion MSE
            _LAST["Mg"] = _LAST["M"][..., GRIP_A].clone(); M_ = _LAST["M"].clone(); M_[..., GRIP_A] = 0.0; _LAST["M"] = M_
        pc = getattr(getattr(policy, "module", policy), "config", None)
        _LAST["pad_masking"] = True if _MASK["kind"] == "act" else bool(getattr(pc, "do_mask_loss_for_padding", False))
        _LAST["w"] = (_WEIGHT_BY_INDEX[ix.reshape(-1).long().cpu()] if torch.is_tensor(ix) and _WEIGHT_BY_INDEX is not None
                      and os.environ.get("USE_EPISODE_WEIGHT", "1") == "1" else None)
        if state["step"] == 0 and rank == 0: print(f"[tagrig] episode weight in batch: {None if _LAST['w'] is None else _LAST['w'][:8].tolist()} keys {sorted(batch)}", flush=True)
        gs = grad_split(policy, batch) if split_every and (state["step"] == 0 or (state["step"] + 1) % split_every == 0) else {}
        if state.get("init") is None:                       # resume: continue the step count of an existing metrics.jsonl
            state["init"] = True; mf = pathlib.Path(LT._TAGRIG_OUT) / "metrics.jsonl"
            if mf.exists() and os.environ.get("RESUME", "0") == "1":
                try: state["step"] = json.loads(mf.read_text().strip().splitlines()[-1])["step"]
                except Exception: pass
        _LAST["grad_norm"] = None
        tm, od = (accum_update if GRAD_ACCUM > 1 else orig_update)(train_metrics, policy, batch, optimizer, grad_clip_norm, *a, **kw)
        if _LAST.get("grad_norm") is None:              # 10-29 GRAD_CLIP=0: accelerate's clip (our hook) is not called -> lerobot's own norm
            gm = getattr(tm, "grad_norm", None); _LAST["grad_norm"] = float(getattr(gm, "val", gm)) if gm is not None else float("nan")
        # 10-29 RunPod budget (user: "잔액에 맞게 체크포인트"): at DEADLINE_TS (unix s) one extra full checkpoint (weights + training_state) is
        # forced by setting the run's save_freq to 1 for exactly one step, then restored; training continues (a top-up keeps it going).
        cfg_ = getattr(LT, "_TAGRIG_CFG", None); dl = float(os.environ.get("DEADLINE_TS", "0") or 0)
        if cfg_ is not None and dl > 0:
            if state.get("dl") is None and time.time() >= dl:
                state["dl"] = "armed"; state["sf"] = cfg_.save_freq; cfg_.save_freq = 1
                if rank == 0: print(f"[tagrig] DEADLINE reached at update {state['step']} -> extra checkpoint now", flush=True)
            elif state.get("dl") == "armed":
                cfg_.save_freq = state["sf"]; state["dl"] = "done"
        state["step"] += 1; g = _LAST.get("grad_norm", float("nan")); clipped = bool(g > _LAST.get("max_norm", math.inf)); state["n_clip"] += clipped
        loss = float(tm.loss.val) if hasattr(tm.loss, "val") else float("nan")
        if rank == 0 and (state["step"] % every == 0 or gs) or not (math.isfinite(g) and math.isfinite(loss)):
            if state["fh"] is None:
                out = pathlib.Path(LT._TAGRIG_OUT); out.mkdir(parents=True, exist_ok=True); state["fh"] = open(out / "metrics.jsonl", "a")
            rec = dict(step=state["step"], t=time.time(), loss=loss, grad_norm=g, clip=_LAST.get("max_norm"), clipped=clipped,
                       clip_frac_so_far=state["n_clip"] / state["step"], lr=optimizer.param_groups[0]["lr"], **{f"loss_{k}": v for k, v in _LAST.get("groups", {}).items()}, **gs,
                       **({k: v for k, v in (od or {}).items() if isinstance(v, (int, float))}))
            state["fh"].write(json.dumps(rec) + "\n"); state["fh"].flush()
        if not (math.isfinite(g) and math.isfinite(loss)):
            raise SystemExit(f"NaN/Inf at step {state['step']}: loss {loss} grad_norm {g} -> stopped")
        return tm, od
    LT.update_policy = update
