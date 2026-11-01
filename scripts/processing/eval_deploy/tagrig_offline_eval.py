#!/usr/bin/env python3
"""[2026-10-22] Offline evaluation of an ACT or Multi-Task DiT checkpoint on the tag-rig val/test roots (no robot).
  deviation   predicted contract chunk (denormalised by the checkpoint's own post-processor) vs the robotized-trajectory chunk,
              translation mm / rotation deg at k = 1, 8, 16, 32 per arm (= base-frame deviation of T(t) @ dT from the robotized
              target, the anchor is shared); baselines: zero motion and the train mean chunk
  L/R swap    cosine of the predicted k=32 base-frame displacement of each arm with the GT of the SAME arm vs the OTHER arm
  gripper     per episode, the k=1 gripper prediction at every --stride frame, thresholded 0.5 (deploy rule) vs the binary target;
              open/close events matched within +-1 s -> timing error ms, missed / spurious events
usage: tagrig_offline_eval.py --ckpt checkpoints/RUN/checkpoints/last --data data/lerobot/tagrig_laundry_v1_val [--stride 2] --out reports/RUN_val"""
import argparse, csv, json, pathlib, sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training")); sys.path.insert(0, str(HERE.parent / "datasets")); sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
from tagrig_common import ActRelDataset, DitRelDataset  # noqa: E402
from contracts.action_contract import pose9_to_T  # noqa: E402

KS = (1, 8, 16, 32)


def load(ckpt, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    pm = pathlib.Path(ckpt); pm = pm / "pretrained_model" if (pm / "pretrained_model").is_dir() else pm
    cfg = PreTrainedConfig.from_pretrained(str(pm)); cfg.device = device
    if cfg.type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    else:
        import lerobot.policies.multi_task_dit.modeling_multi_task_dit as MD
        from safetensors import safe_open
        with safe_open(str(pm / "model.safetensors"), "pt") as f_:
            if any(k.startswith("grip_head.") for k in f_.keys()):        # 10-29 GRIP_BCE checkpoint: build the jaw head before loading
                import sys as _s; _s.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training")); from tagrig_common import install_grip_bce; install_grip_bce(MD)
        P = MD.MultiTaskDiTPolicy
    pol = P.from_pretrained(str(pm), config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(pm), preprocessor_overrides={"device_processor": {"device": device}})
    if cfg.type != "act":
        from diffusers import DDIMScheduler
        pol.objective.noise_scheduler = DDIMScheduler.from_config(pol.objective.noise_scheduler.config); pol.objective.num_inference_steps = 10
    return cfg, pol, pre, post


@torch.no_grad()
def predict(cfg, pol, pre, post, item, seed):
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items() if k.startswith("observation") or k == "task"}
    batch = pre(batch); torch.manual_seed(seed)
    if cfg.type == "act":
        acts = pol.predict_action_chunk(batch)                                            # (1, 32, 20) normalised
        return post(acts)[0].float().cpu().numpy()
    b = pol._prepare_batch(dict(batch)); cond = pol.observation_encoder.encode(b)
    acts = pol.objective.conditional_sample(pol.noise_predictor, 1, cond)                 # (1, 33, 20)
    if hasattr(pol, "grip_head"): acts[..., 18:20] = 2.0 * torch.sigmoid(pol.grip_logits(cond)).to(acts.dtype) - 1.0   # BCE jaw head
    return post(acts)[0].float().cpu().numpy()[1:]


def err(P, G, sl):
    TP, TG = pose9_to_T(P[:, sl].astype(np.float64)), pose9_to_T(G[:, sl].astype(np.float64))
    return (np.linalg.norm(TP[:, :3, 3] - TG[:, :3, 3], axis=1) * 1e3,
            np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(TP[:, :3, :3], 1, 2) @ TG[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))))


@torch.no_grad()
def val_loss(cfg, pol, pre, ds, nb, bs=16):
    """mean training loss of the policy on nb fixed val batches (same targets / normalisation as training; DiT noise fixed by the seed)"""
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(0); dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=0, drop_last=True)
    torch.manual_seed(0); out = []
    for i, b in enumerate(dl):
        if i >= nb: break
        b = pre(b)
        if cfg.type == "act":                 # eval-mode ACT has no VAE posterior (forward() would fail on the KL): L1 of the prior-latent chunk
            m = (~b["action_is_pad"]).unsqueeze(-1).float(); out.append(float(((pol.predict_action_chunk(b) - b["action"]).abs() * m).sum() / (m.sum() * 20)))
        else:
            loss, _ = pol.forward(b); out.append(float(loss))
    return round(float(np.mean(out)), 5) if out else None


def events(b):
    d = np.diff(b.astype(int)); return [(i + 1, "open" if d[i] > 0 else "close") for i in np.flatnonzero(d)]


def match_events(pe, ge, fps, tol_s=1.0):
    used, errs, missed = set(), [], 0
    for gi, gk in ge:
        c = [(abs(pi - gi), j) for j, (pi, pk) in enumerate(pe) if pk == gk and j not in used and abs(pi - gi) <= tol_s * fps]
        if c: d, j = min(c); used.add(j); errs.append((pe[j][0] - gi) / fps * 1e3)
        else: missed += 1
    return errs, missed, len(pe) - len(used)


def gh_report(GH):
    out = {}
    for k, v in GH.items():
        if not v: continue
        a_ = np.array(v, bool); p_, g_, g1 = a_[:, 0], a_[:, 1], a_[:, 2]; sw = g_ != g1
        out[f"k{k}"] = dict(acc=round(float((p_ == g_).mean()), 3), copy_baseline_acc=round(float((g1 == g_).mean()), 3),
                            switch_n=int(sw.sum()), switch_acc=round(float((p_[sw] == g_[sw]).mean()), 3) if sw.any() else None,
                            pred_open_frac=round(float(p_.mean()), 3), gt_open_frac=round(float(g_.mean()), 3))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--val-loss-batches", type=int, default=20, help="policy training loss on N fixed val batches (eval mode, seed 0)")
    ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--video-backend", default=None); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg, pol, pre, post = load(a.ckpt, a.device); fps = 20
    root = pathlib.Path(a.data); kw = dict(video_backend=a.video_backend) if a.video_backend else {}
    if cfg.type == "act":
        dts = {"action": [i / fps for i in range(32)]}; ds = ActRelDataset("local/" + root.name, root=root, delta_timestamps=dts, **kw)
    else:
        dts = {k: [i / fps for i in range(1 - cfg.n_obs_steps, 1)] for k in list(cfg.image_features) + ["observation.state"]}
        dts["action"] = [i / fps for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)]; ds = DitRelDataset("local/" + root.name, root=root, delta_timestamps=dts, **kw)
    vl = val_loss(cfg, pol, pre, ds, a.val_loss_batches) if a.val_loss_batches else None
    hf = ds.hf_dataset.with_format("numpy"); epi = np.asarray(hf["episode_index"]); P = np.asarray(hf["pose_abs"])
    feats = set(ds.features)
    PC = np.asarray(hf["pos_conf"]) if "pos_conf" in feats else None          # 10-23: confident = tag or fgo gap <= 0.5 s
    GW = np.asarray(hf["grasp_win"]) if "grasp_win" in feats else None        # +-0.5 s around jaw transitions
    CF = {f"{s}_{sub}_{m}": [] for s in "LR" for sub in ("all", "grasp", "free") for m in ("mm", "zero")}
    tr = json.loads((pathlib.Path(a.ckpt) / "pretrained_model" / "train_config.json").read_text()) if (pathlib.Path(a.ckpt) / "pretrained_model").is_dir() else {}
    E = {f"{s}_{m}_k{k}": [] for s in "LR" for m in ("mm", "deg", "zero_mm") for k in KS}; cos_same, cos_other = [], []; ev_err, ev_miss, ev_spur, ev_n = [], 0, 0, 0
    g_agree = []; n = 0
    GH = {k: [] for k in (8, 16, 32)}        # 10-29 jaw at chunk rows k: (pred open, gt open, gt open at k1) per anchor and arm
    # 10-26: unconditional breakdown (every evaluated anchor): grasp (anchor within +-0.5 s of a jaw transition, per arm) vs free, and
    # episode phase (anchor position in the episode, thirds) -> k16 / k32 translation mm, rotation deg, zero-motion mm
    BD = {}
    def bd_add(key, mm, dg, zm):
        d = BD.setdefault(key, {f"{m}_k{k}": [] for m in ("mm", "deg", "zero_mm") for k in (16, 32)})
        for k in (16, 32): d[f"mm_k{k}"].append(mm[k - 1]); d[f"deg_k{k}"].append(dg[k - 1]); d[f"zero_mm_k{k}"].append(zm[k - 1])
    PE = []                                  # 10-30: per-episode means (small aux sets: a mean over 9 episodes hides one bad demo)
    try: keys = [r["episode_key"] for r in csv.DictReader(open(root / "episode_table.csv"))]
    except Exception: keys = None
    for e in np.unique(epi):
        n0 = {k: len(v) for k, v in E.items()}; c0 = {k: len(v) for k, v in CF.items()}
        last = int(np.flatnonzero(epi == e).max()); first = int(np.flatnonzero(epi == e).min())
        sel = np.flatnonzero(epi == e)[:: a.stride]
        if a.max_frames: sel = sel[: a.max_frames]
        gp, gg = [], []
        for j, i in enumerate(sel):
            it = ds[int(i)]; gt = it["action"].numpy()
            gt = gt[1:] if gt.shape[0] == 33 else gt
            pr = predict(cfg, pol, pre, post, it, j); n += 1
            zero = np.tile(np.r_[0, 0, 0, 1, 0, 0, 0, 1, 0], 2); Z = np.tile(np.r_[zero, gt[0, 18:20]], (32, 1))
            Rt = {s: pose9_to_T(P[int(i), sl].astype(np.float64))[:3, :3] for s, sl in (("L", slice(0, 9)), ("R", slice(9, 18)))}
            disp = {}
            for s, sl in (("L", slice(0, 9)), ("R", slice(9, 18))):
                mm, dg = err(pr, gt, sl); zm, _ = err(Z, gt, sl)
                for k in KS: E[f"{s}_mm_k{k}"].append(mm[k - 1]); E[f"{s}_deg_k{k}"].append(dg[k - 1]); E[f"{s}_zero_mm_k{k}"].append(zm[k - 1])
                gsub = "grasp" if (GW is not None and GW[int(i), 0 if s == "L" else 1] > 0) else "free"
                ph = ("early", "mid", "late")[min(2, int(3 * (int(i) - first) / max(1, last - first + 1)))]
                for key in ("all", gsub, f"phase_{ph}", f"{gsub}_{s}"): bd_add(key, mm, dg, zm)
                disp[s] = (Rt[s] @ pr[31, sl][:3], Rt[s] @ gt[31, sl][:3])
                if PC is not None and int(i) + 32 <= last:                  # confident anchor AND confident 32-step target only
                    ai = 0 if s == "L" else 1
                    if PC[int(i), ai] > 0 and PC[int(i) + 32, ai] > 0:
                        sub = "grasp" if (GW is not None and GW[int(i), ai] > 0) else "free"
                        for sb in ("all", sub): CF[f"{s}_{sb}_mm"].append(mm[31]); CF[f"{s}_{sb}_zero"].append(zm[31])
            cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
            for s, o in (("L", "R"), ("R", "L")):
                if np.linalg.norm(disp[s][1]) > 0.02 and np.linalg.norm(disp[o][1]) > 0.02:
                    cos_same.append(cs(disp[s][0], disp[s][1])); cos_other.append(cs(disp[s][0], disp[o][1]))
            gp.append(pr[0, 18:20] > 0.5); gg.append(gt[0, 18:20] > 0.5)
            for k in GH: GH[k] += [(pr[k - 1, 18 + c] > 0.5, gt[k - 1, 18 + c] > 0.5, gt[0, 18 + c] > 0.5) for c in (0, 1)]
        pe = dict(episode_index=int(e), key=keys[int(e)] if keys and int(e) < len(keys) else None, n=len(sel))
        for k, v in E.items():
            if any(f"_k{x}" == k[-len(f"_k{x}"):] for x in (8, 16, 32)): pe[k] = round(float(np.mean(v[n0[k]:])), 1) if len(v) > n0[k] else None
        cm = CF["L_all_mm"][c0["L_all_mm"]:] + CF["R_all_mm"][c0["R_all_mm"]:]; cz = CF["L_all_zero"][c0["L_all_zero"]:] + CF["R_all_zero"][c0["R_all_zero"]:]
        pe["conf_k32_mm"] = round(float(np.mean(cm)), 1) if cm else None; pe["conf_k32_zero_mm"] = round(float(np.mean(cz)), 1) if cz else None; pe["conf_n"] = len(cm)
        PE.append(pe)
        gp, gg = np.array(gp), np.array(gg); g_agree.append((gp == gg).mean(0))
        for c in (0, 1):
            er, mi, sp = match_events(events(gp[:, c]), events(gg[:, c]), fps / a.stride); ev_err += er; ev_miss += mi; ev_spur += sp; ev_n += len(events(gg[:, c]))
    q = lambda x: dict(mean=round(float(np.mean(x)), 1), median=round(float(np.median(x)), 1), p90=round(float(np.percentile(x, 90)), 1)) if len(x) else None
    dev = [np.mean(E[f"{s}_mm_k{k}"]) for s in "LR" for k in (16, 32)]
    conf = None
    if PC is not None:
        qq = lambda x: dict(mean=round(float(np.mean(x)), 1), median=round(float(np.median(x)), 1), n=len(x)) if len(x) else None
        conf = {k: qq(v) for k, v in CF.items()}
        mm_all = CF["L_all_mm"] + CF["R_all_mm"]; z_all = CF["L_all_zero"] + CF["R_all_zero"]
        conf["k32_trans_mm_conf"] = round(float(np.mean(mm_all)), 1) if mm_all else None; conf["stay_still_mm_conf"] = round(float(np.mean(z_all)), 1) if z_all else None
        for sub in ("grasp", "free"):
            v = CF[f"L_{sub}_mm"] + CF[f"R_{sub}_mm"]; zz = CF[f"L_{sub}_zero"] + CF[f"R_{sub}_zero"]
            conf[f"{sub}_k32_mm"] = round(float(np.mean(v)), 1) if v else None; conf[f"{sub}_stay_still_mm"] = round(float(np.mean(zz)), 1) if zz else None
    m_ = lambda x: round(float(np.mean(x)), 1) if len(x) else None
    breakdown = {key: {**{m: m_(v) for m, v in d.items()}, "n": len(d["mm_k32"])} for key, d in sorted(BD.items())}
    rep = dict(ckpt=a.ckpt, policy=cfg.type, data=str(root), samples=n, stride=a.stride, val_loss=vl, confident_frames_metrics=conf,
               confident_rule="anchor t and target t+32 both tag-backed or fgo gap <= 0.5 s (per arm); grasp = anchor within +-0.5 s of a jaw transition; cut-offs: < 60 mm meaningful, 40-50 promising, 30-40 rollout-worthy",
               val_score_mm=round(float(np.mean(dev)), 2), val_score_def="mean (not median: val is ~40 % near-static windows) L/R translation deviation (mm) at k=16 and k=32; early-stop metric",
               deviation={k: q(v) for k, v in E.items()},
               per_episode=PE, breakdown=breakdown, breakdown_def="mean over every evaluated anchor and arm (no confidence filter); grasp = anchor within +-0.5 s of a jaw transition (per arm); phase = thirds of the episode; zero_mm = stay-still baseline",
               lr_swap_check=dict(cos_pred_vs_same_arm_gt=q(cos_same), cos_pred_vs_other_arm_gt=q(cos_other), n=len(cos_same)),
               gripper=dict(state_agreement_LR=np.mean(g_agree, 0).round(3).tolist() if g_agree else None, gt_events=ev_n, matched=len(ev_err),
                            timing_err_ms=dict(median_abs=round(float(np.median(np.abs(ev_err))), 0), mean_signed=round(float(np.mean(ev_err)), 0)) if ev_err else None,
                            missed=ev_miss, spurious=ev_spur, rule="pred k=1 gripper > 0.5 = open; events matched within +-1 s"),
               gripper_horizon=gh_report(GH),
               gripper_horizon_def="10-29: k1 agreement is reachable by COPYING the input jaw state (Exp 1/2 did: 98 % here, never opened on the robot). "
                                   "acc_k = jaw at chunk row k vs target; switch_k = only anchors whose target jaw at k differs from row 1 "
                                   "(a copy-the-state policy scores ~0 there); copy_baseline_* = predicting the row-1 target for every row")
    (out / "offline_eval.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
