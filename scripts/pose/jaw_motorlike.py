#!/usr/bin/env python3
"""[2026-10-30] Motor-like jaw labels for camera-only takes (user: "281개 데이터가 너무 그리퍼가 안좋아" -> relabel GOLD-281's gripper).

The wrist-camera jaw classifier (umi_jaw v003) that labelled GOLD-281 agrees with the Feetech servo only ~70-80 % per frame on the motor
takes: event timing is fine (median lag < 0.06 s) but ~40 % of its open/close events are not in the servo trace.  Here a per-hand
ResNet18 is trained on the SAME wrist crops (umi_jaw ROI, 224x112 grey) with the servo as the target, so it can relabel the camera-only
takes.  Two heads (10-30 review): [0] open logit (BCE on the servo open/closed state, same hysteresis as the evaluation) and [1] motor-like
aperture (MSE of sigmoid on the servo g_cont normalised per hand -- right servo plateaus at ~0.65 open: stiff linkage).

  build  <out_dir>                      -> <out_dir>/<hand>.npz  (crops, target, take id) from every motor take, every STRIDE-th frame
  train  <out_dir> --hand H --val TAKE  -> <out_dir>/jaw_motorlike_<hand>.pt (TorchScript, same I/O as umi_jaw models)
  eval   <out_dir> --hand H TAKE...     -> per take: frame F1 / agreement, event recall / precision / F1, lag, new model AND v003
  train --val-date 20261029            -> cross-date test: hold out every take of that date (domain-shortcut check)
  apply  <model_dir> TAKE...            -> <take>/<wrist dir>/umi_jaw_motorlike.npz (same keys as umi_jaw.npz); umi_jaw.npz untouched
"""
import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visual_teleop.tools import umi_jaw as U  # noqa: E402

TR = ROOT / "datasets/tag_rig"
DIRS = {"right": "wrist", "left": "wrist_left"}
OPEN_PLATEAU = {"left": 1.0, "right": 0.65}     # servo g_cont when fully open (10-30 histograms: left 0.8-1.0, right 0.6-0.7)
HI, LO, DWELL = 0.55, 0.35, 0.2                 # hysteresis on the normalised servo AND on the model output (same rule both sides)
# 10-31 (user review: right hand wrong on GOLD 10-26 pm = black cloth over black jaws): a frame is UNKNOWN when the model is unsure
# (p in (UNSURE_LO, UNSURE_HI)) for most of a +-UNSURE_WIN_S window, or the crop is mostly dark (cloth over the lens / jaws invisible).
# UNKNOWN -> `valid` 0 in umi_jaw_motorlike.npz (the dataset builder masks the BCE jaw loss there); `state` keeps the hysteresis hold.
UNSURE_LO, UNSURE_HI, UNSURE_WIN_S, UNSURE_FRAC, DARK_PX, DARK_FRAC = 0.2, 0.8, 0.25, 0.5, 40, 0.6
STRIDE = 3


def motor_takes():
    out = []
    for t in sorted(TR.glob("2026102[9]_*")) + sorted(TR.glob("2026103*_*")):
        z = t / "wrist_left" / "umi_jaw.npz"
        if z.exists() and "motor" in str(np.load(z, allow_pickle=True)["model"]): out.append(t)
    return out


def load_crops(w: pathlib.Path, hand: str, n: int):
    try:
        c = np.load(w / "crops.npz")["crops"]
        if len(c) >= n: return c[:n]
    except Exception: pass
    import cv2   # crops.npz missing / truncated -> re-crop from the saved frames (same ROI as the recorder)
    return np.stack([U.crop(cv2.imread(str(w / f"wrist_{k:05d}.jpg")), U.ROIS[hand]) for k in range(n)])


def servo_target(w: pathlib.Path, hand: str, t: np.ndarray):
    m = np.load(w / "umi_jaw.npz", allow_pickle=True)
    g = np.interp(t.astype(np.float64), m["t_ns"].astype(np.float64), m["g_cont"].astype(np.float64)) / OPEN_PLATEAU[hand]
    v = np.interp(t.astype(np.float64), m["t_ns"].astype(np.float64), m["valid"].astype(np.float64)) > 0.5 if "valid" in m.files else np.ones(len(t), bool)
    return np.clip(g, 0, 1).astype(np.float32), v


def build(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for hand, d in DIRS.items():
        C, Y, S, K = [], [], [], []
        for t in motor_takes():
            w = t / d
            if not (w / "t_ns.npy").exists(): continue
            ts = np.load(w / "t_ns.npy"); crops = load_crops(w, hand, len(ts)); n = min(len(ts), len(crops))
            y, v = servo_target(w, hand, ts[:n]); st, _ = U.events(ts[:n], y, HI, LO, DWELL); idx = np.arange(0, n, STRIDE); idx = idx[v[idx]]
            C.append(crops[idx]); Y.append(y[idx]); S.append(st[idx].astype(np.float32)); K += [t.name] * len(idx)
            print(f"{hand} {t.name}: {len(idx)} frames, open share {float(st[idx].mean()):.2f}", flush=True)
        np.savez(out / f"{hand}.npz", crops=np.concatenate(C), target=np.concatenate(Y), state=np.concatenate(S), take=np.array(K))
        print(f"== {hand}: {sum(map(len, Y))} frames", flush=True)


def train(a):
    import torch, torchvision
    z = np.load(pathlib.Path(a.out) / f"{a.hand}.npz"); X, Y, S, K = z["crops"], z["target"], z["state"], z["take"]
    val = np.array([k.startswith(a.val_date) for k in K]) if a.val_date else np.isin(K, a.val)
    tr = np.where(~val)[0]; vi = np.where(val)[0]
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0); torch.manual_seed(0)
    net = torchvision.models.resnet18(); net.fc = torch.nn.Linear(512, 1)
    net.load_state_dict(torch.jit.load(str(U.MODELS[a.hand])).state_dict())     # start from v003 (same crops, same architecture)
    fc = torch.nn.Linear(512, 2)                                                  # [0] open logit (v003 head copied), [1] aperture
    with torch.no_grad(): fc.weight[0] = net.fc.weight[0]; fc.bias[0] = net.fc.bias[0]
    net.fc = fc; net.to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    pos = float(S[tr].mean()); wpos = (1 - pos) / max(pos, 1e-3)                  # open frames are 10-30 % -> class-balanced BCE
    print(f"{a.hand}: train {len(tr)} val {len(vi)} frames ({a.val_date or ','.join(a.val)} held out), open share {pos:.2f} (pos weight {wpos:.1f}), {dev}", flush=True)
    Xv = torch.tensor(np.stack([U.to_input(X[i]) for i in vi[::2]])) if len(vi) else None
    for ep in range(a.epochs):
        net.train(); rng.shuffle(tr); tot = 0.0
        for b in range(0, len(tr), 64):
            idx = tr[b:b + 64]; x = torch.tensor(np.stack([U.to_input(U.augment(X[i], rng)) for i in idx])).to(dev)
            s_ = torch.tensor(S[idx]).to(dev); y = torch.tensor(Y[idx]).to(dev); o = net(x)
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(o[:, 0], s_, reduction="none") * torch.where(s_ > 0.5, wpos, 1.0)).mean()
            mse = torch.nn.functional.mse_loss(torch.sigmoid(o[:, 1]), y)
            loss = bce + mse; opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        msg = f"epoch {ep + 1}: loss {tot / len(tr):.4f}"
        if Xv is not None:
            net.eval()
            with torch.no_grad(): o = torch.cat([net(Xv[i:i + 128].to(dev)).cpu() for i in range(0, len(Xv), 128)]).numpy()
            sv = S[vi[::2]] > 0.5; pr = o[:, 0] > 0; tp = float((pr & sv).sum())
            f1 = 2 * tp / max(float(pr.sum() + sv.sum()), 1); mae = float(np.abs(1 / (1 + np.exp(-o[:, 1])) - Y[vi[::2]]).mean())
            msg += f" | held-out frame acc {float((pr == sv).mean()):.3f} open-F1 {f1:.3f} aperture MAE {mae:.3f}"
        print(msg, flush=True)
    net.eval().cpu(); tag = f"_val{a.val_date}" if a.val_date else f"_val{'-'.join(a.val)}" if a.val else ""   # no holdout = the model `apply` uses
    o = pathlib.Path(a.out) / f"jaw_motorlike_{a.hand}{tag}.pt"; torch.jit.trace(net, torch.zeros(1, 3, 224, 224)).save(str(o)); print("saved", o)


def predict(model, crops, both=False):
    """sigmoid of head 0 (open probability); both=True also returns head 1 (motor-like aperture). v003 has only head 0."""
    import torch
    net = torch.jit.load(str(model)).eval(); torch.set_num_threads(6)
    with torch.no_grad():
        o = torch.sigmoid(torch.cat([net(torch.tensor(np.stack([U.to_input(c) for c in crops[i:i + 64]]))) for i in range(0, len(crops), 64)])).numpy()
    return (o[:, 0], o[:, 1] if o.shape[1] > 1 else o[:, 0]) if both else o[:, 0]


def match(ev_ref, ev, tol=1.0):
    used, hit, dl = set(), 0, []
    for tr_, k in ev_ref:
        c = [(abs(te - tr_), i, te) for i, (te, ke) in enumerate(ev) if ke == k and i not in used and abs(te - tr_) / 1e9 <= tol]
        if c: _, i, te = min(c); used.add(i); hit += 1; dl.append((te - tr_) / 1e9)
    return hit / max(len(ev_ref), 1), hit / max(len(ev), 1), dl


def evaluate(a):
    res = []
    for take in a.takes:
        w = TR / take / DIRS[a.hand]; ts = np.load(w / "t_ns.npy"); crops = load_crops(w, a.hand, len(ts)); n = min(len(ts), len(crops)); ts = ts[:n]
        y, v = servo_target(w, a.hand, ts); st_ref, ev_ref = U.events(ts, y, HI, LO, DWELL)
        for name, model in (("motorlike", pathlib.Path(a.out) / f"jaw_motorlike_{a.hand}{a.tag}.pt"), ("v003", U.MODELS[a.hand])):
            p = predict(model, crops[:n]); hi, lo = (HI, LO) if name == "motorlike" else (0.6, 0.4)
            st, ev = U.events(ts, p, hi, lo, DWELL); rc, pr, dl = match(ev_ref, ev)
            sp, sr = st[v] > 0, st_ref[v] > 0; ffr = 2 * float((sp & sr).sum()) / max(float(sp.sum() + sr.sum()), 1)
            r = dict(take=take, hand=a.hand, model=name, frame_agree=float((sp == sr).mean()), frame_F1=ffr, recall=rc, precision=pr,
                     event_F1=2 * rc * pr / max(rc + pr, 1e-9), n_ref=len(ev_ref), n_ev=len(ev),
                     lag_med=float(np.median(dl)) if dl else None, lag_abs_p90=float(np.percentile(np.abs(dl), 90)) if dl else None)
            res.append(r); print({k: round(x, 3) if isinstance(x, float) else x for k, x in r.items()}, flush=True)
    for name in ("motorlike", "v003"):
        R = [r for r in res if r["model"] == name]; nr = sum(r["n_ref"] for r in R); ne = sum(r["n_ev"] for r in R)
        hit = sum(r["recall"] * r["n_ref"] for r in R); rc, pr = hit / max(nr, 1), hit / max(ne, 1)
        print(f"== {a.hand} {name}{a.tag}: event recall {rc:.3f} precision {pr:.3f} F1 {2 * rc * pr / max(rc + pr, 1e-9):.3f} | "
              f"frame F1 {np.mean([r['frame_F1'] for r in R]):.3f} agree {np.mean([r['frame_agree'] for r in R]):.3f}", flush=True)
    json.dump(res, open(pathlib.Path(a.out) / f"eval_{a.hand}{a.tag}.json", "w"), indent=1)


def apply(a):
    import cv2  # noqa: F401
    for take in a.takes:
        for hand, d in DIRS.items():
            w = pathlib.Path(take) / d
            if not (w / "t_ns.npy").exists(): continue
            ts = np.load(w / "t_ns.npy"); crops = load_crops(w, hand, len(ts)); n = min(len(ts), len(crops)); ts = ts[:n]
            model = pathlib.Path(a.out) / f"jaw_motorlike_{hand}.pt"; p, ap_ = predict(model, crops[:n], both=True)
            st, ev = U.events(ts, p, HI, LO, DWELL)
            kk = max(1, int(round(0.2 * (n - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9)))) | 1
            g = np.array([np.median(p[max(0, i - kk // 2):i + kk // 2 + 1]) for i in range(n)], np.float32)
            fps = (n - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9); kw = max(1, int(round(UNSURE_WIN_S * fps)))
            uns = ((p > UNSURE_LO) & (p < UNSURE_HI)).astype(float); uns = np.convolve(uns, np.ones(2 * kw + 1) / (2 * kw + 1), mode="same") > UNSURE_FRAC
            dark = (crops[:n] < DARK_PX).reshape(n, -1).mean(1) > DARK_FRAC
            valid = (~uns) & (~dark)
            np.savez(w / "umi_jaw_motorlike.npz", t_ns=ts, p_open=p, g_cont=g, aperture=ap_.astype(np.float32), state=st, valid=valid.astype(np.uint8), unsure=uns, dark=dark, hand=hand, model=model.name,
                     ev_t=np.array([e[0] for e in ev], np.int64), ev_kind=np.array([e[1] for e in ev]))
            old = np.load(w / "umi_jaw.npz", allow_pickle=True) if (w / "umi_jaw.npz").exists() else None
            agree = float((np.interp(ts.astype(float), old["t_ns"].astype(float), old["state"].astype(float))[:n] > 0.5).astype(int).__eq__(st).mean()) if old is not None else float("nan")
            print(f"{pathlib.Path(take).name} {hand}: open share {st.mean():.2f}, events {len(ev)} (old {len(old['ev_t']) if old is not None else '-'}), "
                  f"p in (0.2,0.8) {float(((p > 0.2) & (p < 0.8)).mean()):.2f}, UNKNOWN {1 - valid.mean():.2f} (unsure {uns.mean():.2f}, dark {dark.mean():.2f}), agree with old label {agree:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out")
    t = sub.add_parser("train"); t.add_argument("out"); t.add_argument("--hand", required=True); t.add_argument("--val", nargs="*", default=[]); t.add_argument("--val-date", default=""); t.add_argument("--epochs", type=int, default=3)
    e = sub.add_parser("eval"); e.add_argument("out"); e.add_argument("--hand", required=True); e.add_argument("--tag", default=""); e.add_argument("takes", nargs="+")
    p = sub.add_parser("apply"); p.add_argument("out"); p.add_argument("takes", nargs="+")
    a = ap.parse_args(); {"build": build, "train": train, "eval": evaluate, "apply": apply}[a.cmd](a)
