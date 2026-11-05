#!/usr/bin/env python3
"""[2026-11-05] Grasp held/empty classifier v1 on the user's 188 gold events (gold_events.csv): frozen CLIP ViT-B/16 image features of
that hand's wrist frame at close + {0.4, 0.8, 1.2} s (jaw closed) + closed duration -> L2 logistic regression, leave-one-TAKE-out CV.
usage: grasp_clf_v1.py DIR"""
import json, pathlib, sys
import cv2, numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import grasp_label_v0 as G  # noqa: E402
from transformers import CLIPModel, CLIPProcessor
D = pathlib.Path(sys.argv[1]).expanduser(); dev = "mps"
m = pd.read_csv(D / "gold_feats.csv"); m["take"] = m.video.str.split("_").str[:2].str.join("_")
cache = D / "clip_feats.npy"
if cache.exists(): X = np.load(cache)
else:
    mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval(); proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    X = []
    for _, r in m.iterrows():
        d = int(r.video.split("_")[2][1:]) - 1; t0, _ = G.demo_window(r["take"], d); cd = G.TR / r["take"] / dict(G.CAMS)[r.event[0]]; tw = np.load(cd / "t_ns.npy")
        te = t0 + r.t_close_s * 1e9; ims = []
        for dt in (0.4, 0.8, 1.2):
            t = te + min(dt, max(r.closed_s - 0.1, 0.2)) * 1e9; i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
            im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg")); ims.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev); f = mdl.visual_projection(mdl.vision_model(pixel_values=pv).pooler_output); f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
        X.append(f.reshape(-1))
    X = np.stack(X); np.save(cache, X)
y = (m.gold == "empty").astype(int).values
dur = np.log(m.closed_s.values.clip(0.1))[:, None]


def fit_pred(Xtr, ytr, Xte, lam=1.0, it=400):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); yt = torch.tensor(ytr, dtype=torch.float32)
    pw = (1 - ytr.mean()) / max(ytr.mean(), 1e-3); opt = torch.optim.LBFGS([w, b], max_iter=it)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt, pos_weight=torch.tensor(pw)) + lam * (w ** 2).sum() / len(yt); l.backward(); return l
    opt.step(cl); return torch.sigmoid(B @ w + b).detach().numpy()


def cv(F, lam):
    p = np.zeros(len(y))
    for t in m["take"].unique():
        te = (m["take"] == t).values; p[te] = fit_pred(F[~te], y[~te], F[te], lam)
    return p


def report(name, p, lo=0.5, hi=0.5):
    dec = (p >= hi) | (p < lo); pe = p >= hi; out = {}
    for c, pc, yc in (("empty", pe, y == 1), ("held", ~pe, y == 0)):
        tp = (pc & yc & dec).sum(); pr = tp / max((pc & dec).sum(), 1); rc = tp / max((yc & dec).sum(), 1); out[c] = round(2 * pr * rc / max(pr + rc, 1e-9), 2)
    print(f"{name:34s} F1 empty {out['empty']:.2f} held {out['held']:.2f}  coverage {dec.mean():.2f}", flush=True); return out


for lam in (1.0, 10.0, 100.0):
    report(f"dur only lam{lam}", cv(dur, lam))
    report(f"clip lam{lam}", cv(X, lam))
    report(f"clip+dur lam{lam}", cv(np.hstack([X, dur * 5]), lam))
