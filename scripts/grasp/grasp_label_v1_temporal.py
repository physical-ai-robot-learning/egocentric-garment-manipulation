#!/usr/bin/env python3
"""[2026-11-05] Grasp held / empty labeler v1 (temporal, jaw-local) trained on the user's gold events.

Gold = round 1 (~/Desktop/grasp_label_v0_30, minus the 59 clips that showed two closes) + round 2 (~/Desktop/grasp_gold_r2, incl. those 59
re-labeled on single-close clips).  Per event (that hand's wrist cam):
  jaw ROI video: 8 frames at close + (-0.3 .. 1.1 s), 256 px square around the jaw tip (v0 tip finder), frozen CLIP B/16 embeddings
  -> mean over the closed frames, late - early difference
  motor: log closed duration, closed position rel. to the take's calibration, closing speed; flow: v0 tip-disk dark share / flow ratio
Logistic regression (PCA 32 of the CLIP part inside each fold), leave-one-TAKE-out CV; p in the middle band -> unknown.
usage: grasp_label_v1_temporal.py OUT_DIR
"""
import csv, json, pathlib, sys
import cv2, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import grasp_label_v0 as G  # noqa: E402

H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
R1, R2, R3 = H / "Desktop/grasp_label_v0_30", H / "Desktop/grasp_gold_r2", H / "Desktop/grasp_gold_r3"
DTS = (-0.3, 0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1); CROP = 256; DEV = "mps"


def gold():
    sel = {}
    for d in (R1, R2, R3):
        for e in json.load(open(d / "gold_selection.json")): sel[e["id"]] = e
    two = set((pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad/gl/r1_twoclose_ids.txt")).read_text().split())
    g = {r["id"]: r["gold"] for r in csv.DictReader(open(R1 / "gold_events.csv")) if r["id"] not in two}
    for d in (R2, R3):                 # r3 (11-05) = CNN-mined likely empties + uncertain band
        if (d / "gold_events.csv").exists(): g.update({r["id"]: r["gold"] for r in csv.DictReader(open(d / "gold_events.csv"))})
    return [dict(sel[i], gold=v) for i, v in g.items() if v in ("held", "empty")]


def event_times(e):
    take = "_".join(e["video"].split("_")[:2]); d = int(e["video"].split("_")[2][1:]) - 1; t0, _ = G.demo_window(take, d)
    cam = dict(G.CAMS)[e["event"][0]]; J = np.load(G.TR / take / cam / "umi_jaw.npz")
    ev = list(zip(J["ev_t"].tolist(), J["ev_kind"].tolist()))
    te = min((t for t, k in ev if k == "close"), key=lambda t: abs(t - (t0 + e["t_close_s"] * 1e9)))
    return take, cam, J, te, te + e["closed_s"] * 1e9


def main():
    E = gold(); print(f"gold held/empty events: {len(E)}  empty {sum(e['gold'] == 'empty' for e in E)}", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval(); proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tips, rows = {}, []
    for n, e in enumerate(E):
        take, cam, J, te, tn = event_times(e); cd = G.TR / take / cam; tw = np.load(cd / "t_ns.npy")
        if (take, cam) not in tips: tips[(take, cam)] = G.jaw_mask_and_tip(cd, tw, J)
        mask, tip = tips[(take, cam)]
        cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H
        ims = []
        for dt in DTS:
            i = int(np.searchsorted(tw, te + dt * 1e9).clip(0, len(tw) - 1)); im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg"))
            if im is None: im = np.zeros((720, 1280, 3), np.uint8)
            x0 = int(np.clip(cx - CROP // 2, 0, 1280 - CROP)); y0 = int(np.clip(cy - CROP * 0.6, 0, 720 - CROP))
            ims.append(cv2.cvtColor(im[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
            f = torch.nn.functional.normalize(mdl.visual_projection(mdl.vision_model(pixel_values=pv).pooler_output), dim=-1).cpu().numpy()
        closed = np.array([0 <= dt * 1e9 < tn - te for dt in DTS]); closed[1] = True
        clipf = np.concatenate([f[closed].mean(0), f[closed][-1] - f[1]])
        # motor + v0 flow features
        tj, mp = J["t_ns"], J["motor_pos"]; cal = json.loads(str(J["cal"])); sgn = 1 if cal["closed"] > cal["open"] else -1; span = abs(cal["closed"] - cal["open"])
        s_ = (tj > te + 0.3e9) & (tj < tn); jr = sgn * (np.median(mp[s_]) - cal["closed"]) / span if s_.sum() > 3 else 0.0
        c_ = (tj >= te - 0.2e9) & (tj <= te + 0.3e9); vs = float(np.abs(np.diff(mp[c_])).max() / span) if c_.sum() > 3 else 0.0
        _, _, ff = G.event_features(cd, tw, te + 0.3e9, min(tn, te + 1.5e9), mask, tip, int(0.09 * G.W))
        mot = [np.log(max((tn - te) / 1e9, 0.1)), jr, vs, ff["cloth"] if ff["cloth"] is not None else 0.5, ff["ratio"] if ff["ratio"] is not None else 0.5, float(ff["n_moving"] > 0)]
        rows.append(dict(id=e["id"], take=take, y=int(e["gold"] == "empty"), clip=clipf, mot=np.array(mot, np.float32)))
        if n % 50 == 0: print(f"  feats {n}/{len(E)}", flush=True)
    np.savez(OUT / "v1_feats.npz", ids=[r["id"] for r in rows], take=[r["take"] for r in rows], y=[r["y"] for r in rows],
             clip=np.stack([r["clip"] for r in rows]), mot=np.stack([r["mot"] for r in rows]))
    evaluate(OUT / "v1_feats.npz")


def logreg(A, y, B, lam):
    A = torch.tensor(A, dtype=torch.float32); B = torch.tensor(B, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3))
    opt = torch.optim.LBFGS([w, b], max_iter=300)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt, pos_weight=pw) + lam * (w ** 2).sum() / len(y); l.backward(); return l
    opt.step(cl); return torch.sigmoid(B @ w + b).detach().numpy()


def cv_pred(Z, kind, npc=32, lam=10.0):
    y = Z["y"]; takes = Z["take"]; p = np.zeros(len(y))
    for t in np.unique(takes):
        te = takes == t; parts_tr, parts_te = [], []
        if kind in ("clip", "all"):
            C = Z["clip"]; mu = C[~te].mean(0); U, S, Vt = np.linalg.svd(C[~te] - mu, full_matrices=False); P = Vt[:npc].T
            parts_tr.append((C[~te] - mu) @ P); parts_te.append((C[te] - mu) @ P)
        if kind in ("mot", "all"): parts_tr.append(Z["mot"][~te]); parts_te.append(Z["mot"][te])
        A, B = np.hstack(parts_tr), np.hstack(parts_te); m, s = A.mean(0), A.std(0) + 1e-6
        p[te] = logreg((A - m) / s, y[~te], (B - m) / s, lam)
    return p


def f1s(y, p, lo, hi):
    dec = (p >= hi) | (p < lo); pe = p >= hi; out = {}
    for c, pc, yc in (("empty", pe, y == 1), ("held", ~pe, y == 0)):
        tp = (pc & yc & dec).sum(); pr = tp / max((pc & dec).sum(), 1); rc = tp / max((yc & dec).sum(), 1); out[c] = (round(pr, 2), round(rc, 2), round(2 * pr * rc / max(pr + rc, 1e-9), 2))
    return out, round(dec.mean(), 2)


def evaluate(path):
    Z = dict(np.load(path, allow_pickle=True)); Z["y"] = np.asarray(Z["y"]); Z["take"] = np.asarray(Z["take"])
    print(f"events {len(Z['y'])}  empty {Z['y'].sum()}  takes {len(np.unique(Z['take']))}", flush=True)
    res = {}
    for kind in ("mot", "clip", "all"):
        for lam in (1.0, 10.0, 100.0):
            p = cv_pred(Z, kind, lam=lam); res[(kind, lam)] = p
            for lo, hi in ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8)):
                o, cov = f1s(Z["y"], p, lo, hi)
                print(f"{kind:4s} lam{lam:<5} band[{lo},{hi}] coverage {cov:.2f}  empty P/R/F1 {o['empty']}  held P/R/F1 {o['held']}", flush=True)
    np.save(pathlib.Path(path).with_name("v1_cv_pred.npy"), np.stack([res[k] for k in sorted(res)]))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "eval": evaluate(OUT / "v1_feats.npz")
    else: main()
