#!/usr/bin/env python3
"""[2026-11-05] Render whole demos with the v1 CNN grasp labels (trained on ALL current gold): every close event of both jaws ->
held (p_empty < LO) / empty (p_empty >= HI) / unknown (between), same layout as grasp_label_v0 videos (+ p on the banner via events.csv).
usage: grasp_render_v1.py OUT_DIR N_DEMOS [SEED]"""
import csv, json, pathlib, random, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1105
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_label_v0 as G, grasp_cnn_v1 as C  # noqa: E402
LO, HI = 0.3, 0.7

X, y, tk, ids = C.crops(); m = C.train_fold(X, y); gold_ids = set(ids.tolist()); print(f"CNN trained on {len(y)} gold ({y.sum()} empty)", flush=True)
demos = sorted({e["video"].rsplit("_", 1)[0] for e in json.load(open(H / "Desktop/grasp_gold_r2/all_events.json"))})
pick = sorted(random.Random(SEED).sample(demos, N)); rows = []
for dk in pick:
    take = "_".join(dk.split("_")[:2]); d = int(dk.split("_")[2][1:]) - 1; t0, t1 = G.demo_window(take, d); hands = {}
    for h, cam in G.CAMS:
        cd = G.TR / take / cam; J = np.load(cd / "umi_jaw.npz"); tw = np.load(cd / "t_ns.npy"); mask, tip = G.jaw_mask_and_tip(cd, tw, J)
        cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H; x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464))
        tj = J["t_ns"]; lab = np.where(J["state"] == 0, 4, 0).astype(np.int8); ev = list(zip(J["ev_t"].tolist(), J["ev_kind"].tolist())); E = []
        for k, (te, kind) in enumerate(ev):
            if kind != "close": continue
            tn = next((tb for tb, kb in ev[k + 1:] if kb == "open"), int(tj[-1]))
            if tn < t0 or te > t1: continue
            rest = te < t0 + G.EDGE_S * 1e9 or tn > t1 - G.EDGE_S * 1e9; fr = []
            for dt in C.DTS:
                t = te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)); i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
                im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg")); im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
                fr.append(cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224)))
            p = float(C.predict(m, np.stack(fr)[None])[0]); l = "empty" if p >= HI else "held" if p < LO else "unknown"
            seg = (tj >= te) & (tj < tn); lab[seg] = G.LBL.index(l); lab[seg & (tj < te + G.CLOSING_S * 1e9)] = 1
            eid = f"{dk}_{h}{len(E) + 1}"; E.append(dict(t_close_ns=te, t_open_ns=tn, rest=int(rest)))
            rows.append(dict(demo=dk, event=f"{h}{len(E)}", t_close_s=round((te - t0) / 1e9, 1), closed_s=round((tn - te) / 1e9, 2), p_empty=round(p, 2), label=l, rest=int(rest), in_gold=int(eid in gold_ids)))
        hands[h] = dict(tj=tj, lab=lab, events=E, mask=mask, tip=tip, r=int(0.09 * G.W), tw=tw, cam_dir=cd)
    n = {k: sum(r["label"] == k and not r["rest"] for r in rows if r["demo"] == dk) for k in ("held", "empty", "unknown")}
    name = f"{dk}_H{n['held']}E{n['empty']}U{n['unknown']}"; G.render(OUT / f"{name}.mp4", take, d, t0, t1, hands); print(name, flush=True)
with open(OUT / "events.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
