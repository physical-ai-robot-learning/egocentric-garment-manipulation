#!/usr/bin/env python3
"""[2026-11-05] Round-3 labeling set: train the v1 CNN on ALL current gold, score every not-yet-queued close event
(~/Desktop/grasp_gold_r2/all_events.json), pick the likely empties + the uncertain band -> OUT/events.csv (label = mined_p<prob>).
usage: grasp_mine_r3.py OUT_DIR [N=200]"""
import csv, json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_label_v0 as G, grasp_label_v1_temporal as V, grasp_cnn_v1 as C  # noqa: E402

X, y, tk, ids = C.crops(); m = C.train_fold(X, y); print(f"trained on {len(y)} gold ({y.sum()} empty)", flush=True)
queued = set(ids)
for d in (H / "Desktop/grasp_label_v0_30", H / "Desktop/grasp_gold_r2"):
    queued |= {e["id"] for e in json.load(open(d / "gold_selection.json"))}
pool = [e for e in json.load(open(H / "Desktop/grasp_gold_r2/all_events.json")) if e["id"] not in queued]
print(f"scoring {len(pool)} unqueued events", flush=True)
tips, P = {}, []
for n, e in enumerate(pool):
    take, cam, J, te, tn = V.event_times(e); cd = G.TR / take / cam; tw = np.load(cd / "t_ns.npy")
    if (take, cam) not in tips: tips[(take, cam)] = G.jaw_mask_and_tip(cd, tw, J)[1]
    tip = tips[(take, cam)]; cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H
    x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464)); fr = []
    for dt in C.DTS:
        t = te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)); i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
        im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg")); im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
        fr.append(cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224)))
    P.append(float(C.predict(m, np.stack(fr)[None])[0]))
    if n % 300 == 0: print(f"  {n}/{len(pool)}", flush=True)
P = np.array(P); json.dump([dict(e, p_empty=round(p, 3)) for e, p in zip(pool, P)], open(OUT / "pool_scores.json", "w"))
order = np.argsort(-P); n_top = int(N * 0.7)
top = list(order[:n_top]); band = [i for i in np.argsort(np.abs(P - 0.5)) if i not in set(top)][:N - n_top]
sel = top + band; rng = np.random.default_rng(1105); rng.shuffle(sel)
print(f"pool p>=0.5: {(P >= 0.5).sum()}  top{n_top} min p {P[order[n_top - 1]]:.2f}  band {len(band)}", flush=True)
with open(OUT / "events.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["video", "event", "t_close_s", "closed_s", "label", "rest"]); w.writeheader()
    for i in sel:
        e = pool[i]; w.writerow(dict(video=e["video"], event=e["event"], t_close_s=e["t_close_s"], closed_s=e["closed_s"], label=f"mined_p{P[i]:.2f}", rest=0))
