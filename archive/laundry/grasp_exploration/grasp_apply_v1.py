#!/usr/bin/env python3
"""[2026-11-05] Apply the v1 grasp CNN (trained on ALL current gold) to every close event of the 11-03 + 11-04 flatten demos, with the
agreed rule: p_grasp > 0.9 -> held, p_grasp < 0.2 -> empty, else unknown (masked).  A human gold label overrides the auto label.
-> OUT/grasp_events_v1.csv (demo, hand, event, t_close_ns, t_open_ns, closed_s, rest, p_grasp, auto, gold, label, source)
usage: grasp_apply_v1.py OUT_DIR"""
import csv, json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_label_v0 as G, grasp_label_v1_temporal as V, grasp_cnn_v1 as C  # noqa: E402
HELD_P, EMPTY_P = 0.9, 0.2

X, y, tk, ids = C.crops(); m = C.train_fold(X, y); print(f"CNN trained on {len(y)} gold ({y.sum()} empty)", flush=True)
gold = {e["id"]: e["gold"] for e in V.gold()}
S = "/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad/"
keys = set()
for f in sorted(pathlib.Path(S).glob("flat1103n_*_S03G_split.json")) + sorted(pathlib.Path(S).glob("flat1104_*_S03G_split.json")):
    for sp, ks in json.loads(f.read_text())["episodes"].items():
        if sp != "reject": keys |= set(ks)
rows = []
for n, k in enumerate(sorted(keys)):
    take, dn = k.split("/"); d = int(dn[4:]); t0, t1 = G.demo_window(take, d)
    for h, cam in G.CAMS:
        cd = G.TR / take / cam; J = np.load(cd / "umi_jaw.npz"); tw = np.load(cd / "t_ns.npy"); tip = G.jaw_mask_and_tip(cd, tw, J)[1]
        cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H; x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464))
        ev = list(zip(J["ev_t"].tolist(), J["ev_kind"].tolist())); en = 0
        for i_, (te, kind) in enumerate(ev):
            if kind != "close": continue
            tn = next((tb for tb, kb in ev[i_ + 1:] if kb == "open"), int(J["t_ns"][-1]))
            if tn < t0 or te > t1: continue
            en += 1; fr = []
            for dt in C.DTS:
                t = te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)); i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
                im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg")); im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
                fr.append(cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224)))
            pg = 1.0 - float(C.predict(m, np.stack(fr)[None])[0]); auto = "held" if pg > HELD_P else "empty" if pg < EMPTY_P else "unknown"
            eid = f"{take}_d{d + 1:02d}_{h}{en}"; g = gold.get(eid, "")
            rows.append(dict(demo=k, hand=h, event=f"{h}{en}", t_close_ns=te, t_open_ns=tn, closed_s=round((tn - te) / 1e9, 2),
                             rest=int(te < t0 + G.EDGE_S * 1e9 or tn > t1 - G.EDGE_S * 1e9), p_grasp=round(pg, 3), auto=auto, gold=g,
                             label=g if g else auto, source="gold" if g else "auto"))
    if n % 25 == 0: print(f"  {n}/{len(keys)} demos", flush=True)
with open(OUT / "grasp_events_v1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} events", flush=True)
