#!/usr/bin/env python3
"""[2026-11-05] Clips of the CONFIDENTLY WRONG v2 CV predictions: gold empty with P(grasp) > 0.9, gold held with P(grasp) < 0.2.
One clip per event = that hand's wrist cam cut at the previous v2 open / next v2 close (single close), close - 1 s .. open + 1 s (<= 7 s), 15 fps,
orange bar = v2 jaw closed; the file name carries gold + CV P(grasp).
usage: grasp_wrong_clips.py OUT_DIR"""
import csv, pathlib, subprocess, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(H / "ego_collector/scripts"))
import grasp_label_v0 as G, grasp_v2_events as E2  # noqa: E402
D = H / "laundry_dit_training/data/grasp_labels_v2"; z = np.load(D / "cnn_cv_v2.npz", allow_pickle=True)
pg = dict(zip(z["ids"].tolist(), (1 - z["p"]).tolist())); gold = {g["id"]: g for g in csv.DictReader(open(D / "gold_v2.csv"))}
rows = []
for i, p in pg.items():
    g = gold[i]
    if (g["gold"] == "empty" and p > 0.9) or (g["gold"] == "held" and p < 0.2): rows.append((g, p))
for g, p in sorted(rows, key=lambda r: (r[0]["gold"], -r[1])):
    tj, st, ev = E2.jaw_v2(g["take"], g["cam"]); te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
    po = max([t for t, k in ev if k == "open" and t < te], default=te - 10**10); nc = min([t for t, k in ev if k == "close" and t > te], default=tn + 10**10)
    a = max(te - 1e9, po + 0.15e9); b = min(tn + 1e9, nc - 0.15e9, a + 7e9)
    cd = G.TR / g["take"] / g["cam"]; tw = np.load(cd / "t_ns.npy")
    name = f"{'EMPTY' if g['gold'] == 'empty' else 'HELD'}_gold__P{p:.2f}__{g['id']}"; tmp = OUT / f"{name}.raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 400))
    for t in np.arange(a, b, 1e9 / 15):
        k = int(np.searchsorted(tw, t).clip(0, len(tw) - 1)); im = cv2.imread(str(cd / f"wrist_{k:05d}.jpg"))
        im = np.zeros((360, 640, 3), np.uint8) if im is None else cv2.resize(im, (640, 360)); closed = te <= t < tn
        bar = np.full((40, 640, 3), (0, 160, 255) if closed else (90, 90, 90), np.uint8)
        cv2.putText(bar, f"gold {g['gold'].upper()}  model P(grasp) {p:.2f}   {'JAW CLOSED' if closed else 'jaw open'}  {(t - a) / 1e9:4.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if closed else (255, 255, 255), 2)
        vw.write(np.vstack([im, bar]))
    vw.release()
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(OUT / f"{name}.mp4")], check=True); tmp.unlink()
    print(name, flush=True)
