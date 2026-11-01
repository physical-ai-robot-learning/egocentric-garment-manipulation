#!/usr/bin/env python3
"""[2026-10-30] Human check of the motor-like jaw relabel: for random demos of camera-only takes, a video with both wrist views and,
under each, the OLD label (umi_jaw.npz state, wrist-camera classifier v003) and the NEW one (umi_jaw_motorlike.npz state + aperture).
Event frames get a coloured border (green = opens, red = closes) and a hold of a few frames so they can be checked one by one.
usage: jaw_motorlike_overlay.py <out_dir> --n 20 --seed 0 TAKE_DIR...      (demo windows from <take>/segments.json)"""
import argparse, json, pathlib, random
import cv2
import numpy as np

DIRS = {"left": "wrist_left", "right": "wrist"}
W, H = 480, 270


def lab(w, name):
    f = w / name
    if not f.exists(): return None
    z = np.load(f, allow_pickle=True); return z


def at(z, key, t):
    return np.interp(t, z["t_ns"].astype(np.float64), z[key].astype(np.float64))


def strip(old, new, ap, w=W):
    """2 rows: OLD state / NEW state, plus the aperture line on the NEW row (open = light, closed = dark)."""
    im = np.zeros((44, w, 3), np.uint8)
    for r, (v, nm) in enumerate(((old, "old v003"), (new, "new motorlike"))):
        c = (80, 200, 80) if v > 0.5 else (60, 60, 160)
        cv2.rectangle(im, (90, 3 + r * 21), (w - 4, 20 + r * 21), c, -1)
        cv2.putText(im, f"{nm}", (2, 16 + r * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(im, "OPEN" if v > 0.5 else "closed", (100, 16 + r * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    x = int(90 + ap * (w - 94)); cv2.line(im, (x, 24), (x, 41), (0, 255, 255), 2)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("takes", nargs="+"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--keys", default="", help="file with take/demoNN keys to draw from (e.g. the GOLD-281 list); default = every segments.json entry")
    a = ap.parse_args(); keys = set(open(a.keys).read().split()) if a.keys else None
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    demos = []
    for t in map(pathlib.Path, a.takes):
        segs = json.loads((t / "segments.json").read_text())
        for i, s in enumerate(segs):
            if keys is not None and f"{t.name}/demo{i:02d}" not in keys: continue   # 10-31: segments.json also lists cues never performed / dropped
            t0 = (s.get("hold_pre") or s["out"])[0]; t1 = (s.get("hold_post") or s.get("back") or s["out"])[1]; demos.append((t, i, t0, t1))
    random.Random(a.seed).shuffle(demos); rows = []
    for t, i, t0, t1 in demos[:a.n]:
        V = {h: (t / d, np.load(t / d / "t_ns.npy"), lab(t / d, "umi_jaw.npz"), lab(t / d, "umi_jaw_motorlike.npz")) for h, d in DIRS.items()}
        name = f"{t.name}_demo{i:02d}.mp4"; vw = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * W, H + 44))
        prev = {h: None for h in DIRS}; nev = {h: [0, 0] for h in DIRS}
        for tt in np.arange(t0, t1, 1e9 / a.fps):
            panes, hold = [], 1
            for h in ("left", "right"):
                w, ts, old, new = V[h]; k = int(np.clip(np.searchsorted(ts, tt), 0, len(ts) - 1))
                img = cv2.imread(str(w / f"wrist_{k:05d}.jpg")); img = cv2.resize(img, (W, H)) if img is not None else np.zeros((H, W, 3), np.uint8)
                o = at(old, "state", tt) if old is not None else 0; nw = at(new, "state", tt) if new is not None else 0
                apv = at(new, "aperture", tt) if new is not None and "aperture" in new.files else 0
                if prev[h] is not None and (nw > 0.5) != (prev[h] > 0.5):
                    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 255, 0) if nw > 0.5 else (0, 0, 255), 8); hold = 4; nev[h][0] += 1
                if prev[h] is not None and (o > 0.5) != (prev[h + "_o"] > 0.5): nev[h][1] += 1
                prev[h], prev[h + "_o"] = nw, o
                cv2.putText(img, f"{h} {(tt - t0) / 1e9:5.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panes.append(np.vstack([img, strip(o, nw, float(apv))]))
            fr = np.hstack(panes)
            for _ in range(hold): vw.write(fr)
        vw.release(); rows.append(dict(video=name, take=t.name, demo=i, s=round((t1 - t0) / 1e9, 1),
                                        new_events_L=nev["left"][0], old_events_L=nev["left"][1], new_events_R=nev["right"][0], old_events_R=nev["right"][1]))
        print(rows[-1], flush=True)
    (out / "index.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
