#!/usr/bin/env python3
"""[2026-11-05] p_grasp(t) frame-level take-CV, v2: CACHED crops (decoded once) for several jaw-ROI sizes, results saved per fold.
Same folds / labels / training recipe as grasp_ts_cv.py; frozen rule (15 Hz, W = 0.4 s restarted at jaw state changes).
ROI S px square: centered on the tip x, top at tip_y - 0.6 S (S = 256 is the current crop), resized to 224.
Reason (11-05 hard-held strips): on the 150735 right camera the bite is hidden behind the jaw pads -> does more context help?
usage: grasp_ts_cv2.py OUT_DIR S [S ...]"""
import csv, json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True); SIZES = [int(x) for x in sys.argv[2:]]
sys.argv = [sys.argv[0], str(OUT), "--none"]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_cnn_v1 as C, grasp_label_v0 as G, grasp_render_prob as R, grasp_v2_events as E2  # noqa: E402
HZ, W = 15.0, 0.4


def runs(mask):
    best = cur = 0
    for m in mask: cur = cur + 1 if m else 0; best = max(best, cur)
    return best / HZ


def main():
    gold = list(csv.DictReader(open(H / "laundry_dit_training/data/grasp_labels_v2/gold_v2.csv")))
    for g in gold: g["cls"] = R.OVR.get(g["id"], g["gold"])
    cache = OUT / f"crops_{'_'.join(map(str, SIZES))}.npz"
    if cache.exists(): Z = dict(np.load(cache, allow_pickle=True))
    else:
        cams = {}; TR, EV, OP, TS = {s: [] for s in SIZES}, {s: [] for s in SIZES}, {s: [] for s in SIZES}, []
        for n, g in enumerate(gold):
            k = (g["take"], g["cam"])
            if k not in cams:
                cd = G.TR / k[0] / k[1]; tw = np.load(cd / "t_ns.npy"); tip = G.jaw_mask_and_tip(cd, tw, np.load(cd / "umi_jaw.npz"))[1]
                cams[k] = (cd, tw, tip[0] * 1280 // G.W, tip[1] * 720 // G.H)
            cd, tw, cx, cy = cams[k]; te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
            ts = np.arange(te - 1e9, tn + 0.5e9, 1e9 / HZ).astype(np.int64); TS.append(ts)
            t_tr = [te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)) for dt in C.DTS]; t_op = [te - dt * 1e9 for dt in (0.6, 0.45, 0.3, 0.15)]
            def grab(times):
                out = {s: [] for s in SIZES}
                for t in times:
                    i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1)); im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg"))
                    im = np.zeros((720, 1280, 3), np.uint8) if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    for s in SIZES:
                        x0 = int(np.clip(cx - s // 2, 0, 1280 - s)); y0 = int(np.clip(cy - 0.6 * s, 0, 720 - s))
                        out[s].append(cv2.resize(im[y0:y0 + s, x0:x0 + s], (224, 224)))
                return {s: np.stack(v) for s, v in out.items()}
            a, b, c = grab(t_tr), grab(t_op), grab(ts)
            for s in SIZES: TR[s].append(a[s]); OP[s].append(b[s]); EV[s].append(c[s])
            if n % 50 == 0: print(f"  crops {n}/{len(gold)}", flush=True)
        Z = dict(TS=np.array(TS, dtype=object))
        for s in SIZES: Z[f"TR{s}"] = np.stack(TR[s]); Z[f"OP{s}"] = np.stack(OP[s]); Z[f"EV{s}"] = np.array(EV[s], dtype=object)
        np.savez(cache, **Z)
    tk = np.array([g["take"] for g in gold]); takes = np.unique(tk); rng = np.random.default_rng(1105); rng.shuffle(takes); folds = np.array_split(takes, 5)
    for s in SIZES:
        for fi, ft in enumerate(folds):
            f_out = OUT / f"S{s}_fold{fi}.npy"
            if f_out.exists(): continue
            tr = [i for i, g in enumerate(gold) if g["take"] not in ft and g["cls"] != "slip"]
            X = np.concatenate([Z[f"TR{s}"][tr], Z[f"OP{s}"][tr]]); y = np.concatenate([[int(gold[i]["cls"] in ("empty", "loose")) for i in tr], np.ones(len(tr), int)])
            m = C.train_fold(X, y); res = []
            for i, g in enumerate(gold):
                if g["take"] not in ft: continue
                ts = Z["TS"][i]; ev = Z[f"EV{s}"][i]
                fp = np.concatenate([1 - C.predict(m, ev[b:b + 64][:, None]) for b in range(0, len(ev), 64)])
                tj, st, evj = E2.jaw_v2(g["take"], g["cam"]); chg = np.array([t for t, _ in evj])
                last = np.array([chg[chg <= t].max() if (chg <= t).any() else ts[0] for t in ts])
                res.append(dict(id=g["id"], cls=g["cls"], te=int(g["t_close_ns"]), tn=int(g["t_open_ns"]), ts=ts, fp=fp, last=last))
            np.save(f_out, np.array(res, dtype=object), allow_pickle=True); print(f"S{s} fold {fi} done ({len(res)} events)", flush=True)
        report(s)


def report(s):
    res = [r for fi in range(5) for r in np.load(OUT / f"S{s}_fold{fi}.npy", allow_pickle=True)]
    hard = {"20261103_150735_d15_R3", "20261104_190201_d01_L3", "20261104_161134_d06_L2", "20261103_150735_d01_R1"}
    M = {c: [] for c in ("held", "empty", "loose", "slip")}; hd = []
    for r in res:
        ts, fp, last = r["ts"], r["fp"], r["last"]
        p = np.array([fp[(ts >= max(last[i], ts[i] - W * 1e9)) & (ts <= ts[i])].mean() for i in range(len(ts))])
        jd = (ts >= r["te"] + 0.3e9) & (ts < r["tn"])
        if jd.any(): M[r["cls"]].append(p[jd])
        if r["id"] in hard: hd.append(f"{r['id'][9:]} {np.median(p[jd]):.2f}")
    med = lambda c: np.median([np.median(x) for x in M[c]])
    hs = np.array([runs(x < 0.2) for x in M["held"]]); es = np.array([runs(x > 0.9) for x in M["empty"]])
    sl = sum((x < 0.5).any() for x in M["slip"])
    line = (f"S={s}px | held median {med('held'):.2f} drop any {np.mean(hs > 0):.1%} >=0.3s {np.mean(hs >= 0.3):.1%} >=0.5s {np.mean(hs >= 0.5):.1%} | "
            f"empty median {med('empty'):.2f} tail(med>0.9) {np.mean([np.median(x) > 0.9 for x in M['empty']]):.1%} >0.9 >=0.3s {np.mean(es >= 0.3):.1%} >=0.5s {np.mean(es >= 0.5):.1%} | "
            f"loose {med('loose'):.2f} | slip dropped {sl}/{len(M['slip'])} | hard held: {', '.join(hd)}")
    print(line, flush=True); open(OUT / "report.txt", "a").write(line + "\n")


if __name__ == "__main__":
    main()
