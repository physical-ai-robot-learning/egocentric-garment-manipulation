#!/usr/bin/env python3
"""[2026-11-05] Take-grouped 5-fold CV of the CAUSAL frame-level p_grasp(t) (the conditioning signal), causal window sweep.
Per fold: the grasp_render_prob recipe (v2 gold closes x4 frames + open-approach negatives; slip excluded, loose = not held) on the train
takes; held-out events scored per frame at 15 Hz from close - 1 s to open + 0.5 s; p(t) = mean per-frame P(grasp) over
[max(last jaw state change, t - W), t] for W in WINDOWS.  "judged" = jaw closed and t >= close + 0.3 s.
Metrics (per gold class, event level): held median / false-drop (any judged tick < 0.2); empty median / tail (event median > 0.9, any > 0.9);
loose median; slip: p < 0.5 before the jaw opens + lead time; open approach (close - 1 s .. close) median / share > 0.5.
usage: grasp_ts_cv.py OUT_DIR [report]"""
import csv, json, pathlib, sys
import numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True); MODE = sys.argv[2] if len(sys.argv) > 2 else ""
sys.argv = [sys.argv[0], str(OUT), "--none"]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_cnn_v1 as C, grasp_render_prob as R, grasp_v2_events as E2  # noqa: E402
HZ, WARM_S, WINDOWS = 15.0, 0.3, (0.2, 0.3, 0.4, 0.6)


def main():
    gold = list(csv.DictReader(open(H / "laundry_dit_training/data/grasp_labels_v2/gold_v2.csv")))
    for g in gold: g["cls"] = R.OVR.get(g["id"], g["gold"])
    fns, samples = {}, []
    for g in gold:
        k = (g["take"], g["cam"])
        if k not in fns: fns[k] = R.crop_fn(*k)[0]
    tk = np.array([g["take"] for g in gold]); takes = np.unique(tk); rng = np.random.default_rng(1105); rng.shuffle(takes)
    res = []
    for fi, ft in enumerate(np.array_split(takes, 5)):
        X, y = [], []
        for g in gold:
            if g["take"] in ft or g["cls"] == "slip": continue
            f = fns[(g["take"], g["cam"])]; te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
            X.append(np.stack([f(te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9))) for dt in C.DTS])); y.append(int(g["cls"] in ("empty", "loose")))
            X.append(np.stack([f(te - dt * 1e9) for dt in (0.6, 0.45, 0.3, 0.15)])); y.append(1)
        m = C.train_fold(np.stack(X), np.array(y)); print(f"fold {fi}: trained on {len(y)}", flush=True)
        for g in gold:
            if g["take"] not in ft: continue
            f = fns[(g["take"], g["cam"])]; te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
            ts = np.arange(te - 1e9, tn + 0.5e9, 1e9 / HZ).astype(np.int64)
            fp = np.concatenate([1 - C.predict(m, np.stack([f(t) for t in ts[b:b + 64]])[:, None]) for b in range(0, len(ts), 64)])
            tj, st, ev = E2.jaw_v2(g["take"], g["cam"]); chg = np.array([t for t, _ in ev])
            last = np.array([chg[chg <= t].max() if (chg <= t).any() else ts[0] for t in ts])
            res.append(dict(id=g["id"], cls=g["cls"], te=te, tn=tn, ts=ts, fp=fp, last=last))
    np.save(OUT / "ts_cv_raw.npy", np.array(res, dtype=object), allow_pickle=True)
    report(res)


def report(res):
    lines = []
    for W in WINDOWS:
        S = {c: [] for c in ("held", "empty", "loose", "slip")}; appr = []
        for r in res:
            ts, fp, last = r["ts"], r["fp"], r["last"]
            p = np.array([fp[(ts >= max(last[i], ts[i] - W * 1e9)) & (ts <= ts[i])].mean() for i in range(len(ts))])
            jd = (ts >= r["te"] + WARM_S * 1e9) & (ts < r["tn"]); ap = (ts >= r["te"] - 1e9) & (ts < r["te"])
            if jd.any(): S[r["cls"]].append((p[jd], ts[jd], r["tn"]))
            if ap.any(): appr.append(p[ap])
        med = lambda c: np.median([np.median(x[0]) for x in S[c]]) if S[c] else float("nan")
        h, e = S["held"], S["empty"]
        fd = np.mean([x[0].min() < 0.2 for x in h]); fdt = np.mean(np.concatenate([x[0] < 0.2 for x in h]))
        tail = np.mean([np.median(x[0]) > 0.9 for x in e]); tail_any = np.mean([x[0].max() > 0.9 for x in e])
        sl = S["slip"]; drop = [x for x in sl if (x[0] < 0.5).any()]
        lead = [(x[2] - x[1][np.argmax(x[0] < 0.5)]) / 1e9 for x in drop]; hi = np.mean([x[0].max() > 0.9 for x in sl]) if sl else float("nan")
        ap = np.concatenate(appr)
        lines.append(f"W={W:.1f}s | held n{len(h)} median {med('held'):.2f} false-drop(any<0.2) {fd:.0%} ticks<0.2 {fdt:.1%} | empty n{len(e)} median {med('empty'):.2f} "
                     f"tail(median>0.9) {tail:.0%} any>0.9 {tail_any:.0%} | loose n{len(S['loose'])} median {med('loose'):.2f} | slip n{len(sl)} reached>0.9 {hi:.0%} "
                     f"dropped<0.5 before open {len(drop)}/{len(sl)} lead median {np.median(lead) if lead else float('nan'):.2f}s | open approach median {np.median(ap):.2f} >0.5 {np.mean(ap > 0.5):.0%}")
    txt = "\n".join(lines); print(txt, flush=True); (OUT / "ts_cv_report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    if MODE == "report": report(list(np.load(OUT / "ts_cv_raw.npy", allow_pickle=True)))
    else: main()
