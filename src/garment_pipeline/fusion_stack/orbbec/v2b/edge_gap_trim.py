"""10-27 edge-gap trim (user, 122533 d8): a one-arm CoTracker invalid run (solve3 conf == 0) that touches the start or the end of a demo's
idle-trim window is cut from reports/idle_trim.csv instead of grading the demo down. Rule: run >= MIN_S, starts < EDGE_S after the window
start (cut the head to run end + 0.1 s) or ends < EDGE_S before the window end (cut the tail to run start - 0.1 s); never leaves < KEEP_MIN_S.
Only tightens; backup idle_trim.csv.bak_edge_<time> first.
    python edge_gap_trim.py <pack> [<pack> ...]"""
import csv, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
from garment_pipeline.fusion_stack import core as C  # noqa: E402
from egogarment import paths as egopaths  # noqa: E402
FD = C.FD
CSV = egopaths.IDLE_TRIM_CSV
MIN_S, EDGE_S, KEEP_MIN_S, PAD_S = 0.8, 0.5, 8.0, 0.1


def edges(pack_dir, W):
    """(cut_head_s, cut_tail_s) relative to the current window for one demo dir"""
    head = tail = 0.0
    for arm in ("left", "right"):
        z = np.load(pack_dir / f"solve3_{arm}.npz"); t = z["t"].astype(np.int64); w = (t >= W["w0"]) & (t <= W["w1"])
        if w.sum() < 2: continue
        v = z["conf"][w] > 0; tt = (t[w] - W["w0"]) / 1e9; L = (W["w1"] - W["w0"]) / 1e9; x = np.r_[0, (~v).astype(int), 0]
        for s, e in zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)):
            a, b = tt[s], tt[e - 1]
            if b - a < MIN_S: continue
            if a < EDGE_S: head = max(head, b + PAD_S)
            if b > L - EDGE_S: tail = max(tail, L - a + PAD_S)
    return head, tail


def main(packs):
    T = FD.trims(); cuts = {}
    for p in packs:
        for d in sorted(Path(p).iterdir()):
            if not (d / "solve3_left.npz").exists() or not d.name.startswith("2026"): continue
            take, demo = d.name[:15], int(d.name[-2:]); W = {w["demo"]: w for w in T[take]}[demo]
            h, t = edges(d, W); L = (W["w1"] - W["w0"]) / 1e9
            if (h > 0 or t > 0) and L - h - t >= KEEP_MIN_S: cuts[(take, demo)] = (h, t)
            elif h > 0 or t > 0: print("skip (would leave < %.0f s)" % KEEP_MIN_S, take, demo, round(h, 1), round(t, 1))
    if not cuts: print("edge-gap trim: nothing to cut"); return
    shutil.copy2(CSV, CSV.with_suffix(f".csv.bak_edge_{time.strftime('%m%d_%H%M%S')}"))
    rows = list(csv.DictReader(open(CSV))); f = list(rows[0])
    for r in rows:
        c = cuts.get((r["take"], int(r["demo"])))
        if not c: continue
        ks0, ke0, dl = float(r["keep_start_s"]), float(r["keep_end_s"]), float(r["demo_len_s"]); ks, ke = ks0 + c[0], ke0 - c[1]
        r.update(keep_start_s=round(ks, 2), keep_end_s=round(ke, 2), cut_head_s=round(ks, 1), cut_tail_s=round(dl - ke, 1), kept_s=round(ke - ks, 1))
        print("edge-gap cut", r["take"], r["demo"], "head %.1f s tail %.1f s -> %.1f s" % (c[0], c[1], ke - ks))
    with open(CSV, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=f); w.writeheader(); w.writerows(rows)
    print("edge-gap trim:", len(cuts), "demos")


if __name__ == "__main__":
    main(sys.argv[1:])
