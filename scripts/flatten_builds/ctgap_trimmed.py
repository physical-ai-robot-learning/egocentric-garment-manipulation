"""11-04 user: "cp gap이 조금이라도 발생하면 test나 val로, train에서는 제외". Per demo: CoTracker gap (solve2 conf == 0 or outside the tracked span)
inside the TRIMMED window (reports/idle_trim.csv), both arms -> csv take,demo,gap_frames,max_gap_s,L_gap_s,R_gap_s.   usage: ctgap_trimmed.py OUT.csv PACK [PACK..]"""
import csv, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "laundry_dit_training/datasets")); sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
import tag_rig_to_lerobot as T2L
TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
trim = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(pathlib.Path.home() / "laundry_dit_training/reports/idle_trim.csv"))}
D0 = {}; rows = []
for pk in sys.argv[2:]:
    for d in sorted(pathlib.Path(pk).iterdir()):
        if not ((d / "solve2_left.npz").exists() and (d / "solve2_right.npz").exists()): continue
        take, di = d.name.rsplit("_d", 1); di = int(di)
        if take not in D0: D0[take] = {x["idx"]: x["t0_ns"] for x in T2L.demos_of(TR / take)}
        r = trim.get((take, di))
        if r is None: continue
        w0 = D0[take][di] + float(r["keep_start_s"]) * 1e9; w1 = D0[take][di] + float(r["keep_end_s"]) * 1e9
        tt = np.load(TR / take / "t_ns.npy").astype(np.int64); tt = tt[(tt >= w0) & (tt <= w1)]; g = {}; nf = 0
        for a, A in (("left", "L"), ("right", "R")):
            z = np.load(d / f"solve2_{a}.npz"); tc = z["t"].astype(np.int64); ok = z["conf"] > 0
            gap = ~np.interp(tt, tc, ok.astype(float)).round().astype(bool) | (tt < tc[0]) | (tt > tc[-1])
            x = np.r_[0, gap.astype(int), 0]; s, e = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
            g[A] = max([(tt[min(b, len(tt) - 1)] - tt[a_]) / 1e9 if b < len(tt) else (tt[-1] - tt[a_]) / 1e9 for a_, b in zip(s, e)] + [0.0]) if gap.any() else 0.0
            nf += int(gap.sum())
        rows.append(dict(take=take, demo=di, gap_frames=nf, max_gap_s=round(max(g.values()), 2), L_gap_s=round(g["L"], 2), R_gap_s=round(g["R"], 2)))
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = len(rows); print(n, "demos;", "any gap", sum(r["gap_frames"] > 0 for r in rows), "| >0.2s", sum(r["max_gap_s"] > 0.2 for r in rows), "| >0.5s", sum(r["max_gap_s"] > 0.5 for r in rows))
