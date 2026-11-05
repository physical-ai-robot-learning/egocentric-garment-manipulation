"""11-04 user: "움직임 시작전에 잠깐 멈춰있는 구간 / 끝나고 멈춰있는 구간 trim". Per demo inside the current idle_trim.csv window:
speed = max over both arms of the 0.3 s smoothed fused Cg TCP speed (derived_cgg_1103n umi_traj p_B); head_s / tail_s = still time before the
first / after the last sample with speed > V_MOVE. Output csv feeds tighten_trim.py (which keeps MARGIN on each side)."""
import csv, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "laundry_dit_training/datasets")); sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
import tag_rig_to_lerobot as T2L
V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04; SUS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"; DER = pathlib.Path.home() / "laundry_dit_training/data" / (sys.argv[4] if len(sys.argv) > 4 else "tag_rig_derived_cgg_1103n")
trim = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(pathlib.Path.home() / "laundry_dit_training/reports/idle_trim.csv"))}
out = []
for td in sorted(DER.glob(sys.argv[5] if len(sys.argv) > 5 else "20261103_*")):
    t = TR / td.name; Z = {a: np.load(td / f"umi_traj_{a}.npz") for a in ("left", "right")}
    for d in T2L.demos_of(t):
        r = trim.get((t.name, d["idx"]))
        if r is None: continue
        w0, w1 = d["t0_ns"] + float(r["keep_start_s"]) * 1e9, d["t0_ns"] + float(r["keep_end_s"]) * 1e9
        tg = np.arange(w0, w1, 1e9 / 30); sp = np.zeros(len(tg))
        for a, z in Z.items():
            tz = z["t_ns"].astype(np.float64); ok = z["fused_valid"].astype(bool) if "fused_valid" in z.files else np.ones(len(tz), bool)
            p = np.stack([np.interp(tg, tz[ok], z["p_B"][ok, i]) for i in range(3)], 1)
            k = 9; ps = np.stack([np.convolve(np.pad(p[:, i], k // 2, mode="edge"), np.ones(k) / k, "valid") for i in range(3)], 1)
            sp = np.maximum(sp, np.r_[0, np.linalg.norm(np.diff(ps, axis=0), axis=1) * 30])
        m = sp > V; L = int(SUS * 30); sus = np.convolve(m.astype(int), np.ones(L, int), "valid") == L   # moving for >= SUS s (drops 1-frame jumps)
        st = np.flatnonzero(sus); mv = np.r_[st[:1], st[-1:] + L - 1] if len(st) else st
        h, tl = ((mv[0] / 30, (len(tg) - 1 - mv[-1]) / 30) if len(mv) else (np.nan, np.nan))
        out.append(dict(take=t.name, demo=d["idx"], kept_s=round((w1 - w0) / 1e9, 1), head_s=round(h, 2), tail_s=round(tl, 2), vmax=round(sp.max(), 3)))
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
h = np.array([o["head_s"] for o in out]); tl = np.array([o["tail_s"] for o in out])
print(len(out), "demos  head med/p90/max", np.nanmedian(h), np.nanpercentile(h, 90), np.nanmax(h), " tail", np.nanmedian(tl), np.nanpercentile(tl, 90), np.nanmax(tl))
