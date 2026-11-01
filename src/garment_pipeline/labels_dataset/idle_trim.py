"""Idle head/tail trim per demo from BOTH IMUs (|gyro|) and BOTH fused TCP speeds: active = any arm moving
(|gyro| > GYR rad/s or |v| > VEL m/s, 0.3 s smoothing); trimmed window = first..last active sample with MARGIN s on each side.
Writes ~/laundry_dit_training/reports/idle_trim.csv (all 10-22 takes + the two valid 10-23 takes)."""
import csv, json, pathlib
import numpy as np

from egogarment import paths as egopaths  # noqa: E402
TAG = egopaths.TAG_RIG; DER = egopaths.DERIVED
GYR, VEL, MARGIN, GAP_MERGE = 0.35, 0.03, 0.5, 2.0
TAKES = sorted(p.name for p in TAG.glob("20261022_*") if (DER / f"{p.name}_fused/assignment.json").exists()) + ["20261023_113550", "20261023_114817"]


def sm(x, n): return np.convolve(x, np.ones(n) / n, mode="same")


rows = []
for take in TAKES:
    m = json.loads((TAG / take / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
    fused = (DER / f"{take}_fused").exists()
    src = (lambda arm, rig: DER / f"{take}_fused/umi_traj_{arm}.npz") if fused else (lambda arm, rig: TAG / take / f"umi_traj_{rig}.npz")
    rig_of = json.loads((DER / f"{take}_fused/assignment.json").read_text())["rig_of_arm"] if fused else {"left": "right", "right": "left"}
    tg_end = min(np.load(src(a, rig_of[a]))["t_ns"][-1] for a in ("left", "right"))
    for di, P in enumerate(p for p in m["protocol"] if p["text"].startswith("자유:")):
        a0 = t0 + int(P["t0"] * 1e9); a1 = t0 + int((P["t0"] + P["dur"]) * 1e9)
        if a0 >= tg_end: continue
        grid = np.arange(a0, min(a1, tg_end), int(0.02e9)); act = np.zeros(len(grid), bool)
        for arm in ("left", "right"):
            imu = np.load(TAG / take / f"imu_{rig_of[arm]}.npz"); g = np.interp(grid, imu["host_ns"], np.linalg.norm(imu["gyro"], axis=1))
            u = np.load(src(arm, rig_of[arm])); p = u["p_B"]; v = np.linalg.norm(np.gradient(p, u["t_ns"] / 1e9, axis=0), axis=1)
            act |= (sm(g, 15) > GYR) | (sm(np.interp(grid, u["t_ns"], v), 15) > VEL)
        if not act.any(): continue
        # main working block: merge active runs separated by < GAP_MERGE s, keep the block with the most active time
        # (isolated blips after the task -- putting the UMI down, re-gripping -- are idle, not demo)
        on = np.flatnonzero(act); brk = np.flatnonzero(np.diff(on) * 0.02 > GAP_MERGE)
        blocks = np.split(on, brk + 1); best = max(blocks, key=len); i0, i1 = best[0], best[-1]
        s0 = max(0.0, (grid[i0] - a0) / 1e9 - MARGIN); s1 = min((grid[-1] - a0) / 1e9, (grid[i1] - a0) / 1e9 + MARGIN)
        rows.append(dict(take=take, demo=di, demo_len_s=round((grid[-1] - a0) / 1e9, 1), keep_start_s=round(s0, 2), keep_end_s=round(s1, 2),
                         cut_head_s=round(s0, 1), cut_tail_s=round((grid[-1] - a0) / 1e9 - s1, 1), kept_s=round(s1 - s0, 1), pose_src="fused" if fused else "tag"))
out = egopaths.IDLE_TRIM_CSV
with open(out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
h = np.array([r["cut_head_s"] for r in rows]); t = np.array([r["cut_tail_s"] for r in rows]); k = np.array([r["kept_s"] for r in rows])
print(out, len(rows), "demos | head cut median", np.median(h), "max", h.max(), "| tail cut median", np.median(t), "max", t.max(), "| kept median", np.median(k))
