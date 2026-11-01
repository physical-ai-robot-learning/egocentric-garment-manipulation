"""10-26 prep for takes recorded after the camera was raised (no fused/SLAM, tag-only), without touching existing rows / files:
  1. idle trim  -- same rule as laundry_dit_training/datasets/idle_trim.py (both IMUs |gyro| > GYR or both tag-TCP speeds > VEL, 0.3 s
                   smoothing, MARGIN on each side), APPENDED to reports/idle_trim.csv for takes not already in it (backup written first)
  2. start seed -- orbbec/v2b/start_seed_<take>.json: per ARM the rig body (tag TCP + R o_rig) median over the first 1 s of tag frames of
                   demo 1 (the fixed start spots moved with the 70 mm tags on 10-26; track_v31 seeds / workspace ROI use these)
    python new_take_prep.py <take> [<take> ...]"""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TAG = Path.home() / "ego_collector/datasets/tag_rig"; DER = Path.home() / "laundry_dit_training/data/tag_rig_derived"; CSV = Path.home() / "laundry_dit_training/reports/idle_trim.csv"
GYR, VEL, MARGIN = 0.35, 0.03, 0.5
OFF = json.loads((HERE / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]


def sm(x, n): return np.convolve(x, np.ones(n) / n, mode="same")


def main(takes):
    have = {r["take"] for r in csv.DictReader(open(CSV))}; shutil.copy2(CSV, CSV.with_suffix(".csv.bak_1026")); rows = []
    for take in takes:
        m = json.loads((TAG / take / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"]); rig_of = json.loads((DER / take / "assignment.json").read_text())["rig_of_arm"]
        U = {a: np.load(TAG / take / f"umi_traj_{rig_of[a]}.npz") for a in ("left", "right")}; end = min(U[a]["t_ns"][-1] for a in U); seeds = {}
        for di, P in enumerate(p for p in m["protocol"] if p["text"].startswith("자유:")):
            a0 = t0 + int(P["t0"] * 1e9); a1 = t0 + int((P["t0"] + P["dur"]) * 1e9)
            if a0 >= end: continue
            g = np.arange(a0, min(a1, end), int(0.02e9)); act = np.zeros(len(g), bool)
            for a in ("left", "right"):
                imu = np.load(TAG / take / f"imu_{rig_of[a]}.npz"); gy = np.interp(g, imu["host_ns"], np.linalg.norm(imu["gyro"], axis=1))
                u = U[a]; v = np.linalg.norm(np.gradient(u["p_B"], u["t_ns"] / 1e9, axis=0), axis=1)
                act |= (sm(gy, 15) > GYR) | (sm(np.interp(g, u["t_ns"], v), 15) > VEL)
                if di == 0:
                    k = np.flatnonzero((u["t_ns"] >= a0) & (u["ntags"] > 0))[:15]
                    body = u["p_B"][k] + np.einsum("nij,j->ni", u["R_B_tcp"][k], np.array(OFF[rig_of[a]]) / 1000); seeds[a] = np.median(body, 0).round(3).tolist()
            if not act.any(): continue
            idx = np.flatnonzero(act); ks = max(0.0, (g[idx[0]] - a0) / 1e9 - MARGIN); ke = min((g[-1] - a0) / 1e9, (g[idx[-1]] - a0) / 1e9 + MARGIN); dl = (a1 - a0) / 1e9
            if take not in have:
                rows.append(dict(take=take, demo=di, demo_len_s=round(dl, 1), keep_start_s=round(ks, 2), keep_end_s=round(ke, 2), cut_head_s=round(ks, 1),
                                 cut_tail_s=round(dl - ke, 1), kept_s=round(ke - ks, 1), pose_src="tag_only"))
        (HERE / f"start_seed_{take}.json").write_text(json.dumps(seeds, indent=1)); print(take, "seeds", seeds)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["take", "demo", "demo_len_s", "keep_start_s", "keep_end_s", "cut_head_s", "cut_tail_s", "kept_s", "pose_src"]); [w.writerow(r) for r in rows]
    for r in rows: print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
