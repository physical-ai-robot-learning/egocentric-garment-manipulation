"""10-25 episode-level quality table (whole episodes are kept or dropped, never cut) from the gated CoTracker production run
(solve3_<arm>.npz, conf > 0 = valid) of every pack demo, per arm, inside the idle-trim window:
  cov        valid fraction                         gap_s      longest invalid run [s]
  far10      valid frames > 10 cm from the >= 1-tag body point (fixed rig offset)   burst_s  longest > 10 cm run [s]
  swap_s     frames removed by the L/R-collision gate [s]     seed_ok   first 1 s of the RAW track within 10 cm of the tags (or no tags)
  grasp_ok   no invalid run >= 1.0 s (> fusion bridging range) overlapping +-1 s of a jaw open/close event of that arm's wrist camera
class  GOLD    both arms: cov >= 0.95, gap <= 1.0 s, far10 <= 0.05, burst <= 0.5 s, swap 0, seed_ok, grasp_ok
       REJECT  any arm: gap > 3 s, cov < 0.80, far10 > 0.20, burst > 2 s or swap > 1 s
       SILVER  otherwise
in_130 = the episode is in the v31speed fullmask set (Desktop/orbbec_v31speed_QA_all/episode_stats.csv).
    python episode_quality.py <out.csv> <pack> [<pack> ...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
from garment_pipeline.fusion_stack import core as C  # noqa
from egogarment import paths as egopaths  # noqa: E402
FD = C.FD
STATS = egopaths.DESKTOP / "orbbec_v31speed_QA_all/episode_stats.csv"


def runs_s(b, t, dt):
    x = np.r_[0, b.astype(int), 0]; s, e = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    return [(t[e_ - 1] - t[s_]) / 1e9 + dt for s_, e_ in zip(s, e)]


def grasp_ok(inv, near, t, dt, min_s=1.0):
    """False if an invalid run >= min_s overlaps the +-1 s window of a jaw event (short gaps are bridged by the fusion)"""
    x = np.r_[0, inv.astype(int), 0]
    for s_, e_ in zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)):
        if (t[e_ - 1] - t[s_]) / 1e9 + dt >= min_s and near[s_:e_].any(): return False
    return True


def arm_row(d, take, W, arm, asg):
    z = np.load(d / f"solve3_{arm}.npz"); r = np.load(d / f"solve2_{arm}.npz"); t = z["t"].astype(np.int64); dt = float(np.median(np.diff(t))) / 1e9
    w = (t >= W["w0"]) & (t <= W["w1"]); tw = t[w]; v = (z["conf"] > 0)[w]; c = z["c"][w]; gt = z["gt"][w]
    D = FD.load_arm(take, arm, with_slam=False); k = np.clip(np.searchsorted(D["t"], tw), 0, len(D["t"]) - 1); tag = (D["ntags"][k] >= 1) & (np.abs(D["t"][k] - tw) < 3e7)
    e = np.linalg.norm(c - gt, axis=1); m = v & tag & np.isfinite(e); far = m & (e > 0.10)
    cr = np.where((r["conf"] > 0)[:, None], r["c"], np.nan)[w]; first = np.flatnonzero(np.isfinite(cr[:, 0]))
    s0 = first[(tw[first] - tw[first[0]]) < 1e9] if len(first) else first; es = np.linalg.norm(cr[s0] - gt[s0], axis=1)[tag[s0]] if len(s0) else np.zeros(0)
    J = np.load(FD.TAG / take / asg["cam_of_arm"][arm] / "umi_jaw.npz"); je = J["t_ns"][1:][np.diff(J["state"].astype(int)) != 0]
    je = je[(je >= W["w0"]) & (je <= W["w1"])]; near = np.zeros(len(tw), bool)
    for x in je: near |= np.abs(tw - x) <= 1e9
    return dict(cov=float(v.mean()), gap_s=max(runs_s(~v, tw, dt) or [0.0]), far10=float(far.sum() / max(m.sum(), 1)), burst_s=max(runs_s(far, tw, dt) or [0.0]),
                swap_s=float(z["gate_lr"][w].sum() * dt), seed_ok=bool(len(es) == 0 or np.median(es) < 0.10), grasp_ok=grasp_ok(~v, near, tw, dt),
                n_tag=int(m.sum()), n_grasp_ev=len(je))


def main(out, packs):
    alt = egopaths.REPORTS / "orbbec_v31speed_episode_stats.csv"; src = STATS if STATS.exists() else alt
    st = pd.read_csv(src, dtype={"take": str}) if src.exists() else pd.DataFrame(columns=["take", "demo"]); in130 = set(zip(st["take"], st["demo"].astype(int))); rows = []
    for p in packs:
        for d in sorted(Path(p).iterdir()):
            if not (d / "solve3_left.npz").exists(): continue
            meta = json.loads((d / "meta.json").read_text()); take, demo = meta["take"], int(meta["demo"]); W = {w["demo"]: w for w in FD.trims()[take]}[demo]
            asg = json.loads((egopaths.DERIVED / take / "assignment.json").read_text())
            row = dict(take=take, demo=demo, in_130=(take, demo) in in130, dur_s=round((W["w1"] - W["w0"]) / 1e9, 1))
            A = {a: arm_row(d, take, W, a, asg) for a in ("left", "right")}
            for a in A:
                for k_, v_ in A[a].items(): row[f"{a[0].upper()}_{k_}"] = v_
            gold = all(A[a]["cov"] >= 0.95 and A[a]["gap_s"] <= 1.0 and A[a]["far10"] <= 0.05 and A[a]["burst_s"] <= 0.5 and A[a]["swap_s"] == 0 and A[a]["seed_ok"] and A[a]["grasp_ok"] for a in A)
            rej = any(A[a]["gap_s"] > 3 or A[a]["cov"] < 0.80 or A[a]["far10"] > 0.20 or A[a]["burst_s"] > 2 or A[a]["swap_s"] > 1 for a in A)
            row["class"] = "GOLD" if gold else ("REJECT" if rej else "SILVER")
            why = []
            for a in A:
                x = A[a]; L = a[0].upper()
                for cond, txt in ((x["cov"] < 0.95, f"{L} cov {x['cov']:.2f}"), (x["gap_s"] > 1, f"{L} gap {x['gap_s']:.1f}s"), (x["far10"] > 0.05, f"{L} far10 {x['far10']:.2f}"),
                                  (x["burst_s"] > 0.5, f"{L} burst {x['burst_s']:.1f}s"), (x["swap_s"] > 0, f"{L} swap {x['swap_s']:.1f}s"), (not x["seed_ok"], f"{L} seed"), (not x["grasp_ok"], f"{L} grasp-gap")):
                    if cond: why.append(txt)
            row["why"] = "; ".join(why); rows.append(row)
    df = pd.DataFrame(rows).sort_values(["take", "demo"]); df.to_csv(out, index=False)
    print(df["class"].value_counts().to_dict(), "| in_130:", df[df["in_130"]]["class"].value_counts().to_dict()); return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
