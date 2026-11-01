#!/usr/bin/env python3
"""[2026-10-26] IK-feasibility QA of the A label (action_cartesian_robotized, robotize_ik <arm>_p/R_rob) vs the IK->FK label B
(<arm>_p/R).  NO masking is decided here: this only reports the distributions the loss-mask threshold will be chosen from.

Frames = the idle-trimmed window of each demo (reports/idle_trim.csv), per arm, 20 Hz.  Splits: grasp / free (robotize_ik
<arm>_grasp), episode class (GOLD / SILVER / REJECT from episode_quality.py), and measured / filled (derived fused_valid).
    python ik_qa_stats.py <derived_root> <quality.csv> <out_dir>"""
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd

from egogarment import paths as egopaths  # noqa: E402
TRIM = egopaths.IDLE_TRIM_CSV
POS_T, ROT_T = (10, 20, 30, 50), (10, 20, 30)
SHOULDER = {"left": np.array([0.013, 0.279, 0.132]), "right": np.array([0.020, -0.283, 0.140])}   # <arm>_link2 origin (pinocchio, base frame)
BAD_MM, BAD_DEG, TABLE_Z, REACH = 30.0, 30.0, 0.04, 0.55
# exclusive cause of a bad frame (pos > BAD_MM or rot > BAD_DEG), first match wins; only 'reach' / 'other' are mask candidates
CAUSES = ("arm-spacing", "joint-limit", "roll-relax", "flip", "workspace-table", "reach", "other")


def ang(Ra, Rb): return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ra, Rb) - 1) / 2, -1, 1)))


def frames(root, q):
    trim = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(TRIM))}; cls = {(t, int(d)): c for t, d, c in zip(q["take"], q["demo"], q["class"])}; out = []
    for f in sorted(Path(root).glob("*/ik_v1/demo*.npz")):
        take, demo = f.parents[1].name, int(f.stem[4:]); z = np.load(f, allow_pickle=True); t = z["t_ns"]; tr = trim.get((take, demo))
        if tr is None: print("no trim", take, demo); continue
        k = (t - t[0]) / 1e9; w = (k >= float(tr["keep_start_s"])) & (k <= float(tr["keep_end_s"])); c = cls.get((take, demo), "NA")
        for a in ("left", "right"):
            u = np.load(f.parents[1] / f"umi_traj_{a}.npz"); fv = u["fused_valid"][np.clip(np.searchsorted(u["t_ns"], t), 0, len(u["t_ns"]) - 1)]
            d = dict(take=take, demo=demo, arm=a, cls=c, flip=bool(z[f"{a}_flip"]),
                     grasp=z[f"{a}_grasp"][w], measured=fv[w], valid=z[f"{a}_ik_valid"][w], epos=z[f"{a}_err_pos_rob_mm"][w], erot=z[f"{a}_err_rot_rob_deg"][w],
                     lim=z[f"{a}_lim_margin_deg"][w], roll=z[f"{a}_roll_relax_deg"][w], push=z[f"{a}_gap_push_mm"][w],
                     ab_pos=np.linalg.norm(z[f"{a}_p_rob"] - z[f"{a}_p"], axis=1)[w] * 1000, lift=np.linalg.norm(z[f"{a}_p_rob"] - z[f"{a}_p_fused"], axis=1)[w] * 1000,
                     zr=z[f"{a}_p_rob"][w, 2], reach=np.linalg.norm(z[f"{a}_p_rob"][w] - SHOULDER[a], axis=1))
            d["cause"] = cause(d)
            out.append(d)
    return out


def cause(d):
    bad = (d["epos"] > BAD_MM) | (d["erot"] > BAD_DEG); c = np.full(len(bad), "", dtype=object)
    rules = [("arm-spacing", d["push"] > 0.5), ("joint-limit", d["lim"] < 1.0), ("roll-relax", d["roll"] > 0.5), ("flip", np.full(len(bad), d["flip"]) & (d["erot"] > BAD_DEG)),
             ("workspace-table", d["zr"] < TABLE_Z), ("reach", d["reach"] > REACH), ("other", np.ones(len(bad), bool))]
    for name, m in rules: c[bad & (c == "") & m] = name
    return c


def cause_table(F):
    rows = {}
    for key, sel in (("ALL", lambda d: np.ones(len(d["cause"]), bool)), ("grasp", lambda d: d["grasp"]), ("free", lambda d: ~d["grasp"]),
                     ("left", lambda d: np.full(len(d["cause"]), d["arm"] == "left")), ("right", lambda d: np.full(len(d["cause"]), d["arm"] == "right")),
                     ("GOLD", lambda d: np.full(len(d["cause"]), d["cls"] == "GOLD")), ("GOLD_grasp", lambda d: d["grasp"] & (d["cls"] == "GOLD"))):
        c = np.concatenate([d["cause"][sel(d)] for d in F]); n = len(c)
        rows[key] = {"frames": n, "bad": (c != "").mean() if n else np.nan, **{k: (c == k).mean() if n else np.nan for k in CAUSES}}
    return pd.DataFrame(rows).T


def summ(F, m=None):
    cat = lambda k: np.concatenate([d[k] if m is None else d[k][m(d)] for d in F]) if F else np.zeros(0)
    e, r = cat("epos"), cat("erot"); n = len(e)
    if not n: return dict(frames=0)
    s = dict(frames=n, ik_valid=cat("valid").mean())
    for x in POS_T: s[f"pos>{x}mm"] = (e > x).mean()
    for x in ROT_T: s[f"rot>{x}deg"] = (r > x).mean()
    s.update({"pos_p50": np.median(e), "pos_p95": np.percentile(e, 95), "rot_p50": np.median(r), "rot_p95": np.percentile(r, 95),
              "lim<1deg": (cat("lim") < 1).mean(), "lim<5deg": (cat("lim") < 5).mean(), "roll>0": (cat("roll") > 0.5).mean(), "roll_p95": np.percentile(cat("roll"), 95),
              "push>0": (cat("push") > 0.5).mean(), "push_p95": np.percentile(cat("push"), 95), "AB_pos_p50": np.median(cat("ab_pos")), "AB_pos_p95": np.percentile(cat("ab_pos"), 95),
              "lift_p95": np.percentile(cat("lift"), 95)})
    return s


def main(root, qcsv, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); q = pd.read_csv(qcsv, dtype={"take": str}); F = frames(root, q)
    rows = {"ALL": summ(F), "grasp": summ(F, lambda d: d["grasp"]), "free": summ(F, lambda d: ~d["grasp"]), "measured": summ(F, lambda d: d["measured"]),
            "filled": summ(F, lambda d: ~d["measured"]), "left": summ([d for d in F if d["arm"] == "left"]), "right": summ([d for d in F if d["arm"] == "right"]),
            "flipped": summ([d for d in F if d["flip"]]), "unflipped": summ([d for d in F if not d["flip"]])}
    for c in ("GOLD", "SILVER", "REJECT"): rows[c] = summ([d for d in F if d["cls"] == c]); rows[c + "_grasp"] = summ([d for d in F if d["cls"] == c], lambda d: d["grasp"])
    S = pd.DataFrame(rows).T; S.to_csv(out / "ik_qa_summary.csv")
    C = cause_table(F); C.to_csv(out / "ik_qa_causes.csv")
    E = pd.DataFrame([dict(take=d["take"], demo=d["demo"], arm=d["arm"], cls=d["cls"], flip=d["flip"], frames=len(d["epos"]), grasp_frac=d["grasp"].mean(),
                           ik_valid=d["valid"].mean(), pos_p95=np.percentile(d["epos"], 95), pos_gt30=(d["epos"] > 30).mean(), rot_p95=np.percentile(d["erot"], 95),
                           grasp_pos_p95=np.percentile(d["epos"][d["grasp"]], 95) if d["grasp"].any() else np.nan, lim_lt1=(d["lim"] < 1).mean(),
                           roll_frac=(d["roll"] > 0.5).mean(), push_p95=np.percentile(d["push"], 95), AB_pos_p95=np.percentile(d["ab_pos"], 95), bad=(d["cause"] != "").mean(),
                           **{f"c_{k}": (d["cause"] == k).mean() for k in CAUSES}) for d in F])
    E.to_csv(out / "ik_qa_episodes.csv", index=False)
    pd.set_option("display.width", 250); print(f"{len({(d['take'], d['demo']) for d in F})} demos, {len(F)} arm-episodes, flipped {sum(d['flip'] for d in F)}")
    print(S.round(3).to_string())
    print(f"\nbad-frame causes (pos > {BAD_MM:.0f} mm or rot > {BAD_DEG:.0f} deg; fractions of ALL frames in the split; table z < {TABLE_Z} m, reach > {REACH} m from link2)")
    print(C.round(3).to_string())
    g = E[(E.cls == "GOLD") & ((E.pos_p95 > 30) | (E.rot_p95 > 30))].sort_values("pos_p95", ascending=False)
    print(f"\nGOLD arm-episodes with large IK residual (pos p95 > 30 mm or rot p95 > 30 deg): {len(g)} / {(E.cls == 'GOLD').sum()}")
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
