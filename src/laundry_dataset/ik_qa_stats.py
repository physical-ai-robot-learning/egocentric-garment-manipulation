#!/usr/bin/env python3
"""[2026-10-26] IK-feasibility QA of the A label (action_cartesian_robotized, robotize_ik <arm>_p/R_rob) vs the IK->FK label B
(<arm>_p/R) (dataset stage, QA). NO masking is decided here: this only reports the distributions the loss-mask threshold will be
chosen from.

Frames = the idle-trimmed window of each demo (reports/idle_trim.csv), per arm, 20 Hz. Splits: grasp / free (robotize_ik
<arm>_grasp), episode class (GOLD / SILVER / REJECT from episode_quality.py), and measured / filled (derived fused_valid).
Writes <out_dir>/ik_qa_summary.csv, ik_qa_causes.csv, ik_qa_episodes.csv and prints them.
    python -m laundry_dataset.ik_qa_stats <derived_root> <quality.csv> <out_dir>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from laundry_common import paths as egopaths

from .takes import loadIdleTrim

TRIM_CSV = egopaths.IDLE_TRIM_CSV
POS_T, ROT_T = (10, 20, 30, 50), (10, 20, 30)
# <arm>_link2 origin (pinocchio, base frame)
SHOULDER = {"left": np.array([0.013, 0.279, 0.132]), "right": np.array([0.020, -0.283, 0.140])}
BAD_MM, BAD_DEG, TABLE_Z, REACH = 30.0, 30.0, 0.04, 0.55
# exclusive cause of a bad frame (pos > BAD_MM or rot > BAD_DEG), first match wins; only 'reach' / 'other' are mask candidates
CAUSES = ("arm-spacing", "joint-limit", "roll-relax", "flip", "workspace-table", "reach", "other")
PUSH_ACTIVE_MM = 0.5                 # an arm-gap push / wrist-roll yield above this counts as active
ROLL_ACTIVE_DEG = 0.5
LIMIT_NEAR_DEG, LIMIT_CLOSE_DEG = 1.0, 5.0
P95 = 95
CLASSES = ("GOLD", "SILVER", "REJECT")
DISPLAY_WIDTH = 250
DEMO_PREFIX = "demo"                # ik_v1/demoNN.npz
NS_PER_S = 1e9
MM_PER_M = 1000


def ang(Ra, Rb):
    """rotation angle (deg) between two stacks of rotations"""
    return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ra, Rb) - 1) / 2, -1, 1)))


def frames(root, quality) -> list:
    """per demo and arm: the trimmed-window IK QA columns of ik_v1 + its class / measured flag / bad-frame cause"""
    tTrim = loadIdleTrim(TRIM_CSV)
    tCls = {(tT, int(tD)): tC for tT, tD, tC in zip(quality["take"], quality["demo"], quality["class"])}
    tOut = []
    for tF in sorted(Path(root).glob("*/ik_v1/demo*.npz")):
        tTake, tDemo = tF.parents[1].name, int(tF.stem[len(DEMO_PREFIX):])
        tZ = np.load(tF, allow_pickle=True)
        tT = tZ["t_ns"]
        tRow = tTrim.get((tTake, tDemo))
        if tRow is None:
            print("no trim", tTake, tDemo)
            continue
        tK = (tT - tT[0]) / NS_PER_S
        tW = (tK >= float(tRow["keep_start_s"])) & (tK <= float(tRow["keep_end_s"]))
        tC = tCls.get((tTake, tDemo), "NA")
        for tA in ("left", "right"):
            tU = np.load(tF.parents[1] / f"umi_traj_{tA}.npz")
            tFv = tU["fused_valid"][np.clip(np.searchsorted(tU["t_ns"], tT), 0, len(tU["t_ns"]) - 1)]
            tD = dict(take=tTake, demo=tDemo, arm=tA, cls=tC, flip=bool(tZ[f"{tA}_flip"]),
                      grasp=tZ[f"{tA}_grasp"][tW], measured=tFv[tW], valid=tZ[f"{tA}_ik_valid"][tW],
                      epos=tZ[f"{tA}_err_pos_rob_mm"][tW], erot=tZ[f"{tA}_err_rot_rob_deg"][tW],
                      lim=tZ[f"{tA}_lim_margin_deg"][tW], roll=tZ[f"{tA}_roll_relax_deg"][tW],
                      push=tZ[f"{tA}_gap_push_mm"][tW],
                      ab_pos=np.linalg.norm(tZ[f"{tA}_p_rob"] - tZ[f"{tA}_p"], axis=1)[tW] * MM_PER_M,
                      lift=np.linalg.norm(tZ[f"{tA}_p_rob"] - tZ[f"{tA}_p_fused"], axis=1)[tW] * MM_PER_M,
                      zr=tZ[f"{tA}_p_rob"][tW, 2], reach=np.linalg.norm(tZ[f"{tA}_p_rob"][tW] - SHOULDER[tA], axis=1))
            tD["cause"] = cause(tD)
            tOut.append(tD)
    return tOut


def cause(d):
    """exclusive cause per bad frame (first matching rule), "" for good frames"""
    tBad = (d["epos"] > BAD_MM) | (d["erot"] > BAD_DEG)
    tC = np.full(len(tBad), "", dtype=object)
    tRules = [("arm-spacing", d["push"] > PUSH_ACTIVE_MM), ("joint-limit", d["lim"] < LIMIT_NEAR_DEG),
              ("roll-relax", d["roll"] > ROLL_ACTIVE_DEG), ("flip", np.full(len(tBad), d["flip"]) & (d["erot"] > BAD_DEG)),
              ("workspace-table", d["zr"] < TABLE_Z), ("reach", d["reach"] > REACH), ("other", np.ones(len(tBad), bool))]
    for tName, tM in tRules:
        tC[tBad & (tC == "") & tM] = tName
    return tC


def causeTable(F):
    tRows = {}
    tSel = (("ALL", lambda d: np.ones(len(d["cause"]), bool)), ("grasp", lambda d: d["grasp"]), ("free", lambda d: ~d["grasp"]),
            ("left", lambda d: np.full(len(d["cause"]), d["arm"] == "left")),
            ("right", lambda d: np.full(len(d["cause"]), d["arm"] == "right")),
            ("GOLD", lambda d: np.full(len(d["cause"]), d["cls"] == "GOLD")),
            ("GOLD_grasp", lambda d: d["grasp"] & (d["cls"] == "GOLD")))
    for tKey, tSelect in tSel:
        tC = np.concatenate([tD["cause"][tSelect(tD)] for tD in F])
        tN = len(tC)
        tRows[tKey] = {"frames": tN, "bad": (tC != "").mean() if tN else np.nan,
                       **{tK: (tC == tK).mean() if tN else np.nan for tK in CAUSES}}
    return pd.DataFrame(tRows).T


def summ(F, m=None):
    """distribution summary of the IK residuals over the frames of F (selected by m per arm-episode)"""
    tCat = lambda k: np.concatenate([tD[k] if m is None else tD[k][m(tD)] for tD in F]) if F else np.zeros(0)
    tE, tR = tCat("epos"), tCat("erot")
    tN = len(tE)
    if not tN:
        return dict(frames=0)
    tS = dict(frames=tN, ik_valid=tCat("valid").mean())
    for tX in POS_T:
        tS[f"pos>{tX}mm"] = (tE > tX).mean()
    for tX in ROT_T:
        tS[f"rot>{tX}deg"] = (tR > tX).mean()
    tS.update({"pos_p50": np.median(tE), "pos_p95": np.percentile(tE, P95), "rot_p50": np.median(tR),
               "rot_p95": np.percentile(tR, P95),
               "lim<1deg": (tCat("lim") < LIMIT_NEAR_DEG).mean(), "lim<5deg": (tCat("lim") < LIMIT_CLOSE_DEG).mean(),
               "roll>0": (tCat("roll") > ROLL_ACTIVE_DEG).mean(),
               "roll_p95": np.percentile(tCat("roll"), P95), "push>0": (tCat("push") > PUSH_ACTIVE_MM).mean(),
               "push_p95": np.percentile(tCat("push"), P95),
               "AB_pos_p50": np.median(tCat("ab_pos")), "AB_pos_p95": np.percentile(tCat("ab_pos"), P95),
               "lift_p95": np.percentile(tCat("lift"), P95)})
    return tS


def _episodeRow(d):
    return dict(take=d["take"], demo=d["demo"], arm=d["arm"], cls=d["cls"], flip=d["flip"], frames=len(d["epos"]),
                grasp_frac=d["grasp"].mean(),
                ik_valid=d["valid"].mean(), pos_p95=np.percentile(d["epos"], P95), pos_gt30=(d["epos"] > BAD_MM).mean(),
                rot_p95=np.percentile(d["erot"], P95),
                grasp_pos_p95=np.percentile(d["epos"][d["grasp"]], P95) if d["grasp"].any() else np.nan,
                lim_lt1=(d["lim"] < LIMIT_NEAR_DEG).mean(),
                roll_frac=(d["roll"] > ROLL_ACTIVE_DEG).mean(), push_p95=np.percentile(d["push"], P95),
                AB_pos_p95=np.percentile(d["ab_pos"], P95),
                bad=(d["cause"] != "").mean(), **{f"c_{tK}": (d["cause"] == tK).mean() for tK in CAUSES})


def run(root, qualityCsv, outDir) -> None:
    """the three QA tables of a derived root -> outDir + printed summary (was main(root, qcsv, out))"""
    tOut = Path(outDir)
    tOut.mkdir(parents=True, exist_ok=True)
    tQ = pd.read_csv(qualityCsv, dtype={"take": str})
    tF = frames(root, tQ)
    tRows = {"ALL": summ(tF), "grasp": summ(tF, lambda d: d["grasp"]), "free": summ(tF, lambda d: ~d["grasp"]),
             "measured": summ(tF, lambda d: d["measured"]), "filled": summ(tF, lambda d: ~d["measured"]),
             "left": summ([tD for tD in tF if tD["arm"] == "left"]), "right": summ([tD for tD in tF if tD["arm"] == "right"]),
             "flipped": summ([tD for tD in tF if tD["flip"]]),
             "unflipped": summ([tD for tD in tF if not tD["flip"]])}
    for tClass in CLASSES:
        tRows[tClass] = summ([tD for tD in tF if tD["cls"] == tClass])
        tRows[tClass + "_grasp"] = summ([tD for tD in tF if tD["cls"] == tClass], lambda d: d["grasp"])
    tS = pd.DataFrame(tRows).T
    tS.to_csv(tOut / "ik_qa_summary.csv")
    tC = causeTable(tF)
    tC.to_csv(tOut / "ik_qa_causes.csv")
    tE = pd.DataFrame([_episodeRow(tD) for tD in tF])
    tE.to_csv(tOut / "ik_qa_episodes.csv", index=False)
    pd.set_option("display.width", DISPLAY_WIDTH)
    print(f"{len({(tD['take'], tD['demo']) for tD in tF})} demos, {len(tF)} arm-episodes, flipped {sum(tD['flip'] for tD in tF)}")
    print(tS.round(3).to_string())
    print(f"\nbad-frame causes (pos > {BAD_MM:.0f} mm or rot > {BAD_DEG:.0f} deg; fractions of ALL frames in the split; "
          f"table z < {TABLE_Z} m, reach > {REACH} m from link2)")
    print(tC.round(3).to_string())
    tG = tE[(tE.cls == "GOLD") & ((tE.pos_p95 > BAD_MM) | (tE.rot_p95 > BAD_DEG))].sort_values("pos_p95", ascending=False)
    print(f"\nGOLD arm-episodes with large IK residual (pos p95 > 30 mm or rot p95 > 30 deg): "
          f"{len(tG)} / {(tE.cls == 'GOLD').sum()}")
    print(tG.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("derived_root")
    tAp.add_argument("quality_csv")
    tAp.add_argument("out_dir")
    tA = tAp.parse_args(argv)
    run(tA.derived_root, tA.quality_csv, tA.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
