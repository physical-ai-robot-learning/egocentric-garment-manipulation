"""Out-of-view audit per episode (both arms), added to the episode quality table (was orbbec/v2b/cotrack/out_of_view.py, 10-25):
  <A>_exit_n / <A>_exit_s   CoTracker gaps >= MIN_GAP_S whose last valid body position projects within EDGE_PX of the image border
                            (the hand left the Orbbec view), count / total seconds, with the edge of the longest one
  <A>_outproj_s             seconds the fused rig BODY (fus_Cg_g production TCP + R o_rig) projects outside the image (no margin)
  out_of_view               any arm with exit_s > 0 or outproj_s >= MIN_GAP_S
Then the in-view episodes keep their GOLD / SILVER / REJECT class; folders of hard links to the existing renders:
  <out>/{GOLD,SILVER,REJECT,OUT_OF_VIEW}/<take>_dNN.mp4  + episode_quality_inview.csv and per-class lists (<out> must exist).
Inputs: the episode_quality.py table, pack solve3 files, FUSION_WORK/stitch/qa/fus_Cg_g production runs, per-take calibration.
    python -m laundry_tracking.cotrack.out_of_view <quality.csv> <renders dir> <out dir> <pack> [<pack> ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from laundry_common import paths

from .. import workdirs
from ..camera_calibration import cameraFromBase
from ..constants import NS_PER_S
from ..fgo import arm_data
from ..intervals import trueRuns

EDGE_PX = 40                       # was EDGE
MIN_GAP_S = 0.5
IMAGE_W, IMAGE_H = 848, 480        # Orbbec colour stream
PRODUCTION_VARIANT = "fus_Cg_g"
PRODUCTION_RUN = -1
ARMS = ("left", "right")
CLASSES = ("GOLD", "SILVER", "REJECT")
OUT_OF_VIEW = "OUT_OF_VIEW"


def project(P: np.ndarray, intr: dict, tcb: np.ndarray) -> np.ndarray:
    """base-frame points -> pixels (was out_of_view.proj with the module-level T_C_B)"""
    tX = P @ tcb[:3, :3].T + tcb[:3, 3]
    return np.stack([intr["fx"] * tX[:, 0] / tX[:, 2] + intr["cx"], intr["fy"] * tX[:, 1] / tX[:, 2] + intr["cy"]], 1)


def edgeOf(u: float, v: float) -> str:
    """image border a pixel is near ("" = inside) (was out_of_view.edge_of)"""
    return "top" if v < EDGE_PX else "bottom" if v > IMAGE_H - EDGE_PX else "left" if u < EDGE_PX else "right" if u > IMAGE_W - EDGE_PX else ""


def armAudit(demoDir: Path, take: str, window: arm_data.DemoWindow, arm: str, intr: dict, tcb: np.ndarray,
             offsetM: np.ndarray) -> dict:
    """exit gaps and out-of-image seconds of one arm (was out_of_view.arm_audit)"""
    tZ = np.load(demoDir / f"solve3_{arm}.npz")
    tT = tZ["t"].astype(np.int64)
    tIn = (tT >= window.mW0) & (tT <= window.mW1)
    tValid = tZ["conf"] > 0
    tC = tZ["c"]
    tIdx = np.flatnonzero(tIn)
    tN, tTot, tEdges = 0, 0.0, []
    for tS, tE in trueRuns(~tValid[tIdx]):
        tL = (tT[tIdx[tE - 1]] - tT[tIdx[tS]]) / NS_PER_S
        if tL < MIN_GAP_S:
            continue
        tPrev = np.flatnonzero(tValid[:tIdx[tS]])
        if not len(tPrev):
            continue
        tU, tV = project(tC[tPrev[-1]][None], intr, tcb)[0]
        tEdge = edgeOf(tU, tV)
        if tEdge:
            tN += 1
            tTot += tL
            tEdges.append((tL, tEdge))
    tRun = workdirs.qaRunPath(PRODUCTION_VARIANT, take, arm, window.mDemo, PRODUCTION_RUN)
    tOutS = 0.0
    if tRun.exists():
        tR = np.load(tRun)
        tM = (tR["t_ns"] >= window.mW0) & (tR["t_ns"] <= window.mW1)
        if tM.sum() > 1:
            tBody = tR["p_B"][tM] + np.einsum("nij,j->ni", tR["R_B_tcp"][tM], offsetM)      # rig body = what the Orbbec sees
            tUv = project(tBody, intr, tcb)
            tOut = (tUv[:, 0] < 0) | (tUv[:, 0] > IMAGE_W) | (tUv[:, 1] < 0) | (tUv[:, 1] > IMAGE_H)
            tOutS = float(tOut.sum() * np.median(np.diff(tR["t_ns"][tM])) / NS_PER_S)
    return dict(exit_n=tN, exit_s=round(tTot, 2), exit_edge=max(tEdges)[1] if tEdges else "", outproj_s=round(tOutS, 2))


def auditEpisodes(qualityCsv: str, renders: str, out: str, packs: list[str]) -> None:
    tQuality = pd.read_csv(qualityCsv, dtype={"take": str})
    tRows = []
    tLoc = {}
    for tPack in packs:
        for tDir in Path(tPack).iterdir():
            if (tDir / "solve3_left.npz").exists():
                tLoc[tDir.name] = tDir
    tWindows = arm_data.loadTrimWindows()
    for _, tR in tQuality.iterrows():
        tTcb = cameraFromBase(tR["take"])
        tKey = f"{tR['take']}_d{int(tR['demo']):02d}"
        tDir = tLoc[tKey]
        tWin = {tW.mDemo: tW for tW in tWindows[tR["take"]]}[int(tR["demo"])]
        tIntr = json.loads((paths.TAG_RIG / tR["take"] / "meta.json").read_text())["intrinsics"]
        tRow = dict(tR)
        tAsg = json.loads((paths.DERIVED / tR["take"] / "assignment.json").read_text())
        for tA in ARMS:
            for tK, tV in armAudit(tDir, tR["take"], tWin, tA, tIntr, tTcb, workdirs.bodyOffsetM(tAsg["rig_of_arm"][tA])).items():
                tRow[f"{tA[0].upper()}_{tK}"] = tV
        tRow["out_of_view"] = bool(any(tRow[f"{tL}_exit_s"] > 0 or tRow[f"{tL}_outproj_s"] >= MIN_GAP_S for tL in "LR"))
        tRows.append(tRow)
    tDf = pd.DataFrame(tRows)
    tDf.to_csv(Path(out) / "episode_quality_inview.csv", index=False)
    for tC in CLASSES + (OUT_OF_VIEW,):
        (Path(out) / tC).mkdir(parents=True, exist_ok=True)
    for _, tR in tDf.iterrows():
        tKey = f"{tR['take']}_d{int(tR['demo']):02d}.mp4"
        tSrc = Path(renders) / tR["class"] / tKey
        tDst = Path(out) / (OUT_OF_VIEW if tR["out_of_view"] else tR["class"]) / tKey
        if tSrc.exists() and not tDst.exists():
            os.link(tSrc, tDst)
    for tC in CLASSES:
        tDf[(~tDf["out_of_view"]) & (tDf["class"] == tC)].to_csv(Path(out) / tC / f"{tC.lower()}_list.csv", index=False)
    tDf[tDf["out_of_view"]].to_csv(Path(out) / OUT_OF_VIEW / "out_of_view_list.csv", index=False)
    tIn130 = tDf[tDf["in_130"]]
    print("all:", tDf.groupby(["out_of_view", "class"]).size().unstack(fill_value=0).to_string())
    print("in_130:", tIn130.groupby(["out_of_view", "class"]).size().unstack(fill_value=0).to_string())
    tOov = tDf[tDf["out_of_view"]]
    tEdgesAll = pd.concat([tOov["L_exit_edge"], tOov["R_exit_edge"]])
    print("exit edges:", tEdgesAll[tEdgesAll != ""].value_counts().to_dict())
    print("out-of-view arms: L", int((tOov["L_exit_s"] > 0).sum() + 0), "R", int((tOov["R_exit_s"] > 0).sum()))


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="out-of-view audit + class folders of hard-linked renders")
    tAp.add_argument("quality_csv")
    tAp.add_argument("renders")
    tAp.add_argument("out")
    tAp.add_argument("packs", nargs="*")
    tArgs = tAp.parse_args(argv)
    auditEpisodes(tArgs.quality_csv, tArgs.renders, tArgs.out, tArgs.packs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
