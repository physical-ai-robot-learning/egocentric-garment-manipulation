"""README GIF (was orbbec/v2b/fusion/render_traj_sources.py, 10-27): fused jaw-tip paths of one demo coloured by the source of each frame
(same layout as the 10-21 SLAM-era GIF).  Left: Orbbec RGB with the path so far projected per arm; right: both arms in the robot base
frame (mm).  Sources: green = a gripper marker is visible (markers + CoTracker + depth), blue = CoTracker points only (no marker),
orange = CoTracker gap (IMU acceleration inside the gap).
Inputs: the FUSION_WORK/stitch/qa/<var> production run, the pack's solve2 files, the take's colour frames; needs ffmpeg on PATH.
    python -m laundry_tracking.fusion.render_sources --pack <ctpack> --demo TAKE:demo --out x.gif [--var fus_Cg_g] [--fps 5] [--speed 1.5]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths

from ..camera_calibration import cameraFromBase
from ..constants import MM_PER_M, NS_PER_S
from ..fgo import arm_data
from .render_qa import interp, loadRun

ARMS = ("left", "right")
SRC_MARKER, SRC_COTRACKER, SRC_IMU_GAP = 0, 1, 2
SRC_BGR = {SRC_MARKER: (60, 200, 60), SRC_COTRACKER: (235, 140, 50), SRC_IMU_GAP: (30, 140, 255)}
SRC_RGB = {tK: tuple(tC / 255 for tC in tV[::-1]) for tK, tV in SRC_BGR.items()}
LABELS = ["AprilTag markers + CoTracker + depth", "CoTracker points only (marker hidden)", "IMU inside a CoTracker gap"]
MIN_PROJ_Z_M = 0.05
PANEL_H = 420                      # both panels are scaled to this height before joining
PALETTE_COLOURS = "200"
WHITE_BGR = (255, 255, 255)


def renderSources(pack: str, demoSpec: str, out: str, var: str, fps: float, speed: float, width: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tTake, tDemo = demoSpec.split(":")
    tDemo = int(tDemo)
    tTcb = cameraFromBase(tTake)
    tTakeDir = paths.TAG_RIG / tTake
    tIntr = json.loads((tTakeDir / "meta.json").read_text())["intrinsics"]
    tWin = arm_data.demoWindow(tTake, tDemo)
    S = {}
    for tArm in ARMS:
        D = arm_data.loadArm(tTake, tArm, withSlam=False)
        tT = D["t"]
        tK = np.flatnonzero((tT >= tWin.mW0) & (tT <= tWin.mW1))
        tTt = tT[tK]
        tProd = loadRun(var, tTake, tArm, tDemo, -1)
        tP = interp(tProd["t_ns"], tProd["p_B"], tTt)
        tZ = np.load(Path(pack) / f"{tTake}_d{tDemo:02d}" / f"solve2_{tArm}.npz")
        tTct = tZ["t"].astype(np.int64)
        tOkC = tZ["conf"] > 0
        tGap = ~np.interp(tTt, tTct, tOkC.astype(float)).round().astype(bool) | (tTt < tTct[0]) | (tTt > tTct[-1])
        tSrc = np.where(tGap, SRC_IMU_GAP, np.where(D["ntags"][tK] >= 1, SRC_MARKER, SRC_COTRACKER))
        S[tArm] = dict(k=tK, tt=tTt, p=tP, src=tSrc)
    tK = S["left"]["k"]
    tTt = S["left"]["tt"]
    tTs = (tTt - tTt[0]) / NS_PER_S

    def _project(q):
        if not np.isfinite(q[0]):
            return None
        tX = tTcb[:3, :3] @ q + tTcb[:3, 3]
        return None if tX[2] <= MIN_PROJ_Z_M else (int(tIntr["fx"] * tX[0] / tX[2] + tIntr["cx"]),
                                                   int(tIntr["fy"] * tX[1] / tX[2] + tIntr["cy"]))
    tPix = {tArm: [_project(tQ) for tQ in S[tArm]["p"]] for tArm in ARMS}
    tAll = np.concatenate([S[tArm]["p"] for tArm in ARMS]) * MM_PER_M
    tLo = np.nanmin(tAll, 0)
    tHi = np.nanmax(tAll, 0)
    tFrac = {tArm: {tS: float(np.mean(S[tArm]["src"] == tS)) for tS in (0, 1, 2)} for tArm in ARMS}
    print("source fractions",
          {tArm: {LABELS[tS].split()[0]: round(tV, 3) for tS, tV in tF.items()} for tArm, tF in tFrac.items()})
    tStep = max(1, int(round(len(tTs) / (tTs[-1] - tTs[0]) * speed / fps)))
    tIdx = list(range(0, len(tTs), tStep))
    tTmp = Path(tempfile.mkdtemp())
    for tN, tJ in enumerate(tIdx):
        tImg = cv2.imread(str(tTakeDir / f"color_{tK[tJ]:05d}.jpg"))
        for tArm in ARMS:
            tS = S[tArm]
            tPts = tPix[tArm]
            for tI in range(1, tJ + 1):
                if tPts[tI - 1] and tPts[tI]:
                    cv2.line(tImg, tPts[tI - 1], tPts[tI], SRC_BGR[int(tS["src"][tI])], 3, cv2.LINE_AA)
            if tPts[tJ]:
                cv2.circle(tImg, tPts[tJ], 7, WHITE_BGR, -1)
                cv2.circle(tImg, tPts[tJ], 7, SRC_BGR[int(tS["src"][tJ])], 3)
        cv2.putText(tImg, f"t = {tTs[tJ]:4.1f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE_BGR, 2)
        for tR, tLab in enumerate(LABELS):
            tY = 62 + 22 * tR
            cv2.line(tImg, (14, tY - 5), (40, tY - 5), SRC_BGR[tR], 4)
            cv2.putText(tImg, tLab, (48, tY), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE_BGR, 2)
        tFig = plt.figure(figsize=(5.2, 4.2), dpi=100)
        tAx = tFig.add_subplot(projection="3d")
        for tArm in ARMS:
            tS = S[tArm]
            tQ = tS["p"][: tJ + 1] * MM_PER_M
            tSr = tS["src"][: tJ + 1]
            tCut = np.flatnonzero(np.diff(tSr)) + 1
            for tA0, tA1 in zip(np.r_[0, tCut],
                                np.r_[tCut, len(tSr)]):     # one line per same-source run (joined to the next point)
                tSeg = tQ[tA0:min(tA1 + 1, len(tQ))]
                tAx.plot(tSeg[:, 0], tSeg[:, 1], tSeg[:, 2], color=SRC_RGB[int(tSr[tA0])], lw=1.6)
            if np.isfinite(tQ[-1, 0]):
                tAx.scatter(*tQ[-1], s=40, color="k")
                tAx.text(*tQ[-1], f" {tArm} hand", fontsize=8, weight="bold")
        tAx.set_xlim(tLo[0], tHi[0])
        tAx.set_ylim(tLo[1], tHi[1])
        tAx.set_zlim(min(tLo[2], 0), tHi[2])
        tAx.set_xlabel("x [mm]", fontsize=7)
        tAx.set_ylabel("y [mm]", fontsize=7)
        tAx.set_zlabel("z [mm]", fontsize=7)
        tAx.tick_params(labelsize=6)
        tAx.set_title("jaw-tip paths in the robot base frame", fontsize=9)
        tAx.view_init(24, -135 + 30 * tN / max(1, len(tIdx) - 1))
        tFig.tight_layout()
        tFig.canvas.draw()
        tPlot = cv2.cvtColor(np.asarray(tFig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
        plt.close(tFig)
        tImg = cv2.resize(tImg, (int(tImg.shape[1] * PANEL_H / tImg.shape[0]), PANEL_H))
        tPlot = cv2.resize(tPlot, (int(tPlot.shape[1] * PANEL_H / tPlot.shape[0]), PANEL_H))
        tFrame = np.hstack([tImg, np.full((PANEL_H, 8, 3), 255, np.uint8), tPlot])
        tFrame = cv2.resize(tFrame, (width, int(tFrame.shape[0] * width / tFrame.shape[1])))
        cv2.imwrite(str(tTmp / f"f_{tN:04d}.png"), tFrame)
    tPalette = tTmp / "pal.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tTmp / "f_%04d.png"), "-vf",
                    f"palettegen=max_colors={PALETTE_COLOURS}", str(tPalette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tTmp / "f_%04d.png"), "-i",
                    str(tPalette), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", out], check=True)
    print("wrote", out, len(tIdx), "frames")


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--pack", required=True)
    tAp.add_argument("--demo", required=True)
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--var", default="fus_Cg_g")
    tAp.add_argument("--fps", type=float, default=5)
    tAp.add_argument("--speed", type=float, default=1.5)
    tAp.add_argument("--width", type=int, default=1080)
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    renderSources(tArgs.pack, tArgs.demo, tArgs.out, tArgs.var, tArgs.fps, tArgs.speed, tArgs.width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
