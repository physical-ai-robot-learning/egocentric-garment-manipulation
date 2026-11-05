"""Fusion QA video per demo (both arms) from FUSION_WORK/stitch/qa/<var>/ runs + the CoTracker pack (was
orbbec/v2b/fusion/render_fusion_qa.py, 10-24; writes only to --out).
Left: Orbbec RGB (Orbbec frame rate, idle-trim window) with per arm  production TCP (h-1, filled circle; L green / R cyan), hidden-tag
estimate (the run in which that frame's 4 s block is hidden, orange ring), tag GT (white +, >= 1 tag), raw CoTracker tag-point (small
yellow dot), and a red border flash while |hidden estimate - tag| > 10 cm.  Below: left / right wrist cameras (nearest frame).
Right: per arm x/y/z (production solid, hidden estimate orange, tag dots), |hidden estimate - tag| [mm] with the 100 mm line,
CoTracker gaps shaded red, hidden blocks of the current run not shaded (every frame is hidden in exactly one run); moving cursor.
Needs ffmpeg on PATH; the Orbbec colour frames are 848 x 480.
    python -m laundry_tracking.fusion.render_qa --pack <ctpack> --var fus_Cg_s --demos TAKE:demo ... --out DIR [--tag-csv fusion_metrics_<tag>.csv]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths

from .. import workdirs
from ..camera_calibration import cameraFromBase
from ..constants import MM_PER_M, NS_PER_S
from ..fgo import arm_data

DEFAULT_RAW_VAR = "fus_R_s"
COLOURS = {"left": (60, 220, 60), "right": (230, 200, 40)}     # BGR
ARMS = ("left", "right")
HIDE_RUNS = (0, 1, 2)
HIDE_BLOCK_NS = int(4e9)           # every 4 s block is hidden in exactly one run (block % 3)
MIN_PROJ_Z_M = 0.05                # points closer to the camera plane are not drawn
IMAGE_W, IMAGE_H = 848, 480
FAR_MM = 100
PLOT_SCALE = 1.5                   # the plot panel is 1.5 x the image height
MIN_VIDEO_H = 720
WRIST_TOP, WRIST_BOTTOM, WRIST_W, WRIST_H = 482, 720, 423, 238
WRIST_COL = 424
VIDEO_FPS = "15"
VIDEO_CRF = "23"
ORANGE_BGR = (0, 140, 255)
YELLOW_BGR = (0, 230, 255)
RED_BGR = (0, 0, 255)
WHITE_BGR = (255, 255, 255)
CURSOR_BGR = (0, 0, 200)


def interp(t, p, tq) -> np.ndarray:
    """per-axis linear interpolation, NaN outside [t0, t1]"""
    tOut = np.stack([np.interp(tq, t, p[:, tAx]) for tAx in range(3)], 1)
    tOut[(tq < t[0]) | (tq > t[-1])] = np.nan
    return tOut


def project(p: np.ndarray, intr: dict, tcb: np.ndarray):
    """one base-frame point -> integer pixel, or None (not finite / behind the camera) (was render_fusion_qa.proj)"""
    if not np.isfinite(p[0]):
        return None
    tX = tcb[:3, :3] @ p + tcb[:3, 3]
    if tX[2] <= MIN_PROJ_Z_M:
        return None
    return int(intr["fx"] * tX[0] / tX[2] + intr["cx"]), int(intr["fy"] * tX[1] / tX[2] + intr["cy"])


def loadRun(var: str, take: str, arm: str, demo: int, off: int):
    """a stitch/qa run, or None when missing or with < 2 samples (e.g. an arm gated to 0 %) (was render_fusion_qa.load)"""
    tFile = workdirs.qaRunPath(var, take, arm, demo, off)
    if not tFile.exists():
        return None
    tZ = np.load(tFile)
    return tZ if len(tZ["t_ns"]) >= 2 else None


def render(pack: str, var: str, take: str, demo: int, out: Path, info: dict, rawVar: str = DEFAULT_RAW_VAR) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tTcb = cameraFromBase(take)                                   # 10-26: per-take camera calibration
    tTakeDir = paths.TAG_RIG / take
    tIntr = json.loads((tTakeDir / "meta.json").read_text())["intrinsics"]
    tWin = arm_data.demoWindow(take, demo)
    S = {}
    for tArm in ARMS:
        D = arm_data.loadArm(take, tArm, withSlam=False)
        tT = D["t"]
        tK = np.flatnonzero((tT >= tWin.mW0) & (tT <= tWin.mW1))
        tTt = tT[tK]
        tProd = loadRun(var, take, tArm, demo, -1)
        tOffOf = ((tTt - tT[0]) // HIDE_BLOCK_NS) % 3
        tHid = np.full((len(tK), 3), np.nan)
        for tOff in HIDE_RUNS:
            tR = loadRun(var, take, tArm, demo, tOff)
            if tR is not None:
                tM = tOffOf == tOff
                tHid[tM] = interp(tR["t_ns"], tR["p_B"], tTt[tM])
        tZ = np.load(Path(pack) / f"{take}_d{demo:02d}" / f"solve2_{tArm}.npz")
        tTct = tZ["t"].astype(np.int64)
        tOkC = tZ["conf"] > 0
        tRaw = loadRun(rawVar, take, tArm, demo, -1)
        S[tArm] = dict(k=tK, tt=tTt,
                       prod=interp(tProd["t_ns"], tProd["p_B"], tTt) if tProd is not None else np.full((len(tK), 3), np.nan),
                       hid=tHid, tag=np.where((D["ntags"][tK] >= 1)[:, None], D["p_tag"][tK], np.nan),
                       ct=interp(tRaw["t_ns"], tRaw["p_B"], tTt) if tRaw is not None else None,
                       ctgap=~np.interp(tTt, tTct, tOkC.astype(float)).round().astype(bool) | (tTt < tTct[0]) | (tTt > tTct[-1]))
        tWristDir = tTakeDir / D["cam"]
        tWristT = np.load(tWristDir / "t_ns.npy").astype(np.int64)
        S[tArm]["wrist"] = (tWristDir, np.clip(np.searchsorted(tWristT, tTt), 0, len(tWristT) - 1))
        S[tArm]["err"] = np.linalg.norm(S[tArm]["hid"] - S[tArm]["tag"], axis=1) * MM_PER_M
    tK = S["left"]["k"]
    tTt = S["left"]["tt"]
    tTs = (tTt - tTt[0]) / NS_PER_S
    # static plot
    tFig, tAx = plt.subplots(4, 2, figsize=(8.5, 7.2), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.2]))
    for tCol, tArm in enumerate(ARMS):
        tS = S[tArm]
        tGap = tS["ctgap"]
        for tRow, tLab in enumerate("xyz"):
            tA = tAx[tRow, tCol]
            tA.plot(tTs, tS["prod"][:, tRow], "g-" if tArm == "left" else "c-", lw=1.2)
            tA.plot(tTs, tS["hid"][:, tRow], color="orange", lw=0.9)
            tA.plot(tTs, tS["tag"][:, tRow], "k.", ms=1.5)
            tA.set_ylabel(f"{tArm[0].upper()} {tLab} [m]", fontsize=7)
            tA.tick_params(labelsize=6)
            tA.fill_between(tTs, 0, 1, where=tGap, transform=tA.get_xaxis_transform(), color="red", alpha=0.18, lw=0)
        tA = tAx[3, tCol]
        tA.plot(tTs, tS["err"], color="orange", lw=0.9)
        tA.axhline(FAR_MM, color="r", lw=0.8, ls="--")
        tA.set_ylim(0, max(150, np.nanmax(np.r_[tS["err"], 0]) * 1.05))
        tA.fill_between(tTs, 0, 1, where=tGap, transform=tA.get_xaxis_transform(), color="red", alpha=0.18, lw=0)
        tA.set_ylabel("|hid - tag| mm", fontsize=7)
        tA.tick_params(labelsize=6)
        tA.set_xlabel("s", fontsize=7)
        tAx[0, tCol].set_title(f"{tArm}: " + info.get(tArm, ""), fontsize=7)
    tFig.suptitle(f"{take} d{demo:02d}  {var}   (solid = production, orange = hidden-tag estimate, dots = tag, red = CoTracker gap)",
                  fontsize=8)
    tFig.tight_layout(rect=(0, 0, 1, 0.97))
    tFig.canvas.draw()
    tPlot = cv2.cvtColor(np.asarray(tFig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    tExt0 = tAx[3, 0].get_window_extent()
    tExt1 = tAx[3, 1].get_window_extent()
    tPlotH0 = tPlot.shape[0]
    plt.close(tFig)

    def _cursorX(ext, s):
        return int(ext.x0 + (s - tTs[0]) / (tTs[-1] - tTs[0] + 1e-9) * (ext.x1 - ext.x0))
    tPlot = cv2.resize(tPlot, (int(tPlot.shape[1] * IMAGE_H * PLOT_SCALE / tPlot.shape[0]), int(IMAGE_H * PLOT_SCALE)))
    tScale = tPlot.shape[0] / tPlotH0
    out.parent.mkdir(parents=True, exist_ok=True)
    tVidW = IMAGE_W + tPlot.shape[1]
    tVidH = max(tPlot.shape[0], IMAGE_H + 40)
    tVidH = max(tVidH, MIN_VIDEO_H)
    tVidW += tVidW % 2
    tVidH += tVidH % 2
    tFfmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                f"{tVidW}x{tVidH}", "-r", VIDEO_FPS, "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                                VIDEO_CRF, str(out)], stdin=subprocess.PIPE)
    for tJ, tKk in enumerate(tK):
        tImg = cv2.imread(str(tTakeDir / f"color_{tKk:05d}.jpg"))
        tCanvas = np.zeros((tVidH, tVidW, 3), np.uint8)
        if tImg is None:
            tImg = np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8)
        tBad = False
        for tArm in ARMS:
            tS = S[tArm]
            tC = COLOURS[tArm]
            if tS["ct"] is not None and (tQ := project(tS["ct"][tJ], tIntr, tTcb)):
                cv2.circle(tImg, tQ, 3, YELLOW_BGR, -1)
            if tQ := project(tS["prod"][tJ], tIntr, tTcb):
                cv2.circle(tImg, tQ, 7, tC, -1)
            if tQ := project(tS["hid"][tJ], tIntr, tTcb):
                cv2.circle(tImg, tQ, 11, ORANGE_BGR, 2)
            if tQ := project(tS["tag"][tJ], tIntr, tTcb):
                cv2.drawMarker(tImg, tQ, WHITE_BGR, cv2.MARKER_CROSS, 16, 2)
            tE = tS["err"][tJ]
            tBad |= bool(np.isfinite(tE) and tE > FAR_MM)
            cv2.putText(tImg, f"{tArm[0].upper()} err {tE:5.0f} mm" if np.isfinite(tE) else f"{tArm[0].upper()} err   -",
                        (10 if tArm == "left" else 640, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tC, 1)
            if tS["ctgap"][tJ]:
                cv2.putText(tImg, "CT GAP", (10 if tArm == "left" else 640, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_BGR, 2)
        if tBad:
            cv2.rectangle(tImg, (0, 0), (IMAGE_W - 1, IMAGE_H - 1), RED_BGR, 6)
        cv2.putText(tImg, f"{take} d{demo:02d}  t={tTs[tJ]:5.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_BGR, 2)
        tCanvas[:IMAGE_H, :IMAGE_W] = tImg
        tPlot2 = tPlot.copy()
        for tCol, tArm in enumerate(ARMS):                         # wrist cameras below (L | R)
            tWristDir, tWi = S[tArm]["wrist"]
            tWristImg = cv2.imread(str(tWristDir / f"wrist_{tWi[tJ]:05d}.jpg"))
            if tWristImg is not None:
                tCanvas[WRIST_TOP:WRIST_BOTTOM, WRIST_COL * tCol:WRIST_COL * tCol + WRIST_W] = cv2.resize(tWristImg, (WRIST_W,
                        WRIST_H))
                cv2.putText(tCanvas, f"{tArm} wrist", (WRIST_COL * tCol + 6, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[tArm],
                            1)
        for tExt in (tExt0, tExt1):
            tXx = int(_cursorX(tExt, tTs[tJ]) * tScale)
            cv2.line(tPlot2, (tXx, 0), (tXx, tPlot2.shape[0] - 1), CURSOR_BGR, 1)
        tCanvas[:tPlot2.shape[0], IMAGE_W:IMAGE_W + tPlot2.shape[1]] = tPlot2
        cv2.putText(tCanvas, "fill = production  orange ring = hidden-tag estimate  white + = tag  yellow = raw CoTracker  red border = err > 10 cm",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        tFfmpeg.stdin.write(tCanvas.tobytes())
    tFfmpeg.stdin.close()
    tFfmpeg.wait()


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--pack", required=True)
    tAp.add_argument("--var", required=True)
    tAp.add_argument("--demos", nargs="+", required=True)
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--tag-csv", default="")
    tAp.add_argument("--prefix", default="")
    tAp.add_argument("--raw-var", default="fus_R_s")
    tAp.add_argument("--quality-csv", default="")
    tAp.add_argument("--start", type=int, default=1)
    return tAp


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    tArgs = _buildParser().parse_args(argv)
    tMetrics = pd.read_csv(workdirs.STITCH_QA_DIR / tArgs.tag_csv) if tArgs.tag_csv else None
    tQuality = pd.read_csv(tArgs.quality_csv, dtype={"take": str}) if tArgs.quality_csv else None
    for tI, tSpec in enumerate(tArgs.demos):
        tTake, tDemo = tSpec.split(":")
        tDemo = int(tDemo)
        tInfo = {}
        if tQuality is not None:
            tR = tQuality[(tQuality["take"] == tTake) & (tQuality["demo"] == tDemo)]
            if len(tR):
                tR = tR.iloc[0]
                for tArm in ARMS:
                    tL = tArm[0].upper()
                    tInfo[tArm] = (f"[{tR['class']}] cov {tR[tL + '_cov']:.2f} | gap {tR[tL + '_gap_s']:.1f}s | far10 {tR[tL + '_far10']:.2f} | "
                                   f"burst {tR[tL + '_burst_s']:.1f}s | swap {tR[tL + '_swap_s']:.1f}s")
        if tMetrics is not None:
            for tArm in ARMS:
                tR = tMetrics[(tMetrics["var"] == tArgs.var) & (tMetrics["take"].astype(str) == tTake) & (tMetrics["demo"] == tDemo)
                              & (tMetrics["arm"] == tArm)]
                if len(tR):
                    tR = tR.iloc[0]
                    tInfo[tArm] = f"e32 p90 {tR.e32_p90:.0f} | abs p90 {tR.abs_p90:.0f} | far10 {tR.far10:.2f} | burst {tR.burst_s:.1f}s | gap {tR.gap_s:.1f}s"
        tName = f"{tArgs.prefix}{tI + tArgs.start:02d}_{tTake}_d{tDemo:02d}.mp4" if tArgs.prefix else f"{tTake}_d{tDemo:02d}.mp4"
        render(tArgs.pack, tArgs.var, tTake, tDemo, Path(tArgs.out) / tName, tInfo, tArgs.raw_var)
        print("wrote", tName, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
