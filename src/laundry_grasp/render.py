"""Causal frame-level p_grasp(t) review videos, color = probability (was grasp_render_prob.py).

Classifier: the trainAll recipe on ALL v2 gold (gold_v2.csv + gold_overrides_v2.json), trained at start; --model loads a saved
state dict instead (e.g. the frozen grasp-v2-W0.4 model.pt). p_grasp(t) at PG_HZ (default 15 Hz) with window PG_WIN_S (default
0.2 s since the user review; the frozen generator uses 0.4 s): every tick is scored (open jaw too), the causal window restarts at
each v2 jaw state change (laundry_grasp.pgrasp). Color: P(grasp) 0 red -> 0.5 yellow -> 1 green.

    python -m laundry_grasp.render OUT_DIR DEMO [...]   (DEMO = take_dNN, 1-indexed as in the video names)
        -> OUT_DIR/<demo>_pgrasp.npz (t_ns, <H>_p, <H>_valid) + <demo>_pgrasp.mp4
    python -m laundry_grasp.render OUT_DIR --wrong      -> the user-reviewed override events again, with the causal p strip
        (one clip per gold_overrides_v2.json entry: <KIND>__<id>.mp4)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths as egopaths

from .crops import ANALYSIS_W, TIP_DISK_RADIUS_FRAC, JawRoiCropper
from .demos import HAND_CAMS, demoWindow, nearestFrame, parseVideoName, wristImage
from .events import jawV2
from .gold import GOLD_V2, OVERRIDES_V2, loadOverrides, readCsv
from .model import loadModel, trainAll
from .pgrasp import causalWindowMean, changeTimes, closedAt, framePGrasp, lastChangeTimes, tickTimes
from .video import BAR_GRAY, CLOSED_BGR, RAW_SUFFIX, VIDEO_FPS, encodeH264, rawWriter, readFrame, singleCloseSpan

DEFAULT_HZ = 15.0
DEFAULT_WINDOW_S = 0.2                # 11-05 user after the confident-wrong clips: "window size를 줄이면"
PREDICT_BATCH = 64
PANE_W, PANE_H, BAR_H = 480, 270, 70
BANNER_H = 34
TIMELINE_ROW_H, TIMELINE_ROW_STEP, TIMELINE_TOP = 22, 30, 8
CLIP_W, CLIP_H, CLIP_BAR_H = 640, 360, 70
CLIP_PAD_S, CLIP_GAP_S, CLIP_MAX_S = 1.0, 0.15, 7
DEMO_CRF, CLIP_CRF = 26, 24
TIP_BGR = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
NS_PER_S = 1e9


def probabilityColor(p: float) -> tuple[int, int, int]:
    """BGR: 0 red -> 0.5 yellow -> 1 green"""
    tP = float(np.clip(p, 0, 1))
    tR = 255 if tP < 0.5 else int(255 * (1 - tP) * 2)
    tG = int(255 * tP * 2) if tP < 0.5 else 255
    return (0, tG, tR)


def pSeries(model, take: str, cam: str, ts: np.ndarray, windowS: float) -> dict:
    """p (float64), valid (all True), closed, plus the camera info the renderer needs"""
    tCrop = JawRoiCropper.forCamera(take, cam)
    tJawT, tState, tEvents = jawV2(take, cam)
    tFrame = framePGrasp(model, np.stack([tCrop(tT) for tT in ts]), PREDICT_BATCH).astype(np.float64)
    tP = causalWindowMean(ts, tFrame, lastChangeTimes(ts, changeTimes(tEvents)), windowS)
    return dict(p=tP, valid=np.ones(len(ts), bool), closed=closedAt(tJawT, tState, ts), tw=tCrop.mWristTimes, cd=tCrop.mCamDir,
                tip=tCrop.mTip)


def _pane(hand: str, hands: dict, t: float, k: int) -> np.ndarray:
    tD = hands[hand]
    tImage = readFrame(wristImage(tD["cd"], nearestFrame(tD["tw"], t)), (PANE_W, PANE_H))
    tScale = PANE_W / ANALYSIS_W
    cv2.rectangle(tImage, (0, 0), (PANE_W, BANNER_H), probabilityColor(tD["p"][k]), -1)
    tText = f"{hand}  P(grasp) {tD['p'][k]:.2f}   jaw {'CLOSED' if tD['closed'][k] else 'open'}"
    cv2.putText(tImage, tText, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
    cv2.circle(tImage, (int(tD["tip"][0] * tScale), int(tD["tip"][1] * tScale)), int(TIP_DISK_RADIUS_FRAC * ANALYSIS_W * tScale), TIP_BGR, 1)
    return tImage


def renderDemo(out: Path, demoName: str, ts: np.ndarray, t0: int, t1: int, hands: dict) -> None:
    tRaw = out.with_suffix(RAW_SUFFIX)
    tWriter = rawWriter(tRaw, (2 * PANE_W, PANE_H + BAR_H))
    tTimeline = {}
    for tHand, tD in hands.items():
        tCols = np.linspace(0, len(ts) - 1, 2 * PANE_W).astype(int)
        tTimeline[tHand] = np.array([probabilityColor(tD["p"][tI]) for tI in tCols], np.uint8)
    for tT in np.arange(t0, t1, NS_PER_S / VIDEO_FPS):
        tK = nearestFrame(ts, tT)
        tFrame = np.vstack([np.hstack([_pane(tHand, hands, tT, tK) for tHand in ("L", "R")]),
                            np.full((BAR_H, 2 * PANE_W, 3), BAR_GRAY, np.uint8)])
        for tRow, tHand in enumerate(("L", "R")):
            tY0 = PANE_H + TIMELINE_TOP + tRow * TIMELINE_ROW_STEP
            tFrame[tY0:tY0 + TIMELINE_ROW_H] = tTimeline[tHand][None]
            cv2.putText(tFrame, tHand, (4, tY0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        tX = int((tT - t0) / (t1 - t0) * (2 * PANE_W - 1))
        cv2.line(tFrame, (tX, PANE_H), (tX, PANE_H + BAR_H), WHITE, 2)
        cv2.putText(tFrame, f"{demoName}  {(tT - t0) / NS_PER_S:5.1f}s", (2 * PANE_W - 300, PANE_H + BAR_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        tWriter.write(tFrame)
    tWriter.release()
    encodeH264(tRaw, out, DEMO_CRF)


def renderDemoKey(model, out: Path, demoName: str, hz: float, windowS: float) -> None:
    tTake, tDemo = parseVideoName(demoName)
    tT0, tT1 = demoWindow(tTake, tDemo)
    tTs = tickTimes(tT0, tT1, hz)
    tHands = {tHand: pSeries(model, tTake, tCam, tTs, windowS) for tHand, tCam in HAND_CAMS}
    np.savez(out / f"{demoName}_pgrasp.npz", t_ns=tTs, **{f"{tH}_p": tD["p"] for tH, tD in tHands.items()},
             **{f"{tH}_valid": tD["valid"].astype(np.int8) for tH, tD in tHands.items()})
    renderDemo(out / f"{demoName}_pgrasp.mp4", demoName, tTs, tT0, tT1, tHands)
    print(demoName, {tH: f"valid {tD['valid'].mean():.0%} mean p {tD['p'][tD['valid']].mean():.2f}" if tD["valid"].any() else "-"
                     for tH, tD in tHands.items()}, flush=True)


def renderOverrideClips(model, out: Path, goldCsv: Path, overrides: dict, hz: float, windowS: float) -> None:
    """the user-reviewed override events (was grasp_render_prob.wrong_clips): wrist clip + causal P(grasp) strip"""
    tGold = {tG["id"]: tG for tG in readCsv(goldCsv)}
    for tId, tKind in overrides.items():
        if tId.startswith("_"):
            continue
        tRow = tGold[tId]
        _, _, tEvents = jawV2(tRow["take"], tRow["cam"])
        tClose, tOpen = int(tRow["t_close_ns"]), int(tRow["t_open_ns"])
        tA, tB = singleCloseSpan(tEvents, tClose, tOpen, CLIP_PAD_S, CLIP_GAP_S, CLIP_MAX_S)
        tTs = np.arange(tA, tB, NS_PER_S / hz).astype(np.int64)
        tD = pSeries(model, tRow["take"], tRow["cam"], tTs, windowS)
        tName = f"{tKind.upper()}__{tId}"
        tRaw = out / f"{tName}{RAW_SUFFIX}"
        tWriter = rawWriter(tRaw, (CLIP_W, CLIP_H + CLIP_BAR_H))
        tStrip = np.array([probabilityColor(tX) for tX in tD["p"][np.linspace(0, len(tTs) - 1, CLIP_W).astype(int)]], np.uint8)
        for tT in np.arange(tA, tB, NS_PER_S / VIDEO_FPS):
            tK = nearestFrame(tTs, tT)
            tImage = readFrame(wristImage(tD["cd"], nearestFrame(tD["tw"], tT)), (CLIP_W, CLIP_H))
            tBar = np.full((CLIP_BAR_H, CLIP_W, 3), BAR_GRAY, np.uint8)
            tBar[4:26] = tStrip[None]
            tX = int((tT - tA) / (tB - tA) * (CLIP_W - 1))
            cv2.line(tBar, (tX, 0), (tX, 30), WHITE, 2)
            tClosed = tClose <= tT < tOpen
            if tClosed:
                cv2.rectangle(tBar, (0, 32), (CLIP_W - 1, 38), CLOSED_BGR, -1)
            cv2.putText(tBar, f"{tKind}  P(grasp) {tD['p'][tK]:.2f}  {'JAW CLOSED' if tClosed else 'jaw open'}  {(tT - tA) / NS_PER_S:4.1f}s",
                        (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            tWriter.write(np.vstack([tImage, tBar]))
        tWriter.release()
        encodeH264(tRaw, out / f"{tName}.mp4", CLIP_CRF)
        tIn = (tTs >= tClose) & (tTs < tOpen)
        print(tName, f"P(grasp) closed max {tD['p'][tIn].max():.2f} end {tD['p'][tIn][-1]:.2f}", flush=True)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("out_dir")
    tParser.add_argument("demos", nargs="*", help="take_dNN demo names (1-indexed)")
    tParser.add_argument("--wrong", action="store_true", help="render the override events instead of demos")
    tParser.add_argument("--model", default=None, help="state dict to load instead of training on all gold")
    tParser.add_argument("--gold", default=str(egopaths.GRASP_LABELS_V2 / GOLD_V2))
    tParser.add_argument("--overrides", default=str(egopaths.GRASP_LABELS_V2 / OVERRIDES_V2))
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out_dir).expanduser()
    tOut.mkdir(parents=True, exist_ok=True)
    tHz = float(os.environ.get("PG_HZ", DEFAULT_HZ))
    tWindowS = float(os.environ.get("PG_WIN_S", DEFAULT_WINDOW_S))
    tOverrides = loadOverrides(Path(tArgs.overrides))
    if tArgs.model:
        tModel = loadModel(Path(tArgs.model))
    else:
        tCroppers: dict = {}

        def tCropperFor(take: str, cam: str) -> JawRoiCropper:
            if (take, cam) not in tCroppers:
                tCroppers[(take, cam)] = JawRoiCropper.forCamera(take, cam)
            return tCroppers[(take, cam)]

        tModel = trainAll(readCsv(Path(tArgs.gold)), tOverrides, tCropperFor)
    if tArgs.wrong:
        renderOverrideClips(tModel, tOut, Path(tArgs.gold), tOverrides, tHz, tWindowS)
        return 0
    for tDemo in tArgs.demos:
        renderDemoKey(tModel, tOut, tDemo, tHz, tWindowS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
