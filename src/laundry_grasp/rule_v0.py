"""Grasp held / empty labeler v0: the rule baseline that produced the round-1 labeling sample (was the CLI part of grasp_label_v0.py;
NOT validated as a label source -- the user's gold check showed it is too weak, see README).

Per close event of each HandUMI jaw (wrist/umi_jaw.npz ev_t, the v1 jaw segmentation), decide held / empty / unknown from that hand's
wrist camera. The jaw servo is torque-off position-only (no load/current), and the closed position does not separate cloth from
nothing (right jaw closes to 5263-5279 ticks every time, left varies ~300 ticks with grip force) -> vision cue: while the jaw is
closed AND the hand moves (wrist-cam ego flow), the jaw-tip disk (minus the fixed jaw pixels) shows
    held    = cloth stays there and moves WITH the camera (dark fraction high, tip flow << ego flow)
    empty   = the tip disk sees the table / background (dark fraction low)
    unknown = hand did not move while closed, or the cues disagree (never forced)
Frame labels on the jaw stream: open / closing (first CLOSING_S after a close event) / held / empty / unknown.
Dark = gray < 90 (the 11-03/11-04 shirts are black on a white table -- colored shirts need a different cloth cue).

    python -m laundry_grasp.rule_v0 OUT_DIR TAKE/demoNN [...]
        -> OUT_DIR/<take>_dNN_H<n>E<n>U<n>.mp4 + events.csv + labels/<take>_dNN.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from .crops import ANALYSIS_H, ANALYSIS_W, DARK_GRAY, TIP_DISK_RADIUS_FRAC, jawMaskAndTip, readGray
from .demos import HAND_CAMS, camDir, demoWindow, nearestFrame, parseDemoKey, videoName, wristImage
from .events import closeIntervals, isRest, overlapsWindow
from .video import BAR_GRAY, RAW_SUFFIX, VIDEO_FPS, encodeH264, rawWriter, readFrame

CLOTH_HELD, CLOTH_EMPTY = 0.5, 0.2    # dark share of the tip disk
RATIO_HELD = 0.5                      # tip flow / ego flow below this = cloth moves with the jaw
RATIO_SLIDING = 0.8                   # ... at or above this = cloth slides under the tips
EGO_MOVE_PX = 1.0                     # median flow outside the jaw and disk = the hand moves
MIN_MOVING = 3                        # moving samples needed for a decision
CLOSING_S = 0.3                       # first part of a close is 'closing', not judged
SAMPLE_HZ = 10.0
TEX_GRAD = 40.0                       # textured-pixel gradient
TEX_MIN_FRAC = 0.05                   # min textured share of the disk
FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)   # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
MIN_EGO_PX = 1e-3
FEATURE_DECIMALS = 2
NS_PER_S = 1e9
COLORS = {"open": (200, 200, 200), "closing": (0, 200, 255), "held": (60, 200, 60), "empty": (40, 40, 230),
          "unknown": (200, 120, 200)}
LABELS = ("open", "closing", "held", "empty", "unknown")
UNKNOWN = LABELS.index("unknown")
CLOSING = LABELS.index("closing")
PANE_W, PANE_H, BAR_H = 480, 270, 70
TIMELINE_ROW_H, TIMELINE_ROW_STEP, TIMELINE_TOP = 22, 30, 8
BANNER_H = 34
TIP_BGR = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CRF = 26
NO_DECISION = ("unknown", "hand still while closed")


def diskMask(tip: tuple[int, int], radius: int, jawMask: np.ndarray) -> np.ndarray:
    tYy, tXx = np.mgrid[:ANALYSIS_H, :ANALYSIS_W]
    return ((tXx - tip[0]) ** 2 + (tYy - tip[1]) ** 2 <= radius * radius) & ~jawMask


def decide(features: dict) -> tuple[str, str]:
    """(label, reason) from the event features"""
    if features["n_moving"] < MIN_MOVING:
        return NO_DECISION
    if features["cloth"] < CLOTH_EMPTY:
        return "empty", "tips see table/background"
    if features["ratio"] is None:
        return "unknown", "no texture at the tips (uniform cloth)"
    if features["cloth"] >= CLOTH_HELD and features["ratio"] < RATIO_HELD:
        return "held", "cloth at tips moves with the jaw"
    if features["ratio"] >= RATIO_SLIDING:
        return "empty", "cloth slides under the tips"
    return "unknown", "cues disagree"


def eventFeatures(cameraDir: Path, wristTimes: np.ndarray, tA: float, tB: float, jawMask: np.ndarray, tip: tuple[int, int],
                  radius: int) -> tuple[str, str, dict]:
    """(label, reason, features) of one closed interval [tA, tB) sampled at SAMPLE_HZ"""
    tDisk = diskMask(tip, radius, jawMask)
    tRest = ~jawMask & ~tDisk
    tTs = np.arange(tA, tB, NS_PER_S / SAMPLE_HZ)
    tIdx = np.unique(np.searchsorted(wristTimes, tTs).clip(0, len(wristTimes) - 1))
    tPrev, tRows = None, []
    for tI in tIdx:
        tGray = readGray(cameraDir, tI)
        if tGray is None:
            continue
        if tPrev is not None:
            tFlow = np.linalg.norm(cv2.calcOpticalFlowFarneback(tPrev, tGray, None, *FARNEBACK), axis=2)
            tEgo = float(np.median(tFlow[tRest]))
            tGrad = cv2.magnitude(cv2.Sobel(tGray, cv2.CV_32F, 1, 0), cv2.Sobel(tGray, cv2.CV_32F, 0, 1))
            tTex = tDisk & (tGrad > TEX_GRAD)
            # flow on textured disk pixels only: uniform black cloth gives ~0 Farneback flow whether it is held or sliding
            tRatio = float(np.median(tFlow[tTex])) / max(tEgo, MIN_EGO_PX) if tTex.sum() >= TEX_MIN_FRAC * tDisk.sum() else np.nan
            tRows.append((tEgo, tRatio, float((tGray[tDisk] < DARK_GRAY).mean())))
        tPrev = tGray
    tR = np.array(tRows).reshape(-1, 3)
    tMoving = tR[tR[:, 0] > EGO_MOVE_PX]
    tFeatures = dict(
        n_samples=len(tR), n_moving=len(tMoving),
        ego_med=round(float(np.median(tR[:, 0])), FEATURE_DECIMALS) if len(tR) else 0.0,
        cloth=round(float(np.median(tMoving[:, 2])), FEATURE_DECIMALS) if len(tMoving) else None,
        ratio=(round(float(np.nanmedian(tMoving[:, 1])), FEATURE_DECIMALS)
               if len(tMoving) and np.isfinite(tMoving[:, 1]).sum() >= MIN_MOVING else None))
    tLabel, tWhy = decide(tFeatures)
    return tLabel, tWhy, tFeatures


def labelHand(take: str, cam: str, t0: int, t1: int) -> dict:
    """v0 labels of one hand over a demo window: per-sample jaw labels + one row per close event"""
    tDir = camDir(take, cam)
    tJaw = np.load(tDir / "umi_jaw.npz")
    tWrist = np.load(tDir / "t_ns.npy")
    tMask, tTip = jawMaskAndTip(tDir, tWrist, tJaw)
    tRadius = int(TIP_DISK_RADIUS_FRAC * ANALYSIS_W)
    tEvents = list(zip(tJaw["ev_t"].tolist(), tJaw["ev_kind"].tolist()))
    tJawT = tJaw["t_ns"]
    tLab = np.where(tJaw["state"] == 0, UNKNOWN, 0).astype(np.int8)   # closed -> unknown until an event decides it
    tRows = []
    for tClose, tOpen in closeIntervals(tEvents, int(tJawT[-1])):
        if not overlapsWindow(tClose, tOpen, t0, t1):
            continue
        tA, tB = max(tClose + CLOSING_S * NS_PER_S, t0), min(tOpen, t1)
        tLabel, tWhy, tFeatures = eventFeatures(tDir, tWrist, tA, tB, tMask, tTip, tRadius)
        tSeg = (tJawT >= tClose) & (tJawT < tOpen)
        tLab[tSeg] = LABELS.index(tLabel)
        tLab[tSeg & (tJawT < tClose + CLOSING_S * NS_PER_S)] = CLOSING
        tRows.append(dict(t_close_ns=tClose, t_open_ns=tOpen, closed_s=round((tOpen - tClose) / NS_PER_S, 2), label=tLabel, why=tWhy,
                          rest=int(isRest(tClose, tOpen, t0, t1)), **tFeatures))
    return dict(tj=tJawT, lab=tLab, events=tRows, mask=tMask, tip=tTip, r=tRadius, tw=tWrist, cam_dir=tDir)


def _pane(hand: str, hands: dict, t: float) -> np.ndarray:
    tD = hands[hand]
    tImage = readFrame(wristImage(tD["cam_dir"], nearestFrame(tD["tw"], t)), (PANE_W, PANE_H))
    tScale = PANE_W / ANALYSIS_W
    cv2.circle(tImage, (int(tD["tip"][0] * tScale), int(tD["tip"][1] * tScale)), int(tD["r"] * tScale), TIP_BGR, 1)
    tName = LABELS[tD["lab"][nearestFrame(tD["tj"], t)]]
    tOpenEvents = [tN for tN, tE in enumerate(tD["events"]) if tE["t_close_ns"] <= t < tE["t_open_ns"]]
    tText = f"{hand}  {tName.upper()}"
    if tOpenEvents:
        tText += f"  #{hand}{tOpenEvents[0] + 1}" + (" (rest)" if tD["events"][tOpenEvents[0]]["rest"] else "")
    cv2.rectangle(tImage, (0, 0), (PANE_W, BANNER_H), COLORS[tName], -1)
    cv2.putText(tImage, tText, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
    return tImage


def renderDemo(outMp4: Path, take: str, demo: int, t0: int, t1: int, hands: dict) -> None:
    """two wrist panes with the label banner + per-hand label timelines, real time at 15 fps (was grasp_label_v0.render)"""
    tTs = np.arange(t0, t1, NS_PER_S / VIDEO_FPS)
    tRaw = outMp4.with_suffix(RAW_SUFFIX)
    tWriter = rawWriter(tRaw, (2 * PANE_W, PANE_H + BAR_H))
    tTimeline = {}
    for tHand, tD in hands.items():
        tCols = np.linspace(t0, t1, 2 * PANE_W)
        tLabels = tD["lab"][np.searchsorted(tD["tj"], tCols).clip(0, len(tD["tj"]) - 1)]
        tTimeline[tHand] = np.array([COLORS[LABELS[tX]] for tX in tLabels], np.uint8)
    for tT in tTs:
        tFrame = np.vstack([np.hstack([_pane(tHand, hands, tT) for tHand in ("L", "R")]),
                            np.full((BAR_H, 2 * PANE_W, 3), BAR_GRAY, np.uint8)])
        for tRow, tHand in enumerate(("L", "R")):
            tY0 = PANE_H + TIMELINE_TOP + tRow * TIMELINE_ROW_STEP
            tFrame[tY0:tY0 + TIMELINE_ROW_H] = tTimeline[tHand][None]
            cv2.putText(tFrame, tHand, (4, tY0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        tX = int((tT - t0) / (t1 - t0) * (2 * PANE_W - 1))
        cv2.line(tFrame, (tX, PANE_H), (tX, PANE_H + BAR_H), WHITE, 2)
        cv2.putText(tFrame, f"{take} d{demo + 1:02d}  {(tT - t0) / NS_PER_S:5.1f}s", (2 * PANE_W - 330, PANE_H + BAR_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        tWriter.write(tFrame)
    tWriter.release()
    encodeH264(tRaw, outMp4, CRF)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("out_dir")
    tParser.add_argument("demos", nargs="+", help="take/demoNN keys")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out_dir)
    tOut.mkdir(parents=True, exist_ok=True)
    (tOut / "labels").mkdir(exist_ok=True)
    tRows = []
    for tKey in tArgs.demos:
        tTake, tDemo = parseDemoKey(tKey)
        tT0, tT1 = demoWindow(tTake, tDemo)
        tHands = {tHand: labelHand(tTake, tCam, tT0, tT1) for tHand, tCam in HAND_CAMS}
        tCount = {tK: sum(tE["label"] == tK and not tE["rest"] for tD in tHands.values() for tE in tD["events"])
                  for tK in ("held", "empty", "unknown")}
        tName = f"{videoName(tTake, tDemo)}_H{tCount['held']}E{tCount['empty']}U{tCount['unknown']}"
        renderDemo(tOut / f"{tName}.mp4", tTake, tDemo, tT0, tT1, tHands)
        np.savez_compressed(tOut / "labels" / f"{videoName(tTake, tDemo)}.npz", **{f"{tH}_t_ns": tD["tj"] for tH, tD in tHands.items()},
                            **{f"{tH}_label": tD["lab"] for tH, tD in tHands.items()}, label_names=np.array(LABELS))
        for tHand, tD in tHands.items():
            for tK, tE in enumerate(tD["events"]):
                tRows.append(dict(video=tName, event=f"{tHand}{tK + 1}", t_close_s=round((tE["t_close_ns"] - tT0) / NS_PER_S, 1),
                                  **{tX: tE[tX] for tX in tE if not tX.endswith("_ns")}, user_ok=""))
        print(f"{tName}  " + "  ".join(f"{tH}:" + ",".join(tE["label"][0] for tE in tD["events"]) for tH, tD in tHands.items()),
              flush=True)
    with open(tOut / "events.csv", "w", newline="") as tFh:
        tWriter = csv.DictWriter(tFh, fieldnames=list(tRows[0]))
        tWriter.writeheader()
        tWriter.writerows(tRows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
