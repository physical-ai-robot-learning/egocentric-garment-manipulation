"""Human check of the motor-like jaw relabel (2026-10-30; was scripts/pose/jaw_motorlike_overlay.py).

Pipeline stage: jaw state (stage 2), QA of `motorlike apply`. For random demos of camera-only takes, a video with both wrist views and,
under each, the OLD label (umi_jaw.npz state, wrist-camera classifier v003) and the NEW one (umi_jaw_motorlike.npz state + aperture).
Event frames get a coloured border (green = opens, red = closes) and a hold of a few frames so they can be checked one by one.
Output: <out_dir>/<take>_demoNN.mp4 per demo and <out_dir>/index.json (event counts per hand, old and new).
    python -m laundry_jaw.motorlike_overlay <out_dir> --n 20 --seed 0 TAKE_DIR...      (demo windows from <take>/segments.json)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WRIST_DIRS = {"left": "wrist_left", "right": "wrist"}
# Size of each wrist pane in the video.
W, H = 480, 270
STRIP_H = 44
STRIP_ROW_H = 21
STRIP_LABEL_W = 90
OPEN_COLOUR = (80, 200, 80)
CLOSED_COLOUR = (60, 60, 160)
APERTURE_COLOUR = (0, 255, 255)
TEXT_COLOUR = (255, 255, 255)
EVENT_OPEN_COLOUR = (0, 255, 0)
EVENT_CLOSE_COLOUR = (0, 0, 255)
EVENT_BORDER_PX = 8
# An event frame is written this many times so it can be checked by eye.
EVENT_HOLD_FRAMES = 4
# A 0/1 state (interpolated between label samples) counts as open above this.
BINARY_HALF = 0.5
NS_PER_S = 1e9
DEFAULT_N = 20
DEFAULT_SEED = 0
DEFAULT_FPS = 10.0


def lab(wristDir: Path, name: str):
    """The label file wristDir/name, or None when it does not exist."""
    tFile = wristDir / name
    if not tFile.exists():
        return None
    return np.load(tFile, allow_pickle=True)


def at(label, key: str, tNs):
    """label[key] linearly interpolated at `tNs`."""
    return np.interp(tNs, label["t_ns"].astype(np.float64), label[key].astype(np.float64))


def strip(old: float, new: float, aperture: float, width: int = W) -> np.ndarray:
    """2 rows: OLD state / NEW state, plus the aperture line on the NEW row (open = light, closed = dark)."""
    tImage = np.zeros((STRIP_H, width, 3), np.uint8)
    for tRow, (tValue, tName) in enumerate(((old, "old v003"), (new, "new motorlike"))):
        tColour = OPEN_COLOUR if tValue > BINARY_HALF else CLOSED_COLOUR
        cv2.rectangle(tImage, (STRIP_LABEL_W, 3 + tRow * STRIP_ROW_H), (width - 4, 20 + tRow * STRIP_ROW_H), tColour, -1)
        cv2.putText(tImage, f"{tName}", (2, 16 + tRow * STRIP_ROW_H), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_COLOUR, 1)
        cv2.putText(tImage, "OPEN" if tValue > BINARY_HALF else "closed", (100, 16 + tRow * STRIP_ROW_H), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    TEXT_COLOUR, 1)
    tX = int(STRIP_LABEL_W + aperture * (width - 94))
    cv2.line(tImage, (tX, 24), (tX, 41), APERTURE_COLOUR, 2)
    return tImage


def _demoWindows(takes: list[Path], keys: set[str] | None) -> list[tuple[Path, int, int, int]]:
    """(take, demo index, t0 ns, t1 ns) of every segments.json entry (only the listed take/demoNN keys when `keys` is given)."""
    tDemos = []
    for tTake in takes:
        tSegments = json.loads((tTake / "segments.json").read_text())
        for tI, tSeg in enumerate(tSegments):
            if keys is not None and f"{tTake.name}/demo{tI:02d}" not in keys:
                continue   # 10-31: segments.json also lists cues never performed / dropped
            tT0 = (tSeg.get("hold_pre") or tSeg["out"])[0]
            tT1 = (tSeg.get("hold_post") or tSeg.get("back") or tSeg["out"])[1]
            tDemos.append((tTake, tI, tT0, tT1))
    return tDemos


def writeDemoVideo(take: Path, demo: int, t0: int, t1: int, outDir: Path, fps: float) -> dict:
    """outDir/<take>_demoNN.mp4 of [t0, t1); returns its index.json row."""
    tViews = {tHand: (take / tDirName, np.load(take / tDirName / "t_ns.npy"), lab(take / tDirName, "umi_jaw.npz"),
                      lab(take / tDirName, "umi_jaw_motorlike.npz")) for tHand, tDirName in WRIST_DIRS.items()}
    tName = f"{take.name}_demo{demo:02d}.mp4"
    tWriter = cv2.VideoWriter(str(outDir / tName), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H + STRIP_H))
    tPrevious = {tHand: None for tHand in WRIST_DIRS}
    tEventCounts = {tHand: [0, 0] for tHand in WRIST_DIRS}
    for tT in np.arange(t0, t1, NS_PER_S / fps):
        tPanes = []
        tHold = 1
        for tHand in ("left", "right"):
            tWrist, tTimes, tOld, tNew = tViews[tHand]
            tK = int(np.clip(np.searchsorted(tTimes, tT), 0, len(tTimes) - 1))
            tImage = cv2.imread(str(tWrist / f"wrist_{tK:05d}.jpg"))
            tImage = cv2.resize(tImage, (W, H)) if tImage is not None else np.zeros((H, W, 3), np.uint8)
            tOldState = at(tOld, "state", tT) if tOld is not None else 0
            tNewState = at(tNew, "state", tT) if tNew is not None else 0
            tAperture = at(tNew, "aperture", tT) if tNew is not None and "aperture" in tNew.files else 0
            if tPrevious[tHand] is not None and (tNewState > BINARY_HALF) != (tPrevious[tHand] > BINARY_HALF):
                tBorder = EVENT_OPEN_COLOUR if tNewState > BINARY_HALF else EVENT_CLOSE_COLOUR
                cv2.rectangle(tImage, (0, 0), (W - 1, H - 1), tBorder, EVENT_BORDER_PX)
                tHold = EVENT_HOLD_FRAMES
                tEventCounts[tHand][0] += 1
            if tPrevious[tHand] is not None and (tOldState > BINARY_HALF) != (tPrevious[tHand + "_o"] > BINARY_HALF):
                tEventCounts[tHand][1] += 1
            tPrevious[tHand], tPrevious[tHand + "_o"] = tNewState, tOldState
            cv2.putText(tImage, f"{tHand} {(tT - t0) / NS_PER_S:5.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, APERTURE_COLOUR, 2)
            tPanes.append(np.vstack([tImage, strip(tOldState, tNewState, float(tAperture))]))
        tFrame = np.hstack(tPanes)
        for _ in range(tHold):
            tWriter.write(tFrame)
    tWriter.release()
    return dict(video=tName, take=take.name, demo=demo, s=round((t1 - t0) / NS_PER_S, 1),
                new_events_L=tEventCounts["left"][0], old_events_L=tEventCounts["left"][1],
                new_events_R=tEventCounts["right"][0], old_events_R=tEventCounts["right"][1])


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("out")
    tParser.add_argument("takes", nargs="+")
    tParser.add_argument("--n", type=int, default=DEFAULT_N)
    tParser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tParser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    tParser.add_argument("--keys", default="",
                         help="file with take/demoNN keys to draw from (e.g. the GOLD-281 list); default = every segments.json entry")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tKeys = set(Path(tArgs.keys).read_text().split()) if tArgs.keys else None
    tOutDir = Path(tArgs.out)
    tOutDir.mkdir(parents=True, exist_ok=True)
    tDemos = _demoWindows([Path(tTake) for tTake in tArgs.takes], tKeys)
    random.Random(tArgs.seed).shuffle(tDemos)
    tRows = []
    for tTake, tDemo, tT0, tT1 in tDemos[:tArgs.n]:
        tRows.append(writeDemoVideo(tTake, tDemo, tT0, tT1, tOutDir, tArgs.fps))
        print(tRows[-1], flush=True)
    (tOutDir / "index.json").write_text(json.dumps(tRows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
