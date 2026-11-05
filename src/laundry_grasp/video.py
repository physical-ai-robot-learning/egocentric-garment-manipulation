"""Review-video helpers shared by the grasp tools: an mp4v writer for the raw frames, then H.264 by ffmpeg (laundry_common.paths.FFMPEG,
env EGO_FFMPEG) so the clips play in a browser; the raw file is removed."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths as egopaths

VIDEO_FPS = 15.0                      # every grasp review video is real time at 15 fps
RAW_SUFFIX = ".raw.mp4"
BAR_GRAY = 30                         # background of the timeline bar under the panes
CLOSED_BGR = (0, 160, 255)            # orange = jaw closed (motor)
OPEN_BGR = (90, 90, 90)
NO_NEIGHBOUR_NS = 10 ** 10            # no previous open / next close: 10 s away (never limits a clip)


def rawWriter(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, size)


def encodeH264(raw: Path, out: Path, crf: int, faststart: bool = False) -> None:
    """raw mp4v -> H.264 yuv420p at `crf`, then delete the raw file"""
    tCmd = [str(egopaths.FFMPEG), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
            str(crf)]
    if faststart:
        tCmd += ["-movflags", "+faststart"]
    subprocess.run(tCmd + [str(out)], check=True)
    raw.unlink()


def readFrame(path: Path, size: tuple[int, int]) -> np.ndarray:
    """a jpg resized to size (w, h); black when the file is missing"""
    tImage = cv2.imread(str(path))
    if tImage is None:
        return np.zeros((size[1], size[0], 3), np.uint8)
    return cv2.resize(tImage, size)


def singleCloseSpan(events: list[tuple[int, str]], tClose: int, tOpen: int, padS: float, gapS: float, maxS: float
                    ) -> tuple[float, float]:
    """clip span around one close event that holds no other close: [close - padS, open + padS], cut gapS after the previous open
    and gapS before the next close, at most maxS long (11-05 user: "jaw close가 한번만 있게")"""
    tPrevOpen = max([tT for tT, tK in events if tK == "open" and tT < tClose], default=tClose - NO_NEIGHBOUR_NS)
    tNextClose = min([tT for tT, tK in events if tK == "close" and tT > tClose], default=tOpen + NO_NEIGHBOUR_NS)
    tA = max(tClose - padS * 1e9, tPrevOpen + gapS * 1e9)
    tB = min(tOpen + padS * 1e9, tNextClose - gapS * 1e9, tA + maxS * 1e9)
    return tA, tB
