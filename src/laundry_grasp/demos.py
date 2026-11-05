"""Demo windows, demo keys and wrist cameras of the tag-rig takes as the grasp tools see them (was the helper part of
grasp_label_v0 plus the demo-key readers of grasp_v2_events / grasp_pgrasp_gen).

Names used throughout the package:
    demo key     take/demoNN, 0-indexed (split files, LeRobot split.json)                     20261104_150704/demo19
    video name   take_dNN, 1-indexed (review videos, gold event ids)                          20261104_150704_d20
    event id     <video name>_<hand><n>, n = 1-based close event of that hand in the demo     20261104_150704_d20_L5
    hand         L = left hand = wrist_left camera, R = right hand = wrist camera
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from laundry_common import paths as egopaths
from laundry_dataset.takes import FREE_DEMO_PREFIX, loadIdleTrim

HAND_CAMS = (("L", "wrist_left"), ("R", "wrist"))
NS_PER_S = 1e9
DEMO_KEY_DIGITS_AT = len("demo")       # "demoNN"[4:] -> NN
TAKE_FIELDS = 2                        # a take name is <date>_<time>: two "_" fields


class DemoKeyError(ValueError):
    """a demo key / video name that does not follow the naming above"""


def parseDemoKey(key: str) -> tuple[str, int]:
    """take/demoNN -> (take, NN)"""
    try:
        tTake, tDemo = key.split("/")
        return tTake, int(tDemo[DEMO_KEY_DIGITS_AT:])
    except ValueError as tErr:
        raise DemoKeyError(f"not a take/demoNN key: {key!r}") from tErr


def videoName(take: str, demo: int) -> str:
    """(take, 0-based demo) -> take_dNN (1-based)"""
    return f"{take}_d{demo + 1:02d}"


def parseVideoName(video: str) -> tuple[str, int]:
    """take_dNN[_anything] -> (take, 0-based demo)"""
    tParts = video.split("_")
    return "_".join(tParts[:TAKE_FIELDS]), int(tParts[TAKE_FIELDS][1:]) - 1


def handCam(hand: str) -> str:
    """hand letter (or an event name such as 'L3') -> wrist camera directory name"""
    return dict(HAND_CAMS)[hand[0]]


def camDir(take: str, cam: str) -> Path:
    return egopaths.TAG_RIG / take / cam


def wristImage(cameraDir: Path, index: int) -> Path:
    return cameraDir / f"wrist_{index:05d}.jpg"


def nearestFrame(wristTimes: np.ndarray, tNs) -> int:
    """index of the first wrist frame at or after tNs, clipped to the recording (the frame shown at time tNs)"""
    return int(np.searchsorted(wristTimes, tNs).clip(0, len(wristTimes) - 1))


def demoWindow(take: str, demo: int, tagRig: Path | None = None, trimCsv: Path | None = None) -> tuple[int, int]:
    """[t0, t1] monotonic ns of free demo `demo` (0-based) of a take: the protocol window, cut to the idle-trim keep window when
    idle_trim.csv has a row for it."""
    tTagRig = egopaths.TAG_RIG if tagRig is None else tagRig
    tTrimCsv = egopaths.IDLE_TRIM_CSV if trimCsv is None else trimCsv
    tMeta = json.loads((tTagRig / take / "meta.json").read_text())
    tStart = int(tMeta["t_start_monotonic_ns"])
    tProtocol = [tP for tP in tMeta["protocol"] if tP["text"].startswith(FREE_DEMO_PREFIX)][demo]
    tTrim = loadIdleTrim(tTrimCsv).get((take, demo))
    tA = tStart + int(tProtocol["t0"] * NS_PER_S)
    if tTrim:
        return tA + int(float(tTrim["keep_start_s"]) * NS_PER_S), tA + int(float(tTrim["keep_end_s"]) * NS_PER_S)
    return tA, tA + int(tProtocol["dur"] * NS_PER_S)


def splitDocKeys(splitDir: Path, patterns: Iterable[str], keepSplit: Callable[[str], bool]) -> set[str]:
    """demo keys of the per-take split docs (<set>_split.json: {"episodes": {split: [keys]}}) matching the glob patterns,
    for the splits `keepSplit` accepts"""
    tKeys: set[str] = set()
    for tPattern in patterns:
        for tFile in sorted(Path(splitDir).glob(tPattern)):
            for tSplit, tSplitKeys in json.loads(tFile.read_text())["episodes"].items():
                if keepSplit(tSplit):
                    tKeys |= set(tSplitKeys)
    return tKeys
