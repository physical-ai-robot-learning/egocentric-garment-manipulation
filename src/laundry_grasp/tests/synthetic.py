"""A tiny synthetic tag-rig take for the laundry_grasp tests: meta.json with two free demos, both wrist cams with t_ns.npy, a
umi_jaw.npz (t_ns, g_cont, state, ev_t, ev_kind) and 1280 x 720 jpgs with a dark 'jaw' block, plus an idle_trim.csv."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

TAKE = "20990101_120000"
T_START_NS = 1_000_000_000_000
JAW_HZ = 100.0
WRIST_HZ = 30.0
DURATION_S = 12.0
NS_PER_S = 1_000_000_000
# g_cont profile per hand: (start s, end s, value); default fully open (1.0)
CLOSES = {"wrist": [(1.0, 3.0, 0.0), (3.0, 3.5, 0.2), (3.5, 5.0, 0.0), (7.0, 9.0, 0.05)],
          "wrist_left": [(2.0, 6.0, 0.1)]}
JAW_BLOCK = (slice(380, 720), slice(560, 720))   # rows, cols of the dark jaw in the camera frame


def gCont(cam: str, tS: np.ndarray) -> np.ndarray:
    tG = np.ones(len(tS), np.float32)
    for tA, tB, tV in CLOSES[cam]:
        tG[(tS >= tA) & (tS < tB)] = tV
    return tG


def writeTake(root: Path, nFrames: int | None = None) -> Path:
    """root/<TAKE>/... ; returns the take dir. nFrames limits the jpgs written per camera (all frames when None)."""
    tTake = root / TAKE
    tTake.mkdir(parents=True)
    tProtocol = [dict(text="준비", t0=0.0, dur=0.5), dict(text="자유: 펼치기", t0=0.5, dur=5.0), dict(text="자유: 펼치기", t0=6.0, dur=5.5)]
    (tTake / "meta.json").write_text(json.dumps(dict(t_start_monotonic_ns=T_START_NS, protocol=tProtocol), ensure_ascii=False))
    for tCam in ("wrist", "wrist_left"):
        tDir = tTake / tCam
        tDir.mkdir()
        tJs = np.arange(0, DURATION_S, 1 / JAW_HZ)
        tJawT = (T_START_NS + tJs * NS_PER_S).astype(np.int64)
        tG = gCont(tCam, tJs)
        tState = (tG > 0.5).astype(np.int64)
        tEvents = []
        for tI in range(1, len(tState)):
            if tState[tI] != tState[tI - 1]:
                tEvents.append((int(tJawT[tI]), "open" if tState[tI] else "close"))
        np.savez(tDir / "umi_jaw.npz", t_ns=tJawT, g_cont=tG, state=tState, ev_t=np.array([tE[0] for tE in tEvents], np.int64),
                 ev_kind=np.array([tE[1] for tE in tEvents]))
        tWs = np.arange(0, DURATION_S, 1 / WRIST_HZ)
        tWristT = (T_START_NS + tWs * NS_PER_S).astype(np.int64)
        np.save(tDir / "t_ns.npy", tWristT)
        for tI in range(len(tWristT) if nFrames is None else nFrames):
            tImage = np.full((720, 1280, 3), 230, np.uint8)
            tImage[JAW_BLOCK] = 10
            tImage[:40, :40] = tI % 255
            cv2.imwrite(str(tDir / f"wrist_{tI:05d}.jpg"), tImage)
    return tTake


def writeTrim(path: Path, demo: int = 1, keepStartS: float = 0.5, keepEndS: float = 5.0) -> Path:
    path.write_text("take,demo,demo_len_s,keep_start_s,keep_end_s\n"
                    f"{TAKE},{demo},5.5,{keepStartS},{keepEndS}\n")
    return path
