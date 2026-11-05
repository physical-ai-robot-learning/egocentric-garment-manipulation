"""Characterization of laundry_jaw.motorlike_overlay: label sampling, the label strip image, and the demo videos + index.json."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_jaw import motorlike_overlay
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

MODULE = "laundry_jaw.motorlike_overlay"
N_FRAMES = 40
FRAME_NS = 50_000_000
T0_NS = 3_000_000_000


def _take(root: Path, name: str, withNew: bool) -> Path:
    tTake = root / name
    tT = (T0_NS + np.arange(N_FRAMES) * FRAME_NS).astype(np.int64)
    for tHand, tDirName in motorlike_overlay.WRIST_DIRS.items():
        tDir = tTake / tDirName
        tDir.mkdir(parents=True)
        np.save(tDir / "t_ns.npy", tT)
        for tK in range(N_FRAMES):
            if tK != 7:
                cv2.imwrite(str(tDir / f"wrist_{tK:05d}.jpg"), np.full((90, 160, 3), 3 * tK + (40 if tHand == "left" else 0), np.uint8))
        tOld = ((np.arange(N_FRAMES) // 9) % 2).astype(np.int64)
        np.savez(tDir / "umi_jaw.npz", t_ns=tT, state=tOld)
        if withNew:
            tNew = ((np.arange(N_FRAMES) + 4) // 11 % 2).astype(np.int64)
            np.savez(tDir / "umi_jaw_motorlike.npz", t_ns=tT, state=tNew, aperture=np.linspace(0, 1, N_FRAMES).astype(np.float32))
    tSegs = [{"out": [T0_NS, T0_NS + 900_000_000]},
             {"hold_pre": [T0_NS + 200_000_000, T0_NS + 300_000_000], "out": [0, 0], "back": [0, T0_NS + 1_600_000_000]},
             {"hold_pre": [T0_NS + 500_000_000, 0], "hold_post": [0, T0_NS + 1_400_000_000], "out": [0, 0]}]
    (tTake / "segments.json").write_text(json.dumps(tSegs))
    return tTake


def test_label_helpers_and_strip():
    tZ = {"t_ns": np.array([0, 10, 20]), "state": np.array([0, 1, 1])}
    tStrips = {f"{tOld}_{tNew}_{tAp}": motorlike_overlay.strip(tOld, tNew, tAp) for tOld, tNew, tAp in ((0.0, 1.0, 0.3), (1.0, 0.0, 0.9))}
    checkSnapshot("overlay.helpers", {"at": motorlike_overlay.at(tZ, "state", np.array([5.0, 15.0, 30.0])),
                                      "strips": {tKey: [tImage.shape, int(tImage.astype(np.int64).sum()),
                                                        int((tImage * np.arange(1, 4)).sum() % 1_000_003)]
                                                 for tKey, tImage in tStrips.items()},
                                      "W": motorlike_overlay.W, "H": motorlike_overlay.H})


def test_lab_missing_returns_none(tmp_path):
    assert motorlike_overlay.lab(tmp_path, "umi_jaw.npz") is None


def test_main_writes_videos_and_index(tmp_path, monkeypatch):
    tA = _take(tmp_path, "20261026_150000", withNew=True)
    tB = _take(tmp_path, "20261026_160000", withNew=False)
    tKeys = tmp_path / "keys.txt"
    tKeys.write_text("20261026_150000/demo00\n20261026_150000/demo02 20261026_160000/demo01\n")
    tOutDir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["overlay", str(tOutDir), str(tA), str(tB), "--n", "3", "--seed", "1", "--fps", "8",
                                      "--keys", str(tKeys)])
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        motorlike_overlay.main()
    tFrames = {}
    for tVideo in sorted(tOutDir.glob("*.mp4")):
        tCap = cv2.VideoCapture(str(tVideo))
        tFrames[tVideo.name] = int(tCap.get(cv2.CAP_PROP_FRAME_COUNT))
        tCap.release()
    checkSnapshot("overlay.main", {"index": json.loads((tOutDir / "index.json").read_text()), "frames": tFrames, "stdout": tOut.getvalue()})


def test_cli_defaults(monkeypatch):
    checkSnapshot("overlay.cli", parsedArgs(monkeypatch, MODULE, ["out", "takeA"]))
    assert helpExitCode(monkeypatch, MODULE, []) == 0
