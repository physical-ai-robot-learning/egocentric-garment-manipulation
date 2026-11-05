"""Characterization of laundry_jaw.motorlike: take discovery, crops / servo target, build, train (1 CPU epoch), predict, event
matching, eval and apply, on synthetic takes with the shipped v003 classifier standing in for a trained motor-like model."""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from laundry_common.paths import CONFIGS
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from laundry_jaw import motorlike  # noqa: E402  needs laundry_shared umi_jaw (torch stack)
from laundry_shared.visual_teleop.tools import umi_jaw  # noqa: E402

MODULE = "laundry_jaw.motorlike"
V003 = {tHand: CONFIGS / "calibration" / f"umi_jaw_resnet18_v003_{tHand}.pt" for tHand in ("right", "left")}
N_FRAMES = 30
FRAME_NS = 33_333_333
# torch CPU inference / training on several threads may reorder sums.
TORCH_REL = 1e-4
TORCH_ABS = 1e-5
# One CPU epoch from the v003 weights: losses / held-out metrics as printed (4 decimals).
TRAIN_REL = 1e-3
TRAIN_ABS = 2e-4
TAKES = {"20261029_100000": ("crops", "feetech_sts3215_motor"), "20261030_110000": ("jpg", "feetech_sts3215_motor"),
         "20261028_090000": ("crops", "feetech_sts3215_motor"), "20261031_120000": ("crops", "resnet18_v003")}


def _frame(k: int, hand: str) -> np.ndarray:
    tImage = np.full((120, 160, 3), 170, np.uint8)
    tGap = 10 + int(30 * (0.5 + 0.5 * np.sin(k / 4.0 + (0.7 if hand == "left" else 0.0))))
    cv2.rectangle(tImage, (80 - tGap - 12, 80), (80 - tGap, 119), (15, 15, 15), -1)
    cv2.rectangle(tImage, (80 + tGap, 80), (80 + tGap + 12, 119), (15, 15, 15), -1)
    if k % 10 == 9:
        tImage[:] = 5                                    # cloth over the lens: dark crop
    return tImage


def _makeTake(root: Path, name: str, cropSource: str, model: str) -> Path:
    tTake = root / name
    tT = (np.arange(N_FRAMES) * FRAME_NS + 2_000_000_000).astype(np.int64)
    for tHand, tDirName in motorlike.WRIST_DIRS.items():
        tDir = tTake / tDirName
        tDir.mkdir(parents=True)
        np.save(tDir / "t_ns.npy", tT)
        tImages = [_frame(tK, tHand) for tK in range(N_FRAMES)]
        for tK, tImage in enumerate(tImages):
            cv2.imwrite(str(tDir / f"wrist_{tK:05d}.jpg"), tImage)
        if cropSource == "crops":
            np.savez(tDir / "crops.npz", crops=np.stack([umi_jaw.crop(tImage, umi_jaw.ROIS[tHand]) for tImage in tImages]))
        tG = (0.5 + 0.5 * np.sin(np.arange(N_FRAMES) / 4.0 + (0.7 if tHand == "left" else 0.0))) * motorlike.OPEN_PLATEAU[tHand]
        tState = (tG > 0.3).astype(np.int64)
        np.savez(tDir / "umi_jaw.npz", t_ns=tT, g_cont=tG.astype(np.float32), valid=(np.arange(N_FRAMES) % 8 != 7).astype(np.int8),
                 state=tState, ev_t=tT[np.flatnonzero(np.diff(tState)) + 1], model=model)
    return tTake


@pytest.fixture
def tagRig(tmp_path, monkeypatch):
    tRoot = tmp_path / "tag_rig"
    for tName, (tSource, tModel) in TAKES.items():
        _makeTake(tRoot, tName, tSource, tModel)
    monkeypatch.setattr(motorlike, "TAG_RIG", tRoot)
    monkeypatch.setattr(umi_jaw, "MODELS", dict(V003))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tRoot


def _quiet(fn, *args, **kw):
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        tResult = fn(*args, **kw)
    return tResult, tOut.getvalue()


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as tZ:
        return {tKey: {"dtype": str(tZ[tKey].dtype), "value": tZ[tKey]} for tKey in sorted(tZ.files)}


def test_motor_takes_and_targets(tagRig):
    tTakes = [tPath.name for tPath in motorlike.motorTakes()]
    tWrist = tagRig / "20261029_100000" / "wrist"
    tTarget, tValid = motorlike.servoTarget(tWrist, "right", np.load(tWrist / "t_ns.npy")[::2] + 7_000_000)
    tCropsNpz = motorlike.loadCrops(tWrist, "right", 12)
    tCropsJpg = motorlike.loadCrops(tagRig / "20261030_110000" / "wrist_left", "left", 12)
    checkSnapshot("motorlike.discovery", {"takes": tTakes, "target": tTarget, "target_dtype": str(tTarget.dtype), "valid": tValid,
                                          "crops_npz": [tCropsNpz.shape, int(tCropsNpz.sum())],
                                          "crops_jpg": [tCropsJpg.shape, int(tCropsJpg.astype(np.int64).sum())]})


def test_match_events():
    tRef = [(1_000_000_000, "open"), (2_000_000_000, "close"), (3_500_000_000, "open"), (9_000_000_000, "close")]
    tEv = [(1_100_000_000, "open"), (1_900_000_000, "close"), (2_050_000_000, "close"), (4_600_000_000, "open"), (8_500_000_000, "open")]
    checkSnapshot("motorlike.match", {"default": motorlike.matchEvents(tRef, tEv), "tol2": motorlike.matchEvents(tRef, tEv, tol=2.0),
                                      "empty": motorlike.matchEvents([], [])})


def test_build_train_eval_apply(tagRig, tmp_path):
    tOut = tmp_path / "out"
    _, tBuildStdout = _quiet(motorlike.build, tOut)
    tBuilt = {tHand: _npz(tOut / f"{tHand}.npz") for tHand in ("left", "right")}
    _, tTrainStdout = _quiet(motorlike.train, tOut, "left", ["20261030_110000"], "", 1)
    tTrained = sorted(tPath.name for tPath in tOut.glob("*.pt"))
    tCrops = np.load(tagRig / "20261029_100000" / "wrist" / "crops.npz")["crops"][:9]
    tP1 = motorlike.predict(V003["right"], tCrops)
    tP2, tAp = motorlike.predict(V003["right"], tCrops, both=True)
    for tHand in ("left", "right"):
        shutil.copy(V003[tHand], tOut / f"jaw_motorlike_{tHand}.pt")
    _, tEvalStdout = _quiet(motorlike.evaluate, tOut, "right", "", ["20261029_100000", "20261030_110000"])
    tEval = json.loads((tOut / "eval_right.json").read_text())
    tTake = tagRig / "20261031_120000"
    _, tApplyStdout = _quiet(motorlike.applyModel, tOut, [str(tTake), str(tmp_path / "missing_take")])
    tApplied = {tDir: _npz(tTake / tDir / "umi_jaw_motorlike.npz") for tDir in ("wrist", "wrist_left")}
    checkSnapshot("motorlike.build", {"stdout": tBuildStdout, "built": tBuilt})
    checkSnapshot("motorlike.train", {"models": tTrained, "stdout_lines": len(tTrainStdout.splitlines()),
                                      "stdout_head": tTrainStdout.splitlines()[0]})
    checkSnapshot("motorlike.train.numbers", [float(tX) for tX in re.findall(r"\d+\.\d+", tTrainStdout)], rel=TRAIN_REL, abs_=TRAIN_ABS)
    checkSnapshot("motorlike.predict", {"p": tP1, "p_both": tP2, "aperture": tAp}, rel=TORCH_REL, abs_=TORCH_ABS)
    checkSnapshot("motorlike.evaluate", {"json": tEval, "stdout": tEvalStdout}, rel=TORCH_REL, abs_=TORCH_ABS)
    checkSnapshot("motorlike.apply", {"npz": tApplied, "stdout": tApplyStdout.replace(str(tmp_path), "<tmp>")},
                  rel=TORCH_REL, abs_=TORCH_ABS)


@pytest.mark.parametrize("argv", [["build", "o"], ["train", "o", "--hand", "left"], ["eval", "o", "--hand", "right", "t1", "t2"],
                                  ["apply", "o", "t1"]])
def test_cli_defaults(monkeypatch, argv):
    checkSnapshot("motorlike.cli." + argv[0], parsedArgs(monkeypatch, MODULE, argv))


def test_cli_help(monkeypatch):
    assert helpExitCode(monkeypatch, MODULE, []) == 0
