"""Characterization of laundry_jaw.aperture: Otsu threshold, --calibrate, and the per-frame aperture written into umi_jaw.npz
(ultralytics replaced by a deterministic fake YOLO, so the code path around the detector runs without the model)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from laundry_jaw import aperture
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

MODULE = "laundry_jaw.aperture"
N_FRAMES = 45
FRAME_NS = 33_333_333
MISSING_FRAME = 10
FRAME_SHAPE = (120, 160, 3)


class _Tensor:
    """Just enough of a torch tensor for the detector code: .cpu().numpy(), indexing, float()."""

    def __init__(self, array) -> None:
        self.mArray = np.asarray(array, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.mArray

    def __getitem__(self, index):
        return _Tensor(self.mArray[index])

    def __float__(self) -> float:
        return float(self.mArray)

    def __len__(self) -> int:
        return len(self.mArray)


class _FakeYolo:
    """Call k: no detection every 9th call, a low-confidence one every 7th, else two boxes whose best one has tips d px apart."""
    mModelPaths: list = []

    def __init__(self, path: str) -> None:
        _FakeYolo.mModelPaths.append(Path(path).name)
        self.mCalls = 0


class _Boxes:
    def __init__(self, conf) -> None:
        self.conf = _Tensor(conf)

    def __len__(self) -> int:
        return len(self.conf)


def _fakePredict(self, image, verbose=False, device=None):
    tK = self.mCalls
    self.mCalls += 1
    if tK % 9 == 4:
        return [types.SimpleNamespace(keypoints=None, boxes=_Boxes(np.zeros(0)))]
    tConf = 0.1 if tK % 7 == 2 else 0.5 + 0.01 * (tK % 5)
    tD = 90.0 + 120.0 * (0.5 + 0.5 * np.sin(tK / 5.0))
    tTips = np.array([[[50.0, 60.0], [50.0 + tD, 60.0]], [[0.0, 0.0], [1.0, 1.0]]])
    return [types.SimpleNamespace(keypoints=types.SimpleNamespace(xy=_Tensor(tTips)), boxes=_Boxes([tConf, tConf / 2]))]


_FakeYolo.predict = _fakePredict


@pytest.fixture
def fakeUltralytics(monkeypatch):
    _FakeYolo.mModelPaths = []
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYolo))
    return _FakeYolo


def _wristDir(root: Path, hand: str, withClassifier: bool) -> Path:
    tDir = root / ("wrist" if hand == "right" else "wrist_left")
    tDir.mkdir(parents=True)
    (tDir / "meta.json").write_text(json.dumps({"hand": hand}))
    tT = (np.arange(N_FRAMES) * FRAME_NS + 1_000_000_000).astype(np.int64)
    np.save(tDir / "t_ns.npy", tT)
    for tK in range(N_FRAMES):
        if tK == MISSING_FRAME:
            continue
        tImage = np.full(FRAME_SHAPE, 40 + tK, np.uint8)
        cv2.imwrite(str(tDir / f"wrist_{tK:05d}.jpg"), tImage)
    if withClassifier:
        tP = 0.5 + 0.5 * np.cos(np.arange(N_FRAMES) / 4.0)
        np.savez(tDir / "umi_jaw.npz", t_ns=tT, p_open=tP.astype(np.float32), state=(tP > 0.5).astype(np.int64))
    return tDir


def _npzSnapshot(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as tZ:
        return {tKey: {"dtype": str(tZ[tKey].dtype), "value": tZ[tKey]} for tKey in sorted(tZ.files)}


def test_otsu_threshold():
    tRng = np.random.default_rng(5)
    tX = np.concatenate([tRng.normal(100, 8, 200), tRng.normal(220, 15, 120)])
    checkSnapshot("aperture.otsu", aperture.otsu(tX))


def test_calibrate_writes_json(tmp_path, monkeypatch):
    tRng = np.random.default_rng(6)
    for tHand in ("left", "right"):
        tLabels = []
        for tK in range(40):
            tD = float(tRng.normal(110, 6) if tK % 3 else tRng.normal(260, 12))
            tLabels.append({"visible": tK % 11 != 0, "tips": [[10.0, 20.0], [10.0 + tD, 25.0]] if tK % 13 else None})
        (tmp_path / "labels" / tHand).mkdir(parents=True)
        (tmp_path / "labels" / tHand / "labels.json").write_text(json.dumps({"labels": tLabels}))
    tCal = tmp_path / "cal.json"
    monkeypatch.setattr(aperture, "CALIBRATION_PATH", tCal)
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        aperture.calibrate(tmp_path / "labels")
    checkSnapshot("aperture.calibrate", {"json": json.loads(tCal.read_text()).__repr__().replace(str(tmp_path), "<tmp>"),
                                         "stdout": tOut.getvalue().replace(str(tmp_path), "<tmp>")})


@pytest.mark.parametrize("hand,withClassifier", [("right", True), ("left", False)])
def test_run_detects_then_rescores(tmp_path, fakeUltralytics, hand, withClassifier):
    tDir = _wristDir(tmp_path, hand, withClassifier)
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        aperture.measureAperture(tDir, 0.3, 0.2, "cpu", False)
    tFirst = _npzSnapshot(tDir / "umi_jaw.npz")
    with contextlib.redirect_stdout(tOut):
        aperture.measureAperture(tDir, 0.3, 0.2, "cpu", True)
    tSecond = _npzSnapshot(tDir / "umi_jaw.npz")
    checkSnapshot(f"aperture.run.{hand}", {"first": tFirst, "rescored": tSecond, "models": fakeUltralytics.mModelPaths,
                                           "stdout": tOut.getvalue()})


def test_run_without_detections_exits(tmp_path, monkeypatch):
    class _NoDetection(_FakeYolo):
        def predict(self, image, verbose=False, device=None):
            return [types.SimpleNamespace(keypoints=None, boxes=_Boxes(np.zeros(0)))]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_NoDetection))
    tDir = _wristDir(tmp_path, "left", False)
    with pytest.raises(aperture.ApertureError) as tExit:
        aperture.measureAperture(tDir, 0.3, 0.2, "cpu", False)
    checkSnapshot("aperture.run.no_detection", str(tExit.value))


def test_calibration_file_and_models_resolve():
    tCal = json.loads(aperture.CALIBRATION_PATH.read_text())
    checkSnapshot("aperture.cal_file", {"name": aperture.CALIBRATION_PATH.name, "hands": sorted(tCal)})


def test_cli_defaults(monkeypatch):
    checkSnapshot("aperture.cli", parsedArgs(monkeypatch, MODULE, ["some/wrist"]))
    assert helpExitCode(monkeypatch, MODULE, []) == 0


def test_calibration_models_resolve_in_repository():
    """The calibration json names its models as "configs/calibration/..." (recording tree); they ship under CONFIGS."""
    tCal = json.loads(aperture.CALIBRATION_PATH.read_text())
    for tHand in ("left", "right"):
        assert aperture.modelPath(tCal[tHand]["model"]).is_file()
