"""Characterization of laundry_jaw.kpt_label: crop, tip drafting, `select` output and the `serve` labelling routes."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from laundry_jaw import kpt_label
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

MODULE = "laundry_jaw.kpt_label"
FRAME_H, FRAME_W = 200, 320
N_FRAMES = 60
FRAME_NS = 33_333_333
T_START_NS = 5_000_000_000


def _jawImage(gapPx: int, merged: bool = False, empty: bool = False) -> np.ndarray:
    """Bright frame with the two dark jaws touching the bottom edge of the crop (or one merged blob / none)."""
    tImage = np.full((FRAME_H, FRAME_W, 3), 200, np.uint8)
    if empty:
        return tImage
    tY0 = int(FRAME_H * 0.70)
    tCx = FRAME_W // 2
    if merged:
        cv2.rectangle(tImage, (tCx - 40, tY0), (tCx + 40, FRAME_H - 1), (20, 20, 20), -1)
        return tImage
    cv2.rectangle(tImage, (tCx - gapPx // 2 - 30, tY0), (tCx - gapPx // 2, FRAME_H - 1), (20, 20, 20), -1)
    cv2.rectangle(tImage, (tCx + gapPx // 2, tY0 + 6), (tCx + gapPx // 2 + 30, FRAME_H - 1), (25, 25, 25), -1)
    return tImage


def _take(root: Path) -> Path:
    tTake = root / "20261022_999999"
    tTake.mkdir()
    tMeta = {"t_start_monotonic_ns": T_START_NS, "protocol": [
        {"text": "자유: 아무거나", "t0": 0.1, "dur": 1.2},
        {"text": "멈추세요", "t0": 1.3, "dur": 0.2},
        {"text": "자유: 다시", "t0": 1.5, "dur": 0.4}]}
    (tTake / "meta.json").write_text(json.dumps(tMeta, ensure_ascii=False))
    tT = (T_START_NS + np.arange(N_FRAMES) * FRAME_NS).astype(np.int64)
    for tHand, tDirName in kpt_label.WRIST_DIRS.items():
        tDir = tTake / tDirName
        tDir.mkdir()
        np.save(tDir / "t_ns.npy", tT)
        for tK in range(N_FRAMES):
            cv2.imwrite(str(tDir / f"wrist_{tK:05d}.jpg"), _jawImage(20 + 2 * tK, merged=tK % 17 == 3, empty=tK % 19 == 5))
        if tHand == "left":
            tP = 0.5 + 0.5 * np.sin(np.arange(N_FRAMES) / 6.0)
            np.savez(tDir / "umi_jaw.npz", t_ns=tT, p_open=tP)
    return tTake


def test_crop_window():
    tImage = np.arange(FRAME_H * FRAME_W * 3, dtype=np.int64).reshape(FRAME_H, FRAME_W, 3)
    tCrop = kpt_label.crop(tImage)
    checkSnapshot("kpt_label.crop", {"shape": tCrop.shape, "sum": int(tCrop.sum()), "corner": tCrop[0, 0]})


def test_draft_tips_cases():
    tCases = {"apart": _jawImage(60), "near": _jawImage(10), "merged": _jawImage(0, merged=True), "empty": _jawImage(0, empty=True)}
    checkSnapshot("kpt_label.draft_tips", {tName: kpt_label.draftTips(kpt_label.crop(tImage)) for tName, tImage in tCases.items()})


@pytest.fixture
def selected(tmp_path, monkeypatch):
    tTake = _take(tmp_path)
    monkeypatch.setattr(kpt_label, "JAW_KPT_DIR", tmp_path / "data" / "datasets" / "jaw_kpt")
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        kpt_label.select(tTake, {"left": 12, "right": 7})
    return tmp_path / "data" / "datasets" / "jaw_kpt" / tTake.name, tOut.getvalue().replace(str(tmp_path), "<tmp>")


def test_select_writes_frames_and_labels(selected):
    tOutDir, tStdout = selected
    tResult = {"stdout": tStdout}
    for tHand in ("left", "right"):
        tResult[tHand] = {"labels": json.loads((tOutDir / tHand / "labels.json").read_text()),
                          "frames": sorted(tPath.name for tPath in (tOutDir / tHand / "frames").iterdir())}
    checkSnapshot("kpt_label.select", tResult)


def test_serve_routes(selected, monkeypatch):
    flask = pytest.importorskip("flask")
    tOutDir, _ = selected
    tApps = []
    monkeypatch.setattr(flask.Flask, "run", lambda self, **kw: tApps.append((self, kw)))
    with contextlib.redirect_stdout(io.StringIO()) as tOut:
        kpt_label.serve(tOutDir, 8799)
    tApp, tRunKw = tApps[0]
    tClient = tApp.test_client()
    tIndex = tClient.get("/")
    tLabels = tClient.get("/labels/left").get_json()
    tEdited = dict(tLabels["labels"][0], status="ok", tips=[[1.0, 2.0], [3.0, 4.0]])
    tPost = tClient.post("/labels/left/0", json=tEdited)
    tAfter = json.loads((tOutDir / "left" / "labels.json").read_text())["labels"][0]
    tImage = tClient.get("/img/left/" + tLabels["labels"][1]["file"])
    checkSnapshot("kpt_label.serve", {"run_kw": tRunKw, "stdout": tOut.getvalue(), "index_status": tIndex.status_code,
                                      "index_page": tIndex.get_data(as_text=True), "post": [tPost.status_code, tPost.get_data(as_text=True)],
                                      "after": tAfter, "image_status": tImage.status_code, "image_bytes": len(tImage.get_data())})


def test_cli_defaults(monkeypatch):
    checkSnapshot("kpt_label.cli.select", parsedArgs(monkeypatch, MODULE, ["select", "takeX"]))
    checkSnapshot("kpt_label.cli.serve", parsedArgs(monkeypatch, MODULE, ["serve", "dirY"]))
    assert helpExitCode(monkeypatch, MODULE, []) == 0
