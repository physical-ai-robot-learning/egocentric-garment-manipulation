"""Characterization of laundry_jaw.kpt_train: the YOLO-pose dataset export (labels, split, data.yaml)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from laundry_jaw import kpt_train
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

MODULE = "laundry_jaw.kpt_train"
N_LABELS = 16


def test_export_yolo_dataset(tmp_path):
    tHandDir = tmp_path / "left"
    (tHandDir / "frames").mkdir(parents=True)
    tLabels = []
    for tK in range(N_LABELS):
        tFile = f"{tK * 3:05d}.jpg"
        cv2.imwrite(str(tHandDir / "frames" / tFile), np.full((90, 176, 3), 10 * tK, np.uint8))
        tVisible = tK % 5 != 4
        tTips = [[30.0 + tK, 40.0], [120.0 - tK, 44.0 + tK / 2]] if tVisible and tK % 6 else None
        tLabels.append({"file": tFile, "visible": tVisible, "tips": tTips})
    (tHandDir / "labels.json").write_text(json.dumps({"labels": tLabels}))
    (tHandDir / "yolo" / "stale").mkdir(parents=True)          # export() must clear an earlier export
    tYolo = kpt_train.export(tmp_path, "left")
    tFiles = {}
    for tPath in sorted(tYolo.rglob("*")):
        if tPath.is_file():
            tRel = str(tPath.relative_to(tYolo))
            tFiles[tRel] = tPath.read_text().replace(str(tmp_path), "<tmp>") if tPath.suffix in (".txt", ".yaml") else tPath.stat().st_size
    checkSnapshot("kpt_train.export", {"yolo": str(tYolo.relative_to(tmp_path)), "files": tFiles})


def test_cli_defaults(monkeypatch):
    checkSnapshot("kpt_train.cli", parsedArgs(monkeypatch, MODULE, ["labels/dir", "--hand", "left"]))
    assert helpExitCode(monkeypatch, MODULE, []) == 0
