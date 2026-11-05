"""Classifier recipe pieces that need no weights (frame times, normalization, prediction averaging, training-set rules) and the gold
round merge."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from laundry_grasp import cv, gold, model

NS = 1_000_000_000


def test_event_frame_times_clip_inside_the_close():
    assert model.eventFrameTimes(0, 5 * NS) == [0.3e9, 0.5e9, 0.7e9, 0.9e9]
    # 0.6 s close: frames stop 0.05 s before the open
    assert model.eventFrameTimes(0, int(0.6 * NS)) == [0.3e9, 0.5e9, 0.55e9, 0.55e9]
    # very short close: at least 0.1 s after the close
    assert model.eventFrameTimes(0, int(0.1 * NS)) == [0.1e9] * 4
    assert model.approachFrameTimes(NS) == [0.4e9, 0.55e9, 0.7e9, 0.85e9]


def test_to_tensor_normalization():
    tImages = np.zeros((2, 224, 224, 3), np.uint8)
    tImages[1] = 255
    tX = model.toTensor(tImages)
    assert tX.shape == (2, 3, 224, 224) and tX.dtype == torch.float32
    assert torch.allclose(tX[0, :, 0, 0], torch.tensor(-model.IMAGENET_MEAN / model.IMAGENET_STD))
    assert torch.allclose(tX[1, :, 0, 0], torch.tensor((1 - model.IMAGENET_MEAN) / model.IMAGENET_STD))


class _MeanBrightness(torch.nn.Module):
    """stand-in classifier: logit = mean pixel value of the normalized image"""

    def forward(self, x):
        return x.mean(dim=(1, 2, 3))[:, None]


def test_predict_averages_frames_per_sample():
    tX = np.zeros((3, 2, 224, 224, 3), np.uint8)
    tX[1, 1] = 255
    tP = model.predict(_MeanBrightness(), tX, device="cpu")
    assert tP.shape == (3,)
    assert tP[0] == pytest.approx(tP[2]) and tP[1] > tP[0]


def _crop(take, cam):
    return lambda t: np.full((224, 224, 3), int(t // NS) % 256, np.uint8)


def test_training_set_overrides_and_approach_negatives():
    tRows = [dict(id="a", take="T", cam="wrist", t_close_ns=str(10 * NS), t_open_ns=str(12 * NS), gold="held"),
             dict(id="b", take="T", cam="wrist", t_close_ns=str(20 * NS), t_open_ns=str(22 * NS), gold="held"),
             dict(id="c", take="T", cam="wrist", t_close_ns=str(30 * NS), t_open_ns=str(32 * NS), gold="held"),
             dict(id="d", take="T", cam="wrist", t_close_ns=str(40 * NS), t_open_ns=str(42 * NS), gold="held")]
    tOverrides = {"_note": "x", "b": "slip", "c": "loose", "d": "empty"}
    tX, tY = model.trainingSet(tRows, tOverrides, _crop)
    # a: held + negative; b skipped; c loose -> not held; d 'empty' override is NOT applied by train_all (see README)
    assert tY.tolist() == [0, 1, 1, 1, 0, 1]
    assert tX.shape == (6, 4, 224, 224, 3)
    assert tX[1, 0, 0, 0, 0] == 9                       # approach crop at close - 0.6 s
    tG = [dict(tR, cls=gold.eventClass(tR, tOverrides)) for tR in tRows]
    tCroppers = {("T", "wrist"): _crop("T", "wrist")}
    tX2, tY2 = cv.foldTrainingSet(tG, [], tCroppers)
    assert tY2.tolist() == [0, 1, 1, 1, 1, 1]           # the CV applies the 'empty' override


def test_take_folds_are_grouped_and_seeded():
    tTakes = np.array([f"t{tI % 7}" for tI in range(40)])
    tFolds = cv.takeFolds(tTakes)
    assert len(tFolds) == 5 and sorted(np.concatenate(tFolds).tolist()) == sorted(set(tTakes.tolist()))
    assert [tF.tolist() for tF in tFolds] == [tF.tolist() for tF in cv.takeFolds(tTakes)]


def test_band_f1():
    tY = np.array([1, 1, 0, 0, 0])
    tP = np.array([0.9, 0.4, 0.1, 0.6, 0.45])
    tScores, tCoverage = cv.bandF1(tY, tP, 0.5, 0.5)
    assert tCoverage == 1.0
    assert tScores["empty"] == (0.5, 0.5, 0.5) and tScores["held"] == (pytest.approx(0.67), pytest.approx(0.67), pytest.approx(0.67))


def _writeRound(path, selection, labels):
    path.mkdir(parents=True)
    (path / "gold_selection.json").write_text(json.dumps(selection))
    with open(path / "gold_events.csv", "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=["id", "video", "event", "gold"])
        tW.writeheader()
        tW.writerows([dict(id=tI, video="v", event="e", gold=tG) for tI, tG in labels])


def test_round_gold_merge(tmp_path):
    tSel = lambda tIds: [dict(id=tI, video=f"{tI}_v", event="L1") for tI in tIds]   # noqa: E731
    _writeRound(tmp_path / "grasp_label_v0_30", tSel(["a", "b", "c"]), [("a", "held"), ("b", "empty"), ("c", "held")])
    _writeRound(tmp_path / "grasp_gold_r2", tSel(["b", "d"]), [("b", "held"), ("d", "uncertain")])
    (tmp_path / "grasp_gold_r3").mkdir()
    (tmp_path / "grasp_gold_r3" / "gold_selection.json").write_text(json.dumps(tSel(["e"])))   # labels not written yet
    (tmp_path / "two.txt").write_text("c\n")
    tGold = gold.roundGold(tmp_path, tmp_path / "two.txt")
    # c dropped (two closes in round 1), b relabelled by round 2, d not held/empty
    assert [(tG["id"], tG["gold"]) for tG in tGold] == [("a", "held"), ("b", "held")]
    assert tGold[0]["video"] == "a_v"


def test_two_close_ids_ship_with_the_repo():
    assert len(gold.TWO_CLOSE_IDS.read_text().split()) == 59
