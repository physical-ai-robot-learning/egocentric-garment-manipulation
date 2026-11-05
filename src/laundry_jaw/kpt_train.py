"""Train a jaw-tip keypoint model per hand from `kpt_label` labels (2026-10-22; was scripts/pose/jaw_kpt_train.py).

Pipeline stage: jaw state (stage 2), the model `aperture` runs. One object per image ("gripper") with 2 keypoints (A = image-left
jaw tip, B = image-right jaw tip); the box spans both jaws down to the bottom edge. Split by frame index (every 7th frame -> val)
so near-duplicate neighbours do not leak much. Reports val tip error in px.
    python -m laundry_jaw.kpt_train datasets/jaw_kpt/20261022_105200 --hand left [--epochs 120 --model yolov8n-pose.pt]
-> <labels>/<hand>/yolo/ (dataset), runs in <labels>/<hand>/train/, best weights copied to
   configs/laundry/calibration/jaw_kpt_<hand>_<tag>.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common.paths import CONFIGS

# Every VAL_EVERY-th label (index % VAL_EVERY == VAL_PHASE) goes to the validation split.
VAL_EVERY = 7
VAL_PHASE = 3
# Box around the two tips: widened by 12 % of the crop width on each side, raised by 5 % of its height, down to the bottom edge.
BOX_MARGIN_X_FRAC = 0.12
BOX_MARGIN_Y_FRAC = 0.05
KEYPOINT_VISIBLE = 2
DATA_YAML = "path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: gripper\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n"
TRAIN_BATCH = 16
TRAIN_DEVICE = "mps"
# Augmentation of the 10-22 models.
TRAIN_AUGMENT = dict(fliplr=0.5, degrees=8, translate=0.05, scale=0.2, mosaic=0.0, hsv_v=0.4)
VAL_ERROR_PERCENTILE = 90
DEFAULT_EPOCHS = 120
DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_IMGSZ = 640
DEFAULT_TAG = "v001"


def export(labelDir: Path, hand: str) -> Path:
    """Write the YOLO-pose dataset of labelDir/<hand> to labelDir/<hand>/yolo (replacing an earlier export); returns that dir."""
    tLabels = json.loads((labelDir / hand / "labels.json").read_text())["labels"]
    tYolo = labelDir / hand / "yolo"
    if tYolo.exists():
        shutil.rmtree(tYolo)
    for tSplit in ("train", "val"):
        (tYolo / "images" / tSplit).mkdir(parents=True)
        (tYolo / "labels" / tSplit).mkdir(parents=True)
    for tK, tLabel in enumerate(tLabels):
        tSplit = "val" if tK % VAL_EVERY == VAL_PHASE else "train"
        tImage = cv2.imread(str(labelDir / hand / "frames" / tLabel["file"]))
        tH, tW = tImage.shape[:2]
        shutil.copy(labelDir / hand / "frames" / tLabel["file"], tYolo / "images" / tSplit / tLabel["file"])
        tLabelFile = tYolo / "labels" / tSplit / (Path(tLabel["file"]).stem + ".txt")
        if not tLabel["visible"] or not tLabel["tips"]:
            tLabelFile.write_text("")
            continue
        (tXa, tYa), (tXb, tYb) = tLabel["tips"]
        tX0 = max(0, min(tXa, tXb) - BOX_MARGIN_X_FRAC * tW)
        tX1 = min(tW, max(tXa, tXb) + BOX_MARGIN_X_FRAC * tW)
        tY0 = max(0, min(tYa, tYb) - BOX_MARGIN_Y_FRAC * tH)
        tY1 = tH
        tBox = [(tX0 + tX1) / 2 / tW, (tY0 + tY1) / 2 / tH, (tX1 - tX0) / tW, (tY1 - tY0) / tH]
        tKeypoints = [tXa / tW, tYa / tH, KEYPOINT_VISIBLE, tXb / tW, tYb / tH, KEYPOINT_VISIBLE]
        tLabelFile.write_text("0 " + " ".join(f"{tV:.6f}" for tV in tBox + tKeypoints) + "\n")
    (tYolo / "data.yaml").write_text(DATA_YAML.format(root=tYolo))
    return tYolo


def validationTipErrors(modelFile: Path, labelDir: Path, hand: str) -> tuple[np.ndarray, int]:
    """(tip errors px of the validation labels, number of validation frames without a detection)."""
    from ultralytics import YOLO
    tNet = YOLO(str(modelFile))
    tLabels = json.loads((labelDir / hand / "labels.json").read_text())["labels"]
    tErrors = []
    tMissed = 0
    for tK, tLabel in enumerate(tLabels):
        if tK % VAL_EVERY != VAL_PHASE or not tLabel["visible"]:
            continue
        tResult = tNet.predict(str(labelDir / hand / "frames" / tLabel["file"]), verbose=False, device=TRAIN_DEVICE)[0]
        if tResult.keypoints is None or len(tResult.keypoints.xy) == 0:
            tMissed += 1
            continue
        tBest = int(np.argmax(tResult.boxes.conf.cpu().numpy()))
        tTips = tResult.keypoints.xy[tBest].cpu().numpy()
        tTruth = np.array(tLabel["tips"])
        tTips = tTips[np.argsort(tTips[:, 0])]
        tErrors += list(np.linalg.norm(tTips - tTruth, axis=1))
    return np.array(tErrors), tMissed


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("dir")
    tParser.add_argument("--hand", required=True, choices=["left", "right"])
    tParser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    tParser.add_argument("--model", default=DEFAULT_MODEL)
    tParser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    tParser.add_argument("--tag", default=DEFAULT_TAG)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    from ultralytics import YOLO
    tLabelDir = Path(tArgs.dir).resolve()
    tYolo = export(tLabelDir, tArgs.hand)
    tModel = YOLO(tArgs.model)
    tModel.train(data=str(tYolo / "data.yaml"), epochs=tArgs.epochs, imgsz=tArgs.imgsz, batch=TRAIN_BATCH, device=TRAIN_DEVICE,
                 project=str(tLabelDir / tArgs.hand / "train"), name="run", exist_ok=True, verbose=False, plots=False, **TRAIN_AUGMENT)
    tBest = tLabelDir / tArgs.hand / "train/run/weights/best.pt"
    tOut = CONFIGS / f"calibration/jaw_kpt_{tArgs.hand}_{tArgs.tag}.pt"
    shutil.copy(tBest, tOut)
    # val tip error in px (original crop resolution)
    tErrors, tMissed = validationTipErrors(tOut, tLabelDir, tArgs.hand)
    print(f"[jaw kpt {tArgs.hand}] val tips {len(tErrors)} | error median {np.median(tErrors):.1f} px, "
          f"p90 {np.percentile(tErrors, VAL_ERROR_PERCENTILE):.1f} px | no detection {tMissed} -> {tOut}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
