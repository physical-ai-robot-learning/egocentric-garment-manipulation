"""Jaw mask / tip finder and the jaw-ROI crop geometry on synthetic wrist frames."""
from __future__ import annotations

import cv2
import numpy as np

from laundry_grasp import crops, demos
from laundry_grasp.tests import synthetic


def test_roi_origin_center_and_clipping():
    # tip at the analysis center -> camera (640, 360) -> x0 = 640 - 128, y0 = 360 - 153
    assert crops.roiOrigin((160, 90)) == (512, 207)
    assert crops.roiOrigin((0, 0)) == (0, 0)                        # clipped at the top-left
    assert crops.roiOrigin((319, 179)) == (1024, 464)               # clipped so the square stays inside 1280 x 720
    # integer arithmetic of the 11-05 crops: tip x * 1280 // 320, tip y * 720 // 180
    assert crops.roiOrigin((157, 113)) == (157 * 4 - 128, 113 * 4 - 153)


def test_crop_roi_content_and_missing_frame():
    tImage = np.zeros((720, 1280, 3), np.uint8)
    tImage[207:207 + 256, 512:512 + 256] = (255, 0, 0)               # BGR blue square exactly on the ROI
    tCrop = crops.cropRoi(tImage, (512, 207))
    assert tCrop.shape == (224, 224, 3) and tCrop.dtype == np.uint8
    assert np.all(tCrop[..., 2] == 255) and np.all(tCrop[..., :2] == 0)   # RGB order
    assert not crops.cropRoi(None, (0, 0)).any()


def test_jaw_mask_and_tip_on_synthetic_jaw(synthTakes):
    tDir = demos.camDir(synthetic.TAKE, "wrist")
    tWrist = np.load(tDir / "t_ns.npy")
    tMask, tTip = crops.jawMaskAndTip(tDir, tWrist, np.load(tDir / "umi_jaw.npz"))
    # the jaw block (rows 380.., cols 560..720 of 1280 x 720) at 320 x 180: rows 95.., cols 140..180
    assert tMask.shape == (180, 320)
    assert tMask[100:, 145:175].all() and not tMask[:90].any()
    assert 140 <= tTip[0] <= 180 and 94 <= tTip[1] <= 100           # topmost jaw pixels, centered in x


def test_fallback_tip_without_jaw(tmp_path):
    tDir = tmp_path / "cam"
    tDir.mkdir()
    for tI in range(3):
        cv2.imwrite(str(tDir / f"wrist_{tI:05d}.jpg"), np.full((720, 1280, 3), 230, np.uint8))
    tJaw = dict(t_ns=np.array([0, 1, 2], np.int64), state=np.array([1, 1, 1]))
    tMask, tTip = crops.jawMaskAndTip(tDir, np.array([0, 1, 2], np.int64), tJaw)
    assert not tMask.any() and tTip == (int(0.49 * 320), int(0.62 * 180))


def test_cropper_uses_the_frame_at_or_after_the_time(synthTakes):
    tCrop = crops.JawRoiCropper.forCamera(synthetic.TAKE, "wrist")
    tT = tCrop.mWristTimes[7]
    assert tCrop.frameAt(tT) == 7 and tCrop.frameAt(tT + 1) == 8 and tCrop.frameAt(-1) == 0
    assert tCrop.frameAt(tCrop.mWristTimes[-1] + 10**12) == len(tCrop.mWristTimes) - 1
    assert tCrop(tT).shape == (224, 224, 3)
    assert tCrop.mOrigin == crops.roiOrigin(tCrop.mTip)
