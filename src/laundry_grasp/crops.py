"""Jaw pixels, jaw tip and the jaw-ROI crop of a HandUMI wrist camera (was grasp_label_v0.jaw_mask_and_tip + the crop code repeated
in grasp_cnn_v1 / grasp_v2_events / grasp_render_prob.crop_fn).

The jaw is fixed in the wrist image. Its pixels are the ones dark (gray < DARK_GRAY) in >= 90 % of up to 200 closed-jaw frames of the
take, at the 320 x 180 analysis resolution. The tip is the mean of the topmost 10 % of those pixels in the central 35-65 % of the
width (fallback (0.49, 0.62) of the image when fewer than 30 pixels). The classifier input is a 256 px square of the 1280 x 720 frame,
centred on the tip x, top 153 px above the tip y (clipped into the frame), resized to 224, RGB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .demos import camDir, nearestFrame, wristImage

ANALYSIS_W, ANALYSIS_H = 320, 180     # analysis resolution of the jaw mask / v0 features
DARK_GRAY = 90                        # black shirt / jaw pixels on the white table (11-03 / 11-04 shirts)
MASK_DARK_SHARE = 0.9                 # a jaw pixel is dark in at least this share of the closed frames
MIN_CLOSED_SAMPLES = 20               # fewer closed jaw samples -> use every jaw sample for the mask
MASK_FRAMES = 200
TIP_BAND = (0.35, 0.65)               # the tip is searched in this share of the image width
MIN_JAW_PIXELS = 30
TIP_TOP_PERCENTILE = 10
FALLBACK_TIP = (0.49, 0.62)           # share of (width, height) when no jaw is found
CAMERA_W, CAMERA_H = 1280, 720        # wrist camera frames
ROI_PX = 256
ROI_ABOVE_TIP_PX = 153                # crop top = tip y - 153 px (0.6 of the ROI; written as the integer the crops were made with)
MODEL_PX = 224
TIP_DISK_RADIUS_FRAC = 0.09           # jaw-tip disk radius / analysis width (v0 features, drawn on the review videos)


def readGray(cameraDir: Path, index: int) -> np.ndarray | None:
    """wrist frame `index` at the analysis resolution, gray; None when the jpg is missing"""
    tImage = cv2.imread(str(wristImage(cameraDir, index)))
    if tImage is None:
        return None
    return cv2.cvtColor(cv2.resize(tImage, (ANALYSIS_W, ANALYSIS_H)), cv2.COLOR_BGR2GRAY)


def jawMaskAndTip(cameraDir: Path, wristTimes: np.ndarray, jaw) -> tuple[np.ndarray, tuple[int, int]]:
    """(jaw pixel mask H x W bool, tip (x, y)) at the analysis resolution; `jaw` = the take's umi_jaw.npz (t_ns, state)"""
    tClosed = np.flatnonzero(jaw["state"] == 0)
    if len(tClosed) < MIN_CLOSED_SAMPLES:
        tClosed = np.arange(len(jaw["t_ns"]))
    tPicked = tClosed[np.linspace(0, len(tClosed) - 1, MASK_FRAMES).astype(int)]
    tIdx = np.unique(np.searchsorted(wristTimes, jaw["t_ns"][tPicked]).clip(0, len(wristTimes) - 1))
    tGray = np.stack([tG for tG in (readGray(cameraDir, tI) for tI in tIdx) if tG is not None])
    tMask = (tGray < DARK_GRAY).mean(0) >= MASK_DARK_SHARE
    tX0, tX1 = int(TIP_BAND[0] * ANALYSIS_W), int(TIP_BAND[1] * ANALYSIS_W)
    tYs, tXs = np.nonzero(tMask[:, tX0:tX1])
    tXs = tXs + tX0
    if len(tYs) < MIN_JAW_PIXELS:
        return tMask, (int(FALLBACK_TIP[0] * ANALYSIS_W), int(FALLBACK_TIP[1] * ANALYSIS_H))
    tTop = tYs <= np.percentile(tYs, TIP_TOP_PERCENTILE)
    return tMask, (int(tXs[tTop].mean()), int(tYs[tTop].mean()))


def roiOrigin(tip: tuple[int, int]) -> tuple[int, int]:
    """top-left (x0, y0) of the ROI_PX square in camera pixels for a tip at the analysis resolution"""
    tCx, tCy = tip[0] * CAMERA_W // ANALYSIS_W, tip[1] * CAMERA_H // ANALYSIS_H
    return (int(np.clip(tCx - ROI_PX // 2, 0, CAMERA_W - ROI_PX)),
            int(np.clip(tCy - ROI_ABOVE_TIP_PX, 0, CAMERA_H - ROI_PX)))


def cropRoi(imageBgr: np.ndarray | None, origin: tuple[int, int]) -> np.ndarray:
    """MODEL_PX x MODEL_PX RGB jaw-ROI crop of a camera frame (a missing frame is black)"""
    tImage = np.zeros((CAMERA_H, CAMERA_W, 3), np.uint8) if imageBgr is None else imageBgr
    tX0, tY0 = origin
    tRoi = tImage[tY0:tY0 + ROI_PX, tX0:tX0 + ROI_PX]
    return cv2.resize(cv2.cvtColor(tRoi, cv2.COLOR_BGR2RGB), (MODEL_PX, MODEL_PX))


@dataclass(frozen=True, eq=False)
class JawRoiCropper:
    """jaw-ROI crops of one wrist camera of a take at any time (the frame at or after the time; was grasp_render_prob.crop_fn)"""
    mCamDir: Path
    mWristTimes: np.ndarray = field(repr=False)
    mTip: tuple[int, int]
    mOrigin: tuple[int, int]

    @classmethod
    def forCamera(cls, take: str, cam: str) -> JawRoiCropper:
        tDir = camDir(take, cam)
        tWristTimes = np.load(tDir / "t_ns.npy")
        tTip = jawMaskAndTip(tDir, tWristTimes, np.load(tDir / "umi_jaw.npz"))[1]
        return cls(tDir, tWristTimes, tTip, roiOrigin(tTip))

    def frameAt(self, tNs) -> int:
        return nearestFrame(self.mWristTimes, tNs)

    def __call__(self, tNs) -> np.ndarray:
        return cropRoi(cv2.imread(str(wristImage(self.mCamDir, self.frameAt(tNs)))), self.mOrigin)
