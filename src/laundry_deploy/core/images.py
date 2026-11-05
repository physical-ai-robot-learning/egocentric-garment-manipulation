"""Camera frames -> policy input (deploy stage): Orbbec global view and robot wrist Arducams resized exactly like the training
converter (core.contract), an optional deterministic framing fix for the wrist view, and the item dict LeRobotDataset would give.

Inputs: BGR uint8 frames (Orbbec 848x480, wrist 1920x1080 or 1280x720). Outputs: RGB uint8 384x216, torch float items.
"""

from __future__ import annotations

import cv2
import numpy as np

from .contract import CAMS, IMG_WH, TASK, WRIST_JPEG_QUALITY, WRIST_MID_WH

PIXEL_MAX = 255.0
ROT180_KEY, HFLIP_KEY, SCALE_KEY, DX_KEY, DY_KEY = "rot180", "hflip", "scale", "dx", "dy"  # act_tagrig_cams.json wrist_align
HFLIP_CODE = 1  # cv2.flip: around the vertical axis


def prepGlobal(bgr: np.ndarray) -> np.ndarray:
    """Orbbec colour BGR (848x480, already JPEG q95 like the take files) -> RGB uint8 384x216 (converter load_img)."""
    return cv2.cvtColor(cv2.resize(bgr, IMG_WH, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def prepWrist(bgr: np.ndarray, jpegQ: int = WRIST_JPEG_QUALITY, align: dict | None = None) -> np.ndarray:
    """wrist BGR 1920x1080 -> 1280x720 INTER_AREA (umi_jaw record) -> JPEG q92 round trip (take files) -> converter load_img.
    align: optional deterministic framing transform (tools.domain_gap), applied on the 1280x720 frame; None = as trained."""
    tX = bgr if bgr.shape[:2] == WRIST_MID_WH[::-1] else cv2.resize(bgr, WRIST_MID_WH, interpolation=cv2.INTER_AREA)
    if align is not None:
        tX = applyAlign(tX, align)
    if jpegQ:
        tX = cv2.imdecode(cv2.imencode(".jpg", tX, [cv2.IMWRITE_JPEG_QUALITY, int(jpegQ)])[1], cv2.IMREAD_COLOR)
    return prepGlobal(tX)


def applyAlign(img: np.ndarray, align: dict) -> np.ndarray:
    """Similarity about the image centre: scale s, shift (dx, dy) px of the 1280x720 frame, rotation 0/180 deg, optional h-flip."""
    tImg = img
    tH, tW = tImg.shape[:2]
    if align.get(ROT180_KEY):
        tImg = cv2.rotate(tImg, cv2.ROTATE_180)
    if align.get(HFLIP_KEY):
        tImg = cv2.flip(tImg, HFLIP_CODE)
    tS = float(align.get(SCALE_KEY, 1.0))
    tDx = float(align.get(DX_KEY, 0.0))
    tDy = float(align.get(DY_KEY, 0.0))
    if tS != 1.0 or tDx or tDy:
        tM = np.float32([[tS, 0, (1 - tS) * tW / 2 + tDx], [0, tS, (1 - tS) * tH / 2 + tDy]])
        tInterp = cv2.INTER_AREA if tS < 1 else cv2.INTER_LINEAR
        tImg = cv2.warpAffine(tImg, tM, (tW, tH), flags=tInterp, borderMode=cv2.BORDER_REPLICATE)
    return tImg


def makeItem(imgsRgb: dict, state30) -> dict:
    """uint8 HWC RGB per camera + state -> the dict LeRobotDataset.__getitem__ would give (float32 CHW in [0, 1])."""
    import torch

    def _chw(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / PIXEL_MAX

    tItem = {f"observation.images.{tCam}": _chw(imgsRgb[tCam]) for tCam in CAMS}
    tItem["observation.state"] = torch.from_numpy(np.asarray(state30, np.float32))
    tItem["task"] = TASK
    return tItem


def makeItemPair(imgsRgb: dict, state30, prev: tuple | None = None) -> dict:
    """n_obs_steps = 2 item (DiT): (2, ...) stacks [t - 0.05 s, t]; prev = (imgsRgb, state30) of t - 0.05 s, None -> the current
    frame repeated (= LeRobot's clamped padding at an episode start)."""
    import torch

    tCur = makeItem(imgsRgb, state30)
    tOld = makeItem(*prev) if prev is not None else tCur
    tOut = {tKey: torch.stack([tOld[tKey], tCur[tKey]]) for tKey in tCur if tKey != "task"}
    tOut["task"] = TASK
    return tOut
