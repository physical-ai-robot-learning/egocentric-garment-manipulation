"""Pieces shared by the two LeRobot builders of the tag-rig laundry takes (dataset stage; was part of tag_rig_to_lerobot.py).

Camera layout and image loading, the jaw label ramp / gap rules, the seeded stratified episode split, the LeRobot features common
to both builders and the dataset factory with the fixed writer settings. The v1 builder is tag_rig_to_lerobot, the current
(unsplit) builder is tag_rig_unsplit.
"""
from __future__ import annotations

import cv2
import numpy as np

from laundry_shared.contracts.action_contract import ACTION_DT_S
from laundry_shared.contracts.observation_contract import PROPRIO_DIM, PROPRIO_NAMES
from laundry_shared.umi_to_lerobot import ABS_NAMES

TASK = "laundry folding"
IMG_WH = (384, 216)                  # every camera resized to this (16:9, ONE shape: DiT)
CAMS = ("global", "left_wrist", "right_wrist")
MAX_SRC_GAP_S = 0.15                 # pose bracket longer than this -> invalid frame
MAX_JAW_GAP_S = 0.06                 # nearest jaw sample further than this -> jaw label invalid
GRIP_RAMP_S = 0.1                    # binary jaw state -> linear ramp of 0.1 s at each transition (box filter of the step)
GRIP_RAMP_SAMPLES = 21               # samples of that box filter
FPS = int(round(1 / ACTION_DT_S))
ROBOT_TYPE = "handumi_tagrig_x2"
ACTION_DIM = 20
SPLIT_FRAC = (0.85, 0.08, 0.07)     # user 10-22 night: ~85/8/7 by episode, stratified across takes
MIN_SPLIT_DEMOS = 3                  # fewer eligible demos -> everything is train
DROPPED_JPG_OFFSETS = (1, -1, 2, -2, 3, -3)   # 10-27: a dropped wrist jpg -> nearest existing frame within 0.1 s (30 Hz)
FRAME_DIGITS = 5
JPG_SUFFIX = ".jpg"


def frameFile(camDir, cam: str, i: int):
    """the jpg of frame i of a camera: color_NNNNN.jpg (global) / wrist_NNNNN.jpg"""
    return camDir / (f"color_{i:0{FRAME_DIGITS}d}.jpg" if cam == "global" else f"wrist_{i:0{FRAME_DIGITS}d}.jpg")


def loadImage(build, cam: str, i: int) -> np.ndarray:
    """frame i of camera `cam` of a built demo (its cam_dir) -> RGB uint8 IMG_WH (INTER_AREA) (was load_img)"""
    tPath = frameFile(build["cam_dir"][cam], cam, i)
    tImg = cv2.imread(str(tPath))
    for tD in DROPPED_JPG_OFFSETS:
        if tImg is not None:
            break
        tAlt = tPath.with_name(tPath.name[:-(FRAME_DIGITS + len(JPG_SUFFIX))] + f"{i + tD:0{FRAME_DIGITS}d}.jpg")
        tImg = cv2.imread(str(tAlt)) if tAlt.exists() else None
    if tImg is None:
        raise IOError(f"unreadable {tPath}")
    return cv2.cvtColor(cv2.resize(tImg, IMG_WH, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def rampedJaw(jaw, tj, tg, nearestFrame) -> np.ndarray:
    """binary jaw state -> mean over a GRIP_RAMP_S box around each grid time (= linear ramp at transitions)"""
    tFine = tg[:, None] + np.linspace(-GRIP_RAMP_S / 2, GRIP_RAMP_S / 2, GRIP_RAMP_SAMPLES)[None]
    return jaw["state"][nearestFrame(tj, tFine.ravel())[0]].reshape(tFine.shape).astype(float).mean(1)


def splitAssign(keys, seed, frac=SPLIT_FRAC) -> dict:
    """episode-level, seeded, STRATIFIED across takes: shuffle within each take, interleave takes round-robin (largest first),
    give the first n_val to val, the next n_test to test, the rest to train -> val/test draw from as many takes as possible
    (was split_assign)"""
    keys = sorted(keys)
    tN = len(keys)
    tRng = np.random.default_rng(seed)
    tBy = {}
    for tK in keys:
        tBy.setdefault(tK.split("/")[0], []).append(tK)
    for tT in tBy:
        tBy[tT] = [tBy[tT][tI] for tI in tRng.permutation(len(tBy[tT]))]
    tOrder, tTakes = [], sorted(tBy, key=lambda t: (-len(tBy[t]), t))
    while any(tBy.values()):
        for tT in tTakes:
            if tBy[tT]:
                tOrder.append(tBy[tT].pop(0))
    tNVal = max(1, int(round(frac[1] * tN))) if tN >= MIN_SPLIT_DEMOS else 0
    tNTest = max(1, int(round(frac[2] * tN))) if tN >= MIN_SPLIT_DEMOS else 0
    return {tK: ("val" if tR < tNVal else "test" if tR < tNVal + tNTest else "train") for tR, tK in enumerate(tOrder)}


def baseFeatures() -> dict:
    """LeRobot features both builders start with: three cameras, contract proprio, absolute action, current absolute pose"""
    tImage = dict(dtype="video", shape=(IMG_WH[1], IMG_WH[0], 3), names=["height", "width", "channels"])
    tFeats = {f"observation.images.{tC}": dict(tImage) for tC in CAMS}
    tFeats.update({"observation.state": dict(dtype="float32", shape=(PROPRIO_DIM,), names=list(PROPRIO_NAMES)),
                   "action": dict(dtype="float32", shape=(ACTION_DIM,), names=ABS_NAMES),
                   "pose_abs": dict(dtype="float32", shape=(ACTION_DIM,), names=ABS_NAMES)})
    return tFeats


def tailFeatures() -> dict:
    """LeRobot features both builders end with: continuous jaw, source frame indices and image residuals"""
    return {"gripper_cont": dict(dtype="float32", shape=(2,), names=["g_cont_left", "g_cont_right"]),
            "src_index": dict(dtype="int64", shape=(3,), names=list(CAMS)),
            "img_residual_ms": dict(dtype="float32", shape=(3,), names=list(CAMS))}


def createDataset(repoId: str, root, features: dict):
    """LeRobotDataset.create with the fixed writer settings of both builders (video, streaming encoder, 2 encoder threads)"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset.create(repoId, fps=FPS, features=features, root=root, robot_type=ROBOT_TYPE, use_videos=True,
                                 image_writer_threads=0, streaming_encoding=True, encoder_threads=2)
