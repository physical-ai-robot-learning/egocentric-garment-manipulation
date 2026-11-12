"""[11-06] Robot-side p_grasp for checkpoints trained with PGRASP=<version> (deploy stage): state 18, 19 = causal P(cloth held) of the
L / R jaw. Same frozen classifier and rule as the training side file (laundry_dit_training/datasets/grasp_pgrasp_gen.py, grasp-v2-W0.4):

  frame  wrist raw (1920x1080) -> 1280x720 INTER_AREA -> JPEG q92 round trip (= the take files) -> 256 px crop at (x0, y0) -> 224 RGB
  model  torchvision resnet18, fc -> 1 logit = P(not held); P(held) = 1 - sigmoid
  rule   p(t) = mean P(held) over [max(last jaw change, t - WINDOW_S), t] at HZ; jaw change = v2 hysteresis on the jaw opening g
         (0 closed .. 1 open): open > JAW_HI, closed < JAW_LO, dwell JAW_DWELL_S (jaw_from_motor.hysteresis, change time = dwell start)

Input: the wrist cameras (.mRaw, .mAlign) and a jaw reader. Output: PGraspEstimator.at(t_ns) -> [pL, pR].
[11-06 robot check] the human-trained classifier says P(held) 0.97 / 0.94 on the robot's CLOSED EMPTY jaws (the robot jaw looks larger and
glossier than the HandUMI jaw) -> default source = "jaw": P(held) = 1 while the jaw is closed, 0 while open (same window rule; the
human held-event median is 0.99), until a robot-domain classifier exists. DEPLOY_PGRASP_SOURCE=vision runs the classifier.
ROI: crop corner per arm in 1280x720 pixels, $TAGRIG_STATE_DIR/pgrasp_roi.json (`python -m laundry_deploy.core.pgrasp calibrate`, jaws
CLOSED and empty); the human data put the jaw tip TIP_DY px below the crop top, horizontally centred.
"""

from __future__ import annotations
import collections
import json
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

from laundry_common import paths as egopaths

from .contract import SIDES, TRAIN_ROOT
from .errors import RunRefused
from .images import applyAlign

VERSION = "grasp-v2-W0.4"
SOURCE_ENV = "DEPLOY_PGRASP_SOURCE"  # "jaw" (default) | "vision"
SOURCE_JAW, SOURCE_VISION = "jaw", "vision"
HZ = 15.0
WINDOW_S = 0.4
JAW_HI, JAW_LO, JAW_DWELL_S = 0.30, 0.15, 0.1
MODEL_DIR = TRAIN_ROOT / "data/grasp_labels_v2/pgrasp_v2_W0.4"
ROI_FILE = egopaths.TAGRIG_STATE_DIR / "pgrasp_roi.json"
FRAME_W, FRAME_H = 1280, 720
CROP_PX, NET_PX, TIP_DY = 256, 224, 153
JPEG_QUALITY = 92
HISTORY_S = 4.0
DARK_LEVEL = 60  # jaw pixels: darker than this in >= JAW_DARK_FRAC of the closed frames
JAW_DARK_FRAC = 0.9
TIP_COLUMNS = (0.35, 0.65)  # jaw tip search band (fraction of the width)
TIP_TOP_PCT = 10
MIN_JAW_PIXELS = 30
MARKER_FILE = "GRIP_STATE_MASK"
MARKER_KEY = "PGRASP="
RUN_NAME_TAG = "PGRASPv2W04"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
NS_PER_S = 1e9


def checkpointVersion(ckpt) -> str | None:
    """PGRASP version a checkpoint was trained with (GRIP_STATE_MASK marker text 'PGRASP=<v>' next to / above the step folder, or a
    *PGRASPv2W04* run name), else None"""
    tCk = pathlib.Path(str(ckpt))
    for tDir in (tCk, tCk.parent, tCk.parent.parent):
        tMarker = tDir / MARKER_FILE
        if tMarker.exists() and MARKER_KEY in tMarker.read_text():
            return tMarker.read_text().split(MARKER_KEY)[1].split()[0].rstrip(")")
    return VERSION if RUN_NAME_TAG in str(ckpt) else None


def frame720(rawBgr: np.ndarray, align: dict | None = None) -> np.ndarray:
    """wrist raw BGR -> the 1280x720 take-file frame (images.prepWrist before its 384x216 resize)"""
    tX = rawBgr if rawBgr.shape[:2] == (FRAME_H, FRAME_W) else cv2.resize(rawBgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    if align is not None:
        tX = applyAlign(tX, align)
    return cv2.imdecode(cv2.imencode(".jpg", tX, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1], cv2.IMREAD_COLOR)


def crop224(frame: np.ndarray, xy) -> np.ndarray:
    tX0, tY0 = int(xy[0]), int(xy[1])
    return cv2.resize(cv2.cvtColor(frame[tY0:tY0 + CROP_PX, tX0:tX0 + CROP_PX], cv2.COLOR_BGR2RGB), (NET_PX, NET_PX))


class Classifier:
    def __init__(self, device: str = "mps") -> None:
        import torch
        import torchvision

        tMeta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        if tMeta["version"] != VERSION:
            raise RunRefused(f"p_grasp 모델 버전 {tMeta['version']} != {VERSION}")
        tNet = torchvision.models.resnet18()
        tNet.fc = torch.nn.Linear(tNet.fc.in_features, 1)
        tNet.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
        self.mTorch, self.mDevice, self.mNet = torch, device, tNet.to(device).eval()

    def pHeld(self, crops: np.ndarray) -> np.ndarray:
        """(n, 224, 224, 3) uint8 RGB -> (n,) P(held)"""
        tX = ((crops.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)
        tX = self.mTorch.from_numpy(np.ascontiguousarray(tX)).to(self.mDevice)
        with self.mTorch.no_grad():
            return 1.0 - self.mTorch.sigmoid(self.mNet(tX).squeeze(1)).float().cpu().numpy()


class JawEvents:
    """online v2 hysteresis on the jaw opening g -> time (ns) of the last open / close change (jaw_from_motor.hysteresis)"""

    def __init__(self) -> None:
        self.mState = None
        self.mCandNs = None
        self.mLastNs = None

    def update(self, tNs: int, g) -> int | None:
        if g is None:
            return self.mLastNs
        if self.mState is None:
            self.mState, self.mLastNs = int(g > (JAW_HI + JAW_LO) / 2), tNs
            return self.mLastNs
        tWant = 1 if g > JAW_HI else 0 if g < JAW_LO else self.mState
        if tWant == self.mState:
            self.mCandNs = None
            return self.mLastNs
        if self.mCandNs is None:
            self.mCandNs = tNs
        if (tNs - self.mCandNs) / NS_PER_S >= JAW_DWELL_S:
            self.mState, self.mLastNs, self.mCandNs = tWant, self.mCandNs, None
        return self.mLastNs


class PGraspEstimator(threading.Thread):
    """background HZ p_grasp of both arms. cams = {side: WristCam} (.mRaw full frame, .mAlign = the --wrist-align framing the policy
    images get), jawFn() -> {side: g (0 closed .. 1 open) or None}, roi = {side: (x0, y0)} (default pgrasp_roi.json)."""

    def __init__(self, cams: dict, jawFn, roi: dict | None = None, device: str = "mps", source: str | None = None) -> None:
        super().__init__(daemon=True)
        import os

        self.mCams, self.mJawFn = cams, jawFn
        self.mSource = source or os.environ.get(SOURCE_ENV, SOURCE_JAW)
        if self.mSource not in (SOURCE_JAW, SOURCE_VISION):
            raise RunRefused(f"{SOURCE_ENV}={self.mSource} (jaw | vision)")
        self.mAlign = {tSide: getattr(cams[tSide], "mAlign", None) for tSide in SIDES}
        tVision = self.mSource == SOURCE_VISION
        self.mRoi = (roi or loadRoi()) if tVision else None
        self.mClassifier = Classifier(device) if tVision else None
        self.mJaw = {tSide: JawEvents() for tSide in SIDES}
        tN = int(HZ * HISTORY_S)
        self.mHist = {tSide: collections.deque(maxlen=tN) for tSide in SIDES}  # (t_ns, P(held)) per tick
        self.mOut = collections.deque(maxlen=tN)  # (t_ns, [pL, pR])
        self.mLock = threading.Lock()
        self.mStop = False
        self.mError = None
        self.mTicks = 0

    def step(self, tNs: int, raws: dict, jaw: dict) -> np.ndarray:
        tP = np.zeros(len(SIDES), np.float32)
        tLasts = {tSide: self.mJaw[tSide].update(tNs, jaw.get(tSide)) for tSide in SIDES}
        if self.mClassifier is not None:
            tCrops = np.stack([crop224(frame720(raws[tSide], self.mAlign.get(tSide)), self.mRoi[tSide]) for tSide in SIDES])
            tHeld = self.mClassifier.pHeld(tCrops)
        else:  # jaw source: held while the jaw is closed (hysteresis state 0)
            tHeld = np.array([float(self.mJaw[tSide].mState == 0) for tSide in SIDES], np.float32)
        for tJ, tSide in enumerate(SIDES):
            tLast = tLasts[tSide]
            self.mHist[tSide].append((tNs, float(tHeld[tJ])))
            if self.mClassifier is None:  # jaw source: the state itself (a window over the dwell would ramp 0 -> 1 over 0.4 s)
                tP[tJ] = tHeld[tJ]
                continue
            tFrom = max(tLast if tLast is not None else 0, tNs - int(WINDOW_S * NS_PER_S))
            tP[tJ] = np.mean([tV for tT, tV in self.mHist[tSide] if tT >= tFrom])
        with self.mLock:
            self.mOut.append((tNs, tP))
        self.mTicks += 1
        return tP

    def run(self) -> None:
        tDt = 1.0 / HZ
        tNext = time.monotonic()
        while not self.mStop:
            try:
                tRaws = {tSide: self.mCams[tSide].mRaw for tSide in SIDES}
                if self.mClassifier is None or all(tR is not None for tR in tRaws.values()):
                    self.step(time.monotonic_ns(), tRaws, self.mJawFn() or {})
                    self.mError = None
            except Exception as tErr:  # noqa: BLE001 -- reported through mError, the run refuses / logs it
                self.mError = f"{type(tErr).__name__}: {tErr}"
            tNext = max(tNext + tDt, time.monotonic() - tDt)
            time.sleep(max(0.0, tNext - time.monotonic()))

    def at(self, tNs: int | None = None) -> np.ndarray | None:
        """[pL, pR] of the last tick at or before tNs (latest when None; the oldest tick when tNs precedes all)"""
        with self.mLock:
            if not self.mOut:
                return None
            if tNs is None:
                return self.mOut[-1][1].copy()
            tBest = self.mOut[0][1]
            for tT, tP in self.mOut:
                if tT <= tNs:
                    tBest = tP
            return tBest.copy()


def loadRoi() -> dict:
    if not ROI_FILE.exists():
        raise RunRefused(f"{ROI_FILE} 없음 -- 먼저 `python -m laundry_deploy.core.pgrasp calibrate` (집게 닫고, 비운 채로)")
    tRoi = json.loads(ROI_FILE.read_text())
    return {tSide: tuple(tRoi[tSide]["xy"]) for tSide in SIDES}


def tipFromFrames(frames: list) -> tuple:
    """jaw tip (x, y) in 1280x720: pixels dark in >= JAW_DARK_FRAC of the frames (the fixed black jaw), mean of the topmost TIP_TOP_PCT %
    inside the centre band (grasp_label_v0.jaw_mask_and_tip on full-resolution frames)"""
    tGray = np.stack([cv2.cvtColor(tF, cv2.COLOR_BGR2GRAY) for tF in frames])
    tMask = (tGray < DARK_LEVEL).mean(0) >= JAW_DARK_FRAC
    tH, tW = tMask.shape
    tC0, tC1 = int(TIP_COLUMNS[0] * tW), int(TIP_COLUMNS[1] * tW)
    # 11-06 robot cameras: a dark room behind the table also passes the threshold -> keep only the dark regions that reach the bottom
    # edge inside the centre band (the jaw grows up from the bottom of the frame)
    _, tLabels = cv2.connectedComponents(tMask.astype(np.uint8))
    tJawLabels = set(np.unique(tLabels[tH - 1, tC0:tC1])) - {0}
    tMask = np.isin(tLabels, list(tJawLabels))
    tYs, tXs = np.nonzero(tMask[:, tC0:tC1])
    if len(tYs) < MIN_JAW_PIXELS:
        return None, tMask
    tTop = tYs <= np.percentile(tYs, TIP_TOP_PCT)
    return (int(tXs[tTop].mean()) + tC0, int(tYs[tTop].mean())), tMask


def calibrate(nFrames: int = 60, outDir=None, useAlign: bool = False) -> dict:
    """jaws CLOSED and empty: grab nFrames per wrist camera, find the tip, write pgrasp_roi.json + previews (crop box on the frame and
    the 224 crop) to the Desktop. useAlign = the runs use --wrist-align (default off, like the UI)."""
    from ..session.cameras import WristCam, autoRemapCams, uvcList
    from ..session.paths import CAMS_FILE

    tCfg = autoRemapCams(json.loads(CAMS_FILE.read_text()), uvcList())
    tAlign = tCfg.get("wrist_align") if useAlign else None
    tOut = pathlib.Path(outDir or egopaths.DESKTOP / "pgrasp_roi")
    tOut.mkdir(parents=True, exist_ok=True)
    tRes = {}
    for tSide in SIDES:
        tCam = WristCam(tSide, tCfg[tSide]["location"], tCfg[tSide]["index"], 2, (tAlign or {}).get(tSide), prep=False)
        tCam.start()
        tFrames = []
        while len(tFrames) < nFrames:
            if tCam.mRaw is not None:
                tFrames.append(frame720(tCam.mRaw, (tAlign or {}).get(tSide)))
            time.sleep(1.0 / HZ)
        tCam.mStop = True
        tTip, _ = tipFromFrames(tFrames)
        if tTip is None:
            raise RunRefused(f"{tSide}: 집게 검은 영역을 못 찾음 (집게 닫힘 / 조명 확인)")
        tXy = (int(np.clip(tTip[0] - CROP_PX // 2, 0, FRAME_W - CROP_PX)), int(np.clip(tTip[1] - TIP_DY, 0, FRAME_H - CROP_PX)))
        tRes[tSide] = dict(xy=tXy, tip=tTip, n_frames=nFrames)
        tView = tFrames[-1].copy()
        cv2.rectangle(tView, tXy, (tXy[0] + CROP_PX, tXy[1] + CROP_PX), (0, 0, 255), 3)
        cv2.circle(tView, tTip, 8, (0, 255, 0), -1)
        cv2.imwrite(str(tOut / f"{tSide}_frame.jpg"), tView)
        cv2.imwrite(str(tOut / f"{tSide}_crop.jpg"), cv2.cvtColor(crop224(tFrames[-1], tXy), cv2.COLOR_RGB2BGR))
        print(f"{tSide}: 집게 끝 {tTip} -> 판정 영역 {tXy}", flush=True)
    tRes.update(created=time.strftime("%Y-%m-%d %H:%M"), version=VERSION, wrist_align=bool(useAlign))
    ROI_FILE.write_text(json.dumps(tRes, indent=1))
    print(f"-> {ROI_FILE}  미리보기 {tOut}", flush=True)
    return tRes


if __name__ == "__main__":
    if sys.argv[1:2] == ["calibrate"]:
        calibrate(useAlign="--wrist-align" in sys.argv)
    else:
        print(__doc__)
