"""Cameras of the deploy (deploy stage): UVC device list and wrist-camera auto-mapping by device name, the recorder / UMI-port
checks, the robot wrist Arducams (fixed exposure, white balance, stall watchdog) and the Orbbec poller, each feeding a Ring of
pre-processed policy frames on the host monotonic clock.
"""

from __future__ import annotations
import json
import os
import pathlib
import subprocess
import threading
import time

import cv2
import numpy as np

from ..core.contract import IMG_WH, SIDES
from ..core.errors import HardwareError
from ..core.images import prepGlobal, prepWrist
from .paths import CAMS_FILE, UVC_UTIL, say
from .ring import Ring

ARDUCAM_VENDOR = "0x0c45"  # USB vendor id of the wrist Arducams in the uvc-util list
REMAP_INDEX_CANDIDATES = [0, 1, 2, 3, 4, 5]
RECORDER_PATTERN = "tag_rig_record|umi_jaw|handumi_collector|laundry_collector|wrist_rec|stereo_check"
CAMERA_FREE_UMI_JAW = ("infer", "train")  # 10-26: offline jaw labelling opens no camera
WRIST_W, WRIST_H, WRIST_FPS = 1920, 1080, 30
WRIST_EXPOSURE = 2  # umi_jaw.fixed_exposure level of the UMI recordings
WASHED_OUT_MEAN = 185  # 10-29: the right camera washes out at the minimum exposure (mean ~226) -> brightness down
BRIGHTNESS_DOWN = "brightness=-64"
FLUSH_FRAMES = 6
WB_EVERY_N = 10  # white-balance knots are re-estimated every N frames
WB_STRIDE = 8  # pixel subsampling of that estimate
WB_MID_GREY = (90, 235)  # mid-grey pixels (mostly table)
WB_SHADOW = (25, 80)  # shadow pixels
WB_MIN_MID, WB_MIN_SHADOW = 500, 200  # sample counts needed for the mid-grey / shadow knots
WB_BLACK_PERCENTILE = 1
WB_KEEP, WB_GAIN = 0.85, 0.15  # EMA of the knots (~2 s at 30 fps / every 10th frame); written out, not 1 - WB_KEEP
WB_KNOT_GAP = 2  # the shadow knot must sit this far inside black..grey
LUT_SIZE = 256
STALL_S = 2.0  # no frame for this long = stalled read
WATCHDOG_PERIOD_S = 0.5
REOPEN_RELEASE_S = 0.5
REOPEN_RETRY_S = 2.0
IDLE_READ_S = 0.01
FAKE_WRIST_HZ = 30
REPLAY_HZ = 15
ORBBEC_PROBE_TIMEOUT_S = 1.0
ORBBEC_TIMEOUT_S = 0.5
ORBBEC_IDLE_S = 0.01
ORBBEC_ERROR_S = 0.05
HTTP_OK = 200
REPLAY_PREFIX = "dir:"


def uvcList() -> str:
    return subprocess.run([UVC_UTIL, "-d"], capture_output=True, text=True).stdout


def camKey() -> str:
    return "\n".join(sorted(" ".join(tLine.split()[1:]) for tLine in uvcList().splitlines() if ARDUCAM_VENDOR in tLine))


def autoRemapCams(cfg: dict, lst: str) -> dict:
    """[10-29 user: "하드코딩되어 있으면 자동으로 바꿔줘"] the two wrist Arducams report different device NAMES (firmware strings, stay with
    the physical camera: left 'Arducam 1080P Low Light', right 'FisheyeCamLeft').  If a configured USB location is gone or the
    USB set changed, find each side's camera by its name, re-probe the OpenCV index, rewrite act_tagrig_cams.json; the wave/cover
    confirmation stays valid (same physical cameras), the old mapping goes to act_tagrig_cams.json.bak_auto."""
    tRows = [tLine.split(None, 4) for tLine in lst.splitlines() if ARDUCAM_VENDOR in tLine]  # index vend:prod location uvc name
    tByName = {tRow[4].strip(): tRow[2] for tRow in tRows if len(tRow) >= 5}
    tMoved = [tSide for tSide in SIDES if (cfg.get(tSide, {}).get("location") or "").lower() not in lst.lower()]
    if not tMoved and cfg.get("cam_key") == camKey():
        return cfg
    from laundry_shared.visual_teleop.tools.stereo_check import index_for_location

    tNew = json.loads(json.dumps(cfg))
    for tSide in SIDES:
        tName = cfg.get(tSide, {}).get("name")
        tLoc = tByName.get(tName)
        if tLoc is None:
            raise HardwareError(f"{tSide} 손목 카메라 '{tName}' 가 USB 목록에 없음 -- 연결 확인\n{lst}")
        tNew[tSide]["location"] = tLoc
        tNew[tSide]["index"] = index_for_location(tLoc, candidates=REMAP_INDEX_CANDIDATES)
        say(
            f"손목 카메라 자동 매핑: {tSide} = '{tName}' -> USB {tLoc}, OpenCV index {tNew[tSide]['index']} "
            f"(전: {cfg.get(tSide, {}).get('location')}, index {cfg.get(tSide, {}).get('index')})"
        )
    CAMS_FILE.with_name(CAMS_FILE.name + ".bak_auto").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    tNew["cam_key"] = camKey()
    tNew["probed"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (auto by name)"
    if "wave_test" in tNew:
        tNew["wave_test"]["left"], tNew["wave_test"]["right"] = tNew["left"]["location"], tNew["right"]["location"]
    CAMS_FILE.write_text(json.dumps(tNew, indent=1, ensure_ascii=False))
    return tNew


def recorderRunning() -> list:
    tOut = subprocess.run(["pgrep", "-fl", RECORDER_PATTERN], capture_output=True, text=True).stdout.strip()
    return [
        tLine
        for tLine in tOut.splitlines()
        if str(os.getpid()) not in tLine.split()[0] and not any(f"umi_jaw {tMode}" in tLine for tMode in CAMERA_FREE_UMI_JAW)
    ]


def umiLocationWarning(cfg: dict) -> list:
    """10-23 user: the robot LEFT wrist camera sits on the USB port the left UMI camera used.  umi_jaw.LOCATIONS (UMI recorder) is
    NOT changed here -> warn loudly whenever a robot wrist location matches a UMI recorder prefix (the recorder would open the robot
    cam)."""
    try:
        from laundry_shared.visual_teleop.tools.umi_jaw import LOCATIONS
    except Exception:
        return []
    tOut = []
    for tSide in SIDES:
        for tHand, tPrefix in LOCATIONS.items():
            if cfg.get(tSide, {}).get("location", "").lower().startswith(tPrefix.lower()):
                tOut.append(
                    f"경고: 로봇 {tSide} 손목 카메라 USB {cfg[tSide]['location']} = UMI 녹화기 umi_jaw LOCATIONS['{tHand}'] = {tPrefix} 와 같은 포트 -- "
                    f"이대로 UMI 녹화(tag_rig_record --wrist / umi_jaw record --hand {tHand})를 하면 로봇 카메라가 열림. UMI 녹화 전에 UMI 카메라로 바꿔 꽂으세요"
                )
    for tWarning in tOut:
        print(tWarning, flush=True)
    return tOut


def _openWrist(index: int):
    tCap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    tCap.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_W)
    tCap.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_H)
    tCap.set(cv2.CAP_PROP_FPS, WRIST_FPS)
    return tCap


class WristCam(threading.Thread):
    """Robot wrist Arducam at a USB location: 1920x1080 @30, fixed exposure + IMAGE_CTRL (umi_jaw.fixed_exposure); every frame is
    pre-processed here (core.images.prepWrist -> 384x216 RGB) so the planner never waits on resizing."""

    def __init__(
        self, side: str, loc: str, index: int, exposure: int = WRIST_EXPOSURE, align: dict | None = None, prep: bool = True
    ) -> None:
        super().__init__(daemon=True)
        self.mSide, self.mLoc, self.mIndex, self.mAlign, self.mPrep = side, loc, index, align, prep
        tCap = _openWrist(index)
        if not tCap.isOpened():
            raise HardwareError(f"{side} 손목 카메라 index {index} 를 열 수 없음")
        from laundry_shared.visual_teleop.tools.umi_jaw import fixed_exposure

        self.mExpo = fixed_exposure(tCap, loc, exposure)
        if self.mExpo and self.mExpo.get("mean", 0) > WASHED_OUT_MEAN:
            subprocess.run([UVC_UTIL, "-L", loc, "-s", BRIGHTNESS_DOWN], capture_output=True)
            for _ in range(FLUSH_FRAMES):
                tCap.read()
            _, tFrame = tCap.read()
            say(
                f"[집게] {side} 너무 밝음 (평균 {self.mExpo['mean']:.0f}) -> brightness -64, 평균 {cv2.cvtColor(tFrame, cv2.COLOR_BGR2GRAY).mean():.0f}"
            )
        self.mCap = tCap
        self.mRaw = None
        self.mRing = Ring()
        self.mFrames = 0
        self.mStop = False
        self.mWb = None
        self.mReopens = 0
        self.mReopening = False
        self.mStalled = False
        self.mTLast = None

    def whiteBalance(self, frame: np.ndarray) -> np.ndarray:
        """[10-29] the right Arducam's colour drifts (yellow/green; UVC white-balance-temp has no visible effect on it) and its SHADOWS
        go orange while the table is grey -> per-channel levels: black point (p1) and mid-grey (pixels 90..235, mostly table) of every
        channel mapped onto common targets (min black, max grey), smoothed over ~2 s, applied by LUT."""
        if (
            self.mFrames % WB_EVERY_N == 0
        ):  # knots per channel: black (p1), shadow (grey 25..80 pixels), mid-grey (90..235) -> neutral targets
            tS = frame[::WB_STRIDE, ::WB_STRIDE].reshape(-1, 3).astype(np.float32)
            tGrey = tS.mean(1)
            tMid = tS[(tGrey > WB_MID_GREY[0]) & (tGrey < WB_MID_GREY[1])]
            tDark = tS[(tGrey > WB_SHADOW[0]) & (tGrey < WB_SHADOW[1])]
            if len(tMid) > WB_MIN_MID:
                tShadow = tDark.mean(0) if len(tDark) > WB_MIN_SHADOW else np.full(3, np.nan)
                tNew = np.r_[np.percentile(tS, WB_BLACK_PERCENTILE, axis=0), tShadow, tMid.mean(0)]
                if self.mWb is None:
                    self.mWb = tNew
                else:
                    self.mWb = np.where(
                        np.isnan(self.mWb), tNew, WB_KEEP * self.mWb + WB_GAIN * np.where(np.isnan(tNew), self.mWb, tNew)
                    )
        if self.mWb is None:
            return frame
        tLo, tShadow, tHi = self.mWb[:3], self.mWb[3:6], self.mWb[6:]
        tX = np.arange(LUT_SIZE, dtype=np.float32)
        tLuts = []
        for tC in range(3):
            tKx, tKy = [tLo[tC], tHi[tC]], [tLo.min(), tHi.max()]
            if np.isfinite(tShadow).all() and tLo[tC] + WB_KNOT_GAP < tShadow[tC] < tHi[tC] - WB_KNOT_GAP:
                tKx.insert(1, tShadow[tC])
                tKy.insert(1, tShadow.mean())
            tLuts.append(np.clip(np.interp(tX, [0.0] + tKx + [255.0], [0.0] + tKy + [255.0]), 0, 255).astype(np.uint8))
        return cv2.merge([cv2.LUT(tChannel, tLuts[tI]) for tI, tChannel in enumerate(cv2.split(frame))])

    def _reopen(self) -> None:
        """[10-29] the left Arducam stopped delivering frames 3x today (device still enumerated; a deploy run then starved on stale
        images and stood still) -> release and reopen the same camera index, re-apply the fixed exposure."""
        self.mReopens += 1
        try:
            self.mCap.release()
        except Exception:
            pass
        time.sleep(REOPEN_RELEASE_S)
        if (
            self.mLoc.lower() not in uvcList().lower()
        ):  # unplugged: opening a missing AVFoundation index segfaulted the UI (10-29)
            say(f"[손목] {self.mSide} 카메라가 USB 목록에 없음 ({self.mLoc}) -- 다시 꽂을 때까지 대기")
            time.sleep(REOPEN_RETRY_S)
            return
        try:
            tIdx = (
                self.mIndex
            )  # same OpenCV index (uvc-util's index is NOT the AVFoundation order); a replug -> UI restart remaps
            tCap = _openWrist(tIdx)
            if not tCap.isOpened():
                raise RuntimeError(f"index {tIdx} not opened")
            from laundry_shared.visual_teleop.tools.umi_jaw import fixed_exposure

            fixed_exposure(tCap, self.mLoc, WRIST_EXPOSURE)
            self.mCap = tCap
            self.mIndex = tIdx
            say(f"[손목] {self.mSide} 카메라 프레임 멈춤 -> 다시 엶 ({self.mReopens}회, index {tIdx})")
        except Exception as tExc:
            say(f"[손목] {self.mSide} 카메라 다시 열기 실패 ({type(tExc).__name__}: {tExc}) -- 2초 뒤 재시도")
            time.sleep(REOPEN_RETRY_S)

    def _watchdog(self) -> None:
        """A stalled AVFoundation read() can block forever: release the capture to unblock it."""
        while not self.mStop:
            time.sleep(WATCHDOG_PERIOD_S)
            if time.monotonic() - self.mTLast > STALL_S and not self.mReopening:
                self.mStalled = True
                try:
                    self.mCap.release()
                except Exception:
                    pass

    def run(self) -> None:
        self.mTLast = time.monotonic()
        threading.Thread(target=self._watchdog, daemon=True).start()
        while not self.mStop:
            tOk, tFrame = self.mCap.read()
            tT = time.monotonic_ns()
            if tOk and tFrame is not None:
                tFrame = self.whiteBalance(tFrame)
                self.mRaw = tFrame
                self.mFrames += 1
                self.mTLast = time.monotonic()
                if self.mPrep:
                    self.mRing.put(tT, prepWrist(tFrame, align=self.mAlign))
            elif self.mStalled or time.monotonic() - self.mTLast > STALL_S:
                self.mReopening = True
                self._reopen()
                self.mStalled = False
                self.mTLast = time.monotonic()
                self.mReopening = False
            else:
                time.sleep(IDLE_READ_S)
        self.mCap.release()

    def get(self):
        return self.mRaw, 0


class FakeWristCam(threading.Thread):
    """[10-28] DEPLOY_FAKE_WRIST=<side>: the wrist camera of <side> is missing (not enumerated) -> black 384x216 frames at 30 Hz on
    the same ring interface (user: try 3 policy steps without it).  Out of training distribution -- test only, flagged in the log."""

    def __init__(self, side: str) -> None:
        super().__init__(daemon=True)
        self.mSide = side
        self.mRaw = np.zeros((WRIST_H, WRIST_W, 3), np.uint8)
        self.mRing = Ring()
        self.mFrames = 0
        self.mStop = False
        self.mImg = np.zeros((IMG_WH[1], IMG_WH[0], 3), np.uint8)

    def run(self) -> None:
        while not self.mStop:
            self.mRing.put(time.monotonic_ns(), self.mImg)
            self.mFrames += 1
            time.sleep(1 / FAKE_WRIST_HZ)

    def get(self):
        return self.mRaw, 0


class GlobalCam(threading.Thread):
    """orbbec_server /frame (848x480 JPEG q95, capture time X-Capture-Monotonic-Ns) polled at ~30 Hz; or dir:<take> = recorded
    color_*.jpg replayed at 15 Hz (TEST ONLY: not the live scene)."""

    def __init__(self, src: str) -> None:
        super().__init__(daemon=True)
        self.mSrc = src
        self.mRing = Ring()
        self.mJpg = None
        self.mFrames = 0
        self.mStop = False
        self.mLastIdx = None
        if src.startswith(REPLAY_PREFIX):
            tDir = pathlib.Path(src[len(REPLAY_PREFIX) :])
            self.mFiles = sorted(tDir.glob("color_*.jpg"))
            self.mK = 0
            if not self.mFiles:
                raise HardwareError(f"{tDir} 에 color_*.jpg 없음")
        else:
            import requests

            self.mSession = requests.Session()
            self.mUrl = src.rstrip("/") + "/frame"
            try:
                tReply = self.mSession.get(self.mUrl, timeout=ORBBEC_PROBE_TIMEOUT_S)
                tOk = tReply.status_code == HTTP_OK
            except Exception as tExc:
                tOk = False
                tReply = tExc
            if not tOk:
                raise HardwareError(
                    f"Orbbec 서버 {self.mUrl} 응답 없음 ({tReply}) -- 먼저: sudo ~/ego_collector/.venv/bin/python ~/laundry_system/robot/orbbec_server.py"
                )

    def run(self) -> None:
        while not self.mStop:
            if self.mSrc.startswith(REPLAY_PREFIX):
                tBytes = self.mFiles[self.mK % len(self.mFiles)].read_bytes()
                self.mK += 1
                tT = time.monotonic_ns()
                time.sleep(1 / REPLAY_HZ)
            else:
                try:
                    tReply = self.mSession.get(self.mUrl, timeout=ORBBEC_TIMEOUT_S)
                    if tReply.status_code != HTTP_OK or tReply.headers.get("X-Frame-Index") == self.mLastIdx:
                        time.sleep(ORBBEC_IDLE_S)
                        continue
                    self.mLastIdx = tReply.headers.get("X-Frame-Index")
                    tBytes = tReply.content
                    tT = int(tReply.headers.get("X-Capture-Monotonic-Ns", time.monotonic_ns()))
                except Exception:
                    time.sleep(ORBBEC_ERROR_S)
                    continue
            tFrame = cv2.imdecode(np.frombuffer(tBytes, np.uint8), cv2.IMREAD_COLOR)
            self.mJpg = tBytes
            self.mFrames += 1
            self.mRing.put(tT, (prepGlobal(tFrame), tBytes))
