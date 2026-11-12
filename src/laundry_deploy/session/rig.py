"""The persistent hardware / kinematics side shared by every run (CLI: one run; UI: many): wrist cameras, live Orbbec, 25 Hz robot
poller (read-only HttpBridge), arms + capsule collision model, and a cache of loaded policies.
"""

from __future__ import annotations
import json
import os
import pathlib
import time

import numpy as np

from ..core.collision import Collision
from ..core.contract import SIDES
from ..core.errors import HardwareError, RunRefused
from ..core.kinematics import makeArms
from ..core.policy_runner import PolicyRunner
from ..core.xvla_runner import XvlaRunner
from .bridge import RobotPoller
from .cameras import FakeWristCam, GlobalCam, WristCam, autoRemapCams, camKey, recorderRunning, umiLocationWarning, uvcList
from .paths import CAMS_FILE, START_FILE, say

FAKE_WRIST_ENV = "DEPLOY_FAKE_WRIST"  # comma list of sides whose wrist camera is replaced by black frames (test only)
MIN_BRIDGE_TIMEOUT_S = 0.3
FIRST_FRAME_TIMEOUT_S = 8
FRAME_POLL_S = 0.05
REPLAY_PREFIX = "dir:"
XVLA_CONFIG_MARK = '"xvla"'  # 10-28 X-VLA checkpoints run in xvla_server.py


def _fakeWristSides() -> list:
    return [tX for tX in os.environ.get(FAKE_WRIST_ENV, "").split(",") if tX in SIDES]


class Rig:
    def __init__(self, args) -> None:
        from laundry_shared.visual_teleop import config as vcfg
        from laundry_shared.visual_teleop.robot.bridge import HttpBridge
        from laundry_shared.visual_teleop.tools.robot_moves import READY_Q_DEG

        tRecorders = recorderRunning()
        if tRecorders:
            raise HardwareError(f"녹화/카메라 프로세스가 실행 중 -- 끄고 다시: {tRecorders}")
        self.mTeleopCfg = vcfg.load()
        self.mUrl = getattr(args, "bridge_url", None) or self.mTeleopCfg.robot["base_url"]
        self.mQReady = {tSide: np.radians(READY_Q_DEG[tSide]) for tSide in SIDES}
        tCfg = json.loads(CAMS_FILE.read_text())
        tList = uvcList()
        if not _fakeWristSides():
            tCfg = autoRemapCams(tCfg, tList)
        self.mCamsCfg = tCfg
        tFake = _fakeWristSides()
        if tFake:
            say(f"주의: {tFake} 손목 카메라 = 검은 화면 (DEPLOY_FAKE_WRIST, 학습 분포 밖 -- 테스트 전용)")
        for tSide in SIDES:
            if tSide in tFake:
                continue
            if not tCfg.get(tSide, {}).get("location") or tCfg[tSide]["location"].lower() not in tList.lower():
                raise HardwareError(
                    f"{tSide} 손목 카메라 USB 위치 {tCfg.get(tSide, {}).get('location')} 없음 -- 연결/매핑 확인 (wave 테스트)"
                )
        if tCfg.get("cam_key") != camKey() and not tFake:
            raise HardwareError("USB 카메라 구성이 바뀜 -- 먼저: act_tagrig_deploy.py cams (index 다시 찾기)")
        self.mWaveOk = bool(tCfg.get("wave_test", {}).get("confirmed"))
        if not self.mWaveOk:
            say("주의: 손목 카메라 좌우가 wave 테스트로 아직 확인되지 않음 (act_tagrig_cams.json wave_test)")
        umiLocationWarning(tCfg)
        self.mArms, self.mRobotArgs = makeArms()
        self.mCollision = Collision(self.mArms)
        self.mFakeQ = None
        if getattr(args, "assume_start", False):
            if not START_FILE.exists():
                raise RunRefused("start_pose.json 없음 -- 먼저 `start`")
            self.mFakeQ = {tSide: np.asarray(json.loads(START_FILE.read_text())["q"][tSide]) for tSide in SIDES}
        elif getattr(args, "assume_ready", False):
            self.mFakeQ = self.mQReady
        tTimeout = max(self.mTeleopCfg.robot["request_timeout_s"], MIN_BRIDGE_TIMEOUT_S)
        self.mBridge = None if self.mFakeQ is not None else HttpBridge(self.mUrl, live=False, timeout_s=tTimeout)
        self.mPoller = RobotPoller(
            self.mBridge, self.mFakeQ if self.mFakeQ is not None else self.mQReady, self.mFakeQ is not None
        )
        tAlign = tCfg.get("wrist_align") if getattr(args, "wrist_align", False) else None
        self.mReplay = bool(getattr(args, "replay_global", ""))
        self.mGlobalCam = GlobalCam((REPLAY_PREFIX + args.replay_global) if self.mReplay else args.global_url)
        self.mWristCams = {
            tSide: (
                FakeWristCam(tSide)
                if tSide in tFake
                else WristCam(tSide, tCfg[tSide]["location"], tCfg[tSide]["index"], args.exposure, (tAlign or {}).get(tSide))
            )
            for tSide in SIDES
        }
        self.mRunners = {}
        self.mPGrasp = None
        self.mJawFallback = {tSide: None for tSide in SIDES}  # the run's jaw command, used when the poller has no jaw reading

    def pGrasp(self, device: str):
        """[11-06] the robot-side p_grasp estimator (PGRASP checkpoints, core.pgrasp), started once and kept for every run. Jaw = the
        measured opening from the robot poller; without one (dry-run --assume-ready) the run's jaw command (mJawFallback)."""
        if self.mPGrasp is None:
            from ..core.pgrasp import PGraspEstimator

            def jaw() -> dict:
                tRead = self.mPoller.mRing.latest()[1]
                tG = tRead[1] if tRead is not None else None
                return {tSide: (tG[tSide] if tG and tG.get(tSide) is not None else self.mJawFallback[tSide]) for tSide in SIDES}

            self.mPGrasp = PGraspEstimator(self.mWristCams, jaw, device=device)
            self.mPGrasp.start()
        return self.mPGrasp

    def _threads(self) -> tuple:
        return (*self.mWristCams.values(), self.mGlobalCam, self.mPoller)

    def start(self) -> "Rig":
        for tThread in self._threads():
            tThread.start()
        tT0 = time.monotonic()
        while any(tCam.mRing.latest()[1] is None for tCam in (*self.mWristCams.values(), self.mGlobalCam)):
            if time.monotonic() - tT0 > FIRST_FRAME_TIMEOUT_S:
                tCounts = {tSide: tCam.mFrames for tSide, tCam in self.mWristCams.items()}
                raise HardwareError(f"카메라 프레임 없음: {tCounts} global {self.mGlobalCam.mFrames}")
            time.sleep(FRAME_POLL_S)
        return self

    def runner(self, ckpt, device: str, nSamples: int):
        tKey = (str(ckpt), nSamples)
        if tKey not in self.mRunners:
            self.mRunners = {}  # keep ONE model in memory (DiT is 1.1 GB)
            tConfig = pathlib.Path(ckpt) / "pretrained_model" / "config.json"
            tIsXvla = XVLA_CONFIG_MARK in tConfig.read_text() if tConfig.exists() else False
            self.mRunners[tKey] = XvlaRunner(ckpt) if tIsXvla else PolicyRunner(ckpt, device, nSamples=nSamples)
        return self.mRunners[tKey]

    def close(self) -> None:
        for tThread in self._threads():
            tThread.mStop = True
        if self.mPGrasp is not None:
            self.mPGrasp.mStop = True
        if self.mBridge is not None:
            self.mBridge.close()
