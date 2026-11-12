"""Run control behind the web UI (deploy stage): UiApp owns the started Rig and, around it, the bridge status poller, the GO
heartbeat watchdog, the runs (UI settings -> run arguments, incl. the GO relaxations), the motions (READY, start pose, park), the
live views and /api/status. Was tagrig_ui/control.py App.
"""

from __future__ import annotations
import argparse
import threading
import time
import traceback

import cv2
import numpy as np

from ..core import height_calib as heightCalib
from ..core.camera_model import project, robotCam
from ..core.contract import CHUNK_ROWS, SIDES
from ..core.errors import DeployError, HardwareError
from ..core.kinematics import fk
from ..session.cli import buildParser
from ..session.report import ARM_COLOURS_BGR
from ..session.rig import Rig
from ..session.run_loop import runSession
from ..session.start_pose import startCmd
from .ready_move import moveReady
from .run_control import UiRunControl
from .settings import CKPT_DIRS, HB_TIMEOUT_S, listCkpts, loadSettings, saveSettings

RO_BRIDGE_TIMEOUT_S = 0.5
LIVE_BRIDGE_TIMEOUT_S = 0.5
POLL_PERIOD_S = 0.5
WATCHDOG_PERIOD_S = 0.05
WATCHDOG_HOLD_PAUSE_S = 0.5
JAW_PERIOD_S = 0.5
STATUS_TIMEOUT_S = 1.0
CLEAR_ESTOP_TIMEOUT_S = 2.0
CLEAR_ESTOP_SETTLE_S = 0.5
CLEAR_COLLISION_TIMEOUT_S = 20.0
CLEAR_ESTOP_UI_TIMEOUT_S = 3.0
REST_TIMEOUT_S = 3.0
STOP_TIMEOUT_S = 2.0
XVLA_STATUS_URL = "http://127.0.0.1:8775/status"
XVLA_STATUS_TIMEOUT_S = 0.3
UMI_REF_WH = (1280, 720)
WRIST_VIEW_WH = (640, 360)
GLOBAL_JPEG_QUALITY, WRIST_JPEG_QUALITY = 75, 70
K1_SHOWN = 300
RTC_BETA = 5.0
DEFAULT_START_ROOT = "flat300_A_gold281_train"
# GO relaxations (10-28 .. 10-29, user: the operator approves every chunk on screen; collision / table / joint-limit / workspace /
# IK
# STOPs stay, see runSession): image / plan age for a robot that is still between steps (Mac DiT plan 0.2-0.45 s; X-VLA ~0.9 s on
# MPS),
# continue from the CURRENT pose (no READY / start-pose check), IK misses never stop ("리스트릭션 다 없애줘", closest reachable pose),
# re-seed only on a real drive dropout (10-29: normal tracking lag 50-80 mm crossed 6 deg -> a jump back, "뒤로 돌아가는 움직임"), and
# the speed caps at the bridge maximum (10-28: joint 30 -> 60 deg/s, TCP 0.25 -> 0.5 m/s; then 90 deg/s, 1 m/s, 360 deg/s)
GO_MAX_IMG_AGE_MS, GO_MAX_PLAN_AGE_S = 600.0, 1.5
GO_READY_TOL_DEG = 180.0
GO_MAX_IK_STOP_MM = 1e9
GO_RESYNC_DEG = 20.0
GO_MAX_DEG_S, GO_MAX_TCP_MPS, GO_MAX_ROT_DPS = 90.0, 1.0, 360.0
GRIP_MODES_SHAPED = ("sigmoid", "relu")
GRIP_GAIN_DEFAULT = 10.0
BLACK, WHITE, YELLOW, GREEN = (0, 0, 0), (255, 255, 255), (0, 255, 255), (0, 255, 0)


def _jawNumbers(measurement: dict) -> dict:
    return dict(
        area=round(measurement["jaw_area_pct"], 1),
        top=measurement.get("jaw_top_y"),
        axis=round(measurement.get("jaw_axis_deg_from_vertical", float("nan")), 0),
    )


class UiApp:
    """Was App."""

    def __init__(self, args) -> None:
        self.mArgs = args
        self.mSettings = loadSettings()
        saveSettings(self.mSettings)
        tRunArgv = (
            ["run"]
            + (["--replay-global", args.replay_global] if args.replay_global else [])
            + (["--global-url", args.global_url])
            + (["--bridge-url", args.bridge_url] if args.bridge_url else [])
        )
        self.mRunArgs = buildParser().parse_args(tRunArgv)
        self.mRig = Rig(self.mRunArgs).start()
        self.mLock = threading.Lock()
        self.mCtl = None
        self.mThread = None
        self.mMode = "idle"
        self.mMsg = "준비됨"
        self.mLast = None
        self.mBusy = None
        self.mBridgeStatus = {}
        self.mMotors = {}
        self.mXvla = None
        from laundry_shared.visual_teleop.robot.bridge import HttpBridge

        self.mBridgeRo = HttpBridge(self.mRig.mUrl, live=False, timeout_s=RO_BRIDGE_TIMEOUT_S)
        threading.Thread(target=self._pollBridge, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()
        self.mUmiRef = self._umiRefs()
        self.mJaw = {}
        threading.Thread(target=self._jawLoop, daemon=True).start()

    # ---- background
    def _pollBridge(self) -> None:
        while True:
            try:
                self.mBridgeStatus = self.mBridgeRo.s.get(self.mRig.mUrl + "/status", timeout=RO_BRIDGE_TIMEOUT_S).json()
            except Exception as tExc:
                self.mBridgeStatus = {"error": f"{type(tExc).__name__}"}
            try:
                self.mMotors = self.mBridgeRo.s.get(self.mRig.mUrl + "/motor_states", timeout=RO_BRIDGE_TIMEOUT_S).json()
            except Exception:
                self.mMotors = {}
            time.sleep(POLL_PERIOD_S)

    def _watchdog(self) -> None:
        """Server-side: GO running and no heartbeat for > HB_TIMEOUT_S -> /estop at once (hold), even if runSession is inside an inference."""
        while True:
            tCtl = self.mCtl
            if tCtl is not None and tCtl.mGo and self.mMode == "go" and time.monotonic() - tCtl.mHb > HB_TIMEOUT_S:
                tCtl.mStopReq = tCtl.mStopReq or f"UI heartbeat lost (> {HB_TIMEOUT_S:.0f} s)"
                self.mBridgeRo.estop()
                self.mMode = "hold"
                self.mMsg = "하트비트 끊김 -> HOLD (E-STOP 홀드)"
                print(
                    f"[UI] watchdog HOLD {time.strftime('%H:%M:%S')} (heartbeat {time.monotonic() - tCtl.mHb:.2f} s old)",
                    flush=True,
                )
                time.sleep(WATCHDOG_HOLD_PAUSE_S)
            time.sleep(WATCHDOG_PERIOD_S)

    def _umiRefs(self) -> dict:
        try:
            from ..tools.domain_gap import umiFrames

            tFrames = umiFrames(1, startOnly=True)
            return {tSide: cv2.imread(str(tFrames[tSide][len(tFrames[tSide]) // 2])) for tSide in SIDES}
        except Exception:
            return {}

    def _jawLoop(self) -> None:
        from ..tools.domain_gap import measureFrame

        while True:
            if self.mSettings.get("show_umi_ref"):
                for tSide in SIDES:
                    tFrame = self.mRig.mWristCams[tSide].mRaw
                    if tFrame is not None:
                        tM, _, _ = measureFrame(cv2.resize(tFrame, UMI_REF_WH, interpolation=cv2.INTER_AREA))
                        self.mJaw[tSide] = _jawNumbers(tM)
                if not self.mJaw.get("umi") and self.mUmiRef:
                    self.mJaw["umi"] = {tSide: _jawNumbers(measureFrame(self.mUmiRef[tSide])[0]) for tSide in SIDES}
            time.sleep(JAW_PERIOD_S)

    # ---- runs
    def startRun(self, go: bool, step: bool = False):
        with self.mLock:
            if self.mThread is not None and self.mThread.is_alive():
                return False, "이미 실행 중"
            if self.mBusy:
                return False, f"{self.mBusy} 중"
            if go and self.mArgs.dry_only:
                return False, "이 UI 는 --dry-only 로 실행됨 (GO 비활성)"
            if go and self.mRig.mReplay:
                return False, "녹화 프레임(replay) 모드에서는 GO 불가"
            if go and not self.mRig.mWaveOk:
                return False, "손목 카메라 wave 테스트 미확인"
            tS = self.mSettings
            tRa = self.mRunArgs
            tKind = tS["policy"]
            tRa.policy = tKind
            tRa.ckpt = tS[f"ckpt_{tKind}"]
            tRa.commit = int(tS[f"commit_{tKind}"])
            tRa.speed = float(tS["speed"])
            tRa.slow_start_speed = float(tS["slow_start_speed"])
            tRa.secs = float(tS["secs"])
            tRa.dit_steps = int(tS["dit_steps"])
            tRa.go = bool(go)
            tRa.allow_start_pose = True  # READY or start_pose.json
            tRa.step_mode = bool(go and step)
            tRa.secs = 0.0 if tRa.step_mode else tRa.secs  # step mode: no time limit (the user drives it)
            tRa.active_arms = tS.get("active_arms", "both")  # 10-28: one-arm execution (other arm dry)
            tRa.grip_fake_open = bool(tS.get("grip_fake_open", False))  # 10-29 diagnostic: jaw state fed as OPEN
            tGripMode = tS.get(
                "grip_mode", "linear" if tS.get("grip_cont", True) else "binary"
            )  # 10-30: binary | linear | sigmoid
            tRa.grip_continuous = tGripMode == "linear"
            tRa.grip_mode = tGripMode if tGripMode in GRIP_MODES_SHAPED else ""
            tRa.grip_gain = float(tS.get("grip_gain", GRIP_GAIN_DEFAULT))
            tRa.z_offset_mm = float(tS.get("z_offset_mm", 0.0))  # 10-30 height correction (not a limit)
            if tS.get("z_floor_mm") is not None:  # 11-06 robot-frame floor for the TCP targets (settings file key; absent = off)
                tRa.z_floor_mm = float(tS["z_floor_mm"])
            if tS.get("grip_thr") is not None:  # 11-06 user: binary jaw opens only above this model output (absent = cli 0.5)
                tRa.grip_thr = float(tS["grip_thr"])
            if tS.get("grip_at") is not None:  # 11-06: "row" = the jaw follows every committed chunk row (absent = cli mid)
                tRa.grip_at = str(tS["grip_at"])
            if tS.get("grip_hold_s") is not None:  # 11-06: a closed jaw may not re-open for this many seconds (absent = off)
                tRa.grip_hold_s = float(tS["grip_hold_s"])
            if tS.get("grip_lead_rows") is not None:  # 11-06: jaw rows taken this many chunk rows ahead (grip_at row; absent = 0)
                tRa.grip_lead_rows = int(tS["grip_lead_rows"])
            # 11-06 UMI latency matching: send-ahead + capture-time state from $TAGRIG_STATE_DIR/latency.json (absent = off)
            tRa.latency_match = bool(tS.get("latency_match", False))
            # 11-06 user "땅밑으로 가면 토크로": per-arm floor raised at a measured table contact (core.contact; absent = off)
            tRa.contact_floor = bool(tS.get("contact_floor", False))
            tRa.contact_gap_mm = float(tS.get("contact_gap_mm", 8.0))
            tRa.contact_tau_nm = float(tS.get("contact_tau_nm", 2.0))
            tRa.amp = float(tS.get("amp", 1.0))  # 11-03 motion amplitude scale (1 = as predicted)
            tRa.height_calib = bool(
                tS.get("height_calib", False)
            )  # 11-01 robot-measured per-arm table height (tools.height_calib)
            # 10-30: temporal ensembling averaged old plans anchored on the MEASURED (lagging) TCP -> every new no-look-ahead
            # chunk was
            # pulled ~45 mm back ("백포지션").  Off unless the user ticks it.
            if not tS.get("tens", False):
                tRa.temporal_ensemble = 0.0
            tRa.no_lookahead = tS.get("rtc") == "nolook"  # 10-30: no look-ahead (whole commit, then infer)
            # 10-29 Real-Time Chunking (GO look-ahead only). Offline (scratchpad rtc_offline_test.py): blend = prefix 0 mm,
            # overlap
            # -30..35 %, no extra latency; guided ~no effect on the DDIM eps model
            tRa.rtc = "off" if tRa.no_lookahead else tS.get("rtc", "blend")
            tRa.rtc_beta = RTC_BETA
            if go:  # 10-29 user: "스텝모드 말고 그냥" -- the same relaxations for the continuous GO too
                tRa.max_img_age_ms, tRa.max_plan_age_s = GO_MAX_IMG_AGE_MS, GO_MAX_PLAN_AGE_S
                tRa.ready_tol_deg = GO_READY_TOL_DEG
                tRa.max_ik_stop_mm = GO_MAX_IK_STOP_MM
                tRa.resync_deg = GO_RESYNC_DEG
                tRa.no_safety = True  # collision / table / joint-limit / workspace STOPs -> WARN (as the 10-27 label replay)
                tRa.no_floor = True  # 10-29 user "바닥 제한을 없애줘": no finger-corner table floor (cornerLift) -- targets may reach the table
                tRa.max_deg_s, tRa.max_tcp_mps, tRa.max_rot_dps = GO_MAX_DEG_S, GO_MAX_TCP_MPS, GO_MAX_ROT_DPS
                try:  # resume after a HOLD: the bridge keeps holding the measured pose
                    if (self.mBridgeRo.s.get(self.mRig.mUrl + "/status", timeout=STATUS_TIMEOUT_S).json() or {}).get("estop"):
                        self.mBridgeRo.s.post(self.mRig.mUrl + "/clear_estop", timeout=CLEAR_ESTOP_TIMEOUT_S)
                        time.sleep(CLEAR_ESTOP_SETTLE_S)
                except Exception as tExc:
                    return False, f"E-STOP 해제 실패: {type(tExc).__name__}"
            self.mCtl = UiRunControl(go)
            self.mCtl.mStepMode = tRa.step_mode
            self.mMode = "go" if go else "dry"
            self.mMsg = (
                "실행 (GO, 스텝 모드: 버튼 한 번 = 한 구간)" if tRa.step_mode else "실행 (GO)" if go else "DRY-RUN"
            ) + " 시작 중..."
            self.mThread = threading.Thread(target=self._run, args=(tRa, self.mCtl), daemon=True)
            self.mThread.start()
            return True, "시작"

    def _run(self, runArgs, ctl) -> None:
        try:
            tSummary = runSession(runArgs, self.mRig, ctl)
            self.mLast = tSummary or {}
            self.mMsg = f"끝: {(tSummary or {}).get('stop_reason')}" + (
                " -- 로봇은 현재 자세 유지 (HOLD). 다음 동작은 버튼으로만" if runArgs.go else ""
            )
        except (SystemExit, DeployError) as tExc:
            self.mMsg = f"시작 안 됨: {tExc}"
        except Exception as tExc:
            self.mMsg = f"오류: {type(tExc).__name__}: {tExc}"
            traceback.print_exc()
        finally:
            self.mMode = "hold" if runArgs.go else "idle"

    def step(self):
        tCtl = self.mCtl
        if tCtl is None or self.mMode != "go" or not tCtl.mStepMode:
            return False, "스텝 모드 실행 중이 아님"
        with tCtl.mLock:
            tCtl.mStepN = 1  # at most one pending step
        return True, f"한 스텝 요청 (지금까지 {tCtl.mStepsDone} 스텝)"

    def hold(self):
        tCtl = self.mCtl
        tGoRunning = self.mMode == "go"
        if tCtl is not None:
            tCtl.mStopReq = tCtl.mStopReq or "정지 / HOLD (UI)"
        # 11-01: soft stop -- a running GO brakes onto its current command (runSession), idle arms get the bridge /stop (hold, no
        # latch)
        if not self.mArgs.dry_only and not tGoRunning and self.mMode in ("hold", "idle"):
            try:
                self.mBridgeRo.s.post(self.mRig.mUrl + "/stop", timeout=STOP_TIMEOUT_S)
            except Exception:
                pass
        if tGoRunning:
            self.mMode = "hold"
        self.mMsg = (
            "정지 (DRY 전용 UI: 브리지 명령 없음)" if self.mArgs.dry_only else "정지 -- 감속 후 현재 자세 유지 (E-STOP 아님)"
        )
        return True, self.mMsg

    def estop(self):
        """11-01: the hard stop kept as its own button: latch the bridge E-STOP (instant hold at the measured pose)."""
        tCtl = self.mCtl
        if tCtl is not None:
            tCtl.mStopReq = tCtl.mStopReq or "E-STOP (UI)"
        if not self.mArgs.dry_only:
            self.mBridgeRo.estop()
        if self.mMode == "go":
            self.mMode = "hold"
        self.mMsg = "E-STOP -- 즉시 정지 (현재 자세 유지, 해제 필요)"
        return True, self.mMsg

    def _motion(self, name: str, fn):
        with self.mLock:
            if (self.mThread is not None and self.mThread.is_alive()) or self.mBusy:
                return False, "실행 중에는 불가"
            if self.mArgs.dry_only:
                return False, "이 UI 는 --dry-only 로 실행됨 (로봇 이동 비활성)"
            self.mBusy = name

        def work():
            from laundry_shared.visual_teleop.robot.bridge import HttpBridge

            tBridge = None
            try:
                tBridge = HttpBridge(self.mRig.mUrl, live=True, timeout_s=LIVE_BRIDGE_TIMEOUT_S)
                fn(tBridge)
                self.mMsg = f"{name} 끝"
            except (SystemExit, DeployError) as tExc:
                self.mMsg = f"{name} 중단: {tExc}"
            except Exception as tExc:
                self.mMsg = f"{name} 오류: {type(tExc).__name__}: {tExc}"
                traceback.print_exc()
            finally:
                if tBridge is not None:
                    tBridge.close()
                self.mBusy = None
                self.mMode = "idle"

        threading.Thread(target=work, daemon=True).start()
        return True, f"{name} 시작"

    def goReady(self):
        return self._motion("READY로 이동", lambda tBridge: moveReady(tBridge, self.mRig))

    def goStart(self):
        """10-23 user: READY -> training start pose from the UI = the deploy `start --go` (same plan + checks, its own bridge client)."""

        def work(bridge):
            bridge.close()  # startCmd opens its own live client; keep a single writer
            # 10-29: GOLD-281 medoid start (ep 104), no lift: the training starts sit at the tag edge (L 291/148/33, R 305/-216/28
            # mm;
            # plan: IK <= 3 mm, link-table 24 mm) -- was the OLD dataset ep 82 + 3 cm (5-6 cm off).
            # 11-03: settings "start_root" picks the medoid dataset (FLAT149 flatten: flatS05C_train149 -- its starts sit 19/31 mm
            # (L/R) from GOLD-281's)
            startCmd(
                argparse.Namespace(
                    go=True,
                    episode=-1,
                    # 11-06: the 11-03..05 flatten starts put the jaw tip ON the table (medoid ep 161: tcp z 0 mm, link -5 mm) ->
                    # settings "start_lift_m" (0.02 passes: link-table 14 mm); the policy's first chunk goes down from there
                    start_lift=float(self.mSettings.get("start_lift_m", 0.0)),
                    deg_s=10.0,
                    assume_ready=False,
                    root=self.mSettings.get("start_root") or DEFAULT_START_ROOT,
                    label="",
                    height_calib=bool(self.mSettings.get("height_calib", False)),
                )
            )

        return self._motion("시작 자세로 이동", work)

    def clearEstop(self):
        """10-30 user "e stop 푸는 버튼도 넣어줘": idle only.  /clear_estop; if the bridge refuses because an arm's torque is off (the guard
        stays latched), try /clear_collision once.  Nothing moves: the arms keep holding the measured pose."""
        if self.mMode in ("go", "dry") or self.mBusy:
            return False, "실행/이동 중에는 해제할 수 없습니다"
        try:
            tUrl = self.mRig.mUrl
            tReply = self.mBridgeRo.s.post(tUrl + "/clear_estop", timeout=CLEAR_ESTOP_UI_TIMEOUT_S).json()
            if not tReply.get("ok", True) or tReply.get("error"):
                # 10-30: after a left-arm power cycle /reenable succeeded but the guard stayed latched; /clear_collision clears
                # the latch,
                # re-enables both arms at the measured pose and releases E-STOP in one call
                tCollision = self.mBridgeRo.s.post(tUrl + "/clear_collision", timeout=CLEAR_COLLISION_TIMEOUT_S).json()
                tReply = {"ok": True} if tCollision.get("ok") else tCollision
                if not tReply.get("ok", True) or tReply.get("error"):
                    return (
                        False,
                        f"E-STOP 해제 실패: {tReply.get('error', tReply)} | clear_collision: {tCollision.get('arms', tCollision)} -- 팔 토크/모터 커넥터 확인",
                    )
            self.mMode = "idle"
            self.mMsg = "E-STOP 해제됨 (팔은 현재 자세 유지)"
            return True, self.mMsg
        except Exception as tExc:
            return False, f"E-STOP 해제 실패: {type(tExc).__name__}: {tExc}"

    def park(self):
        def work(bridge):
            tStatus = bridge.status()
            if tStatus.get("estop"):
                bridge.s.post(bridge.base_url + "/clear_estop", timeout=CLEAR_ESTOP_TIMEOUT_S)
            tReply = bridge.s.post(bridge.base_url + "/rest", timeout=REST_TIMEOUT_S).json()
            if not tReply.get("ok"):
                raise HardwareError(f"/rest 거부: {tReply}")

        return self._motion("휴식(park)", work)

    # ---- views
    def tcp(self):
        _, (tQ, _, _) = self.mRig.mPoller.mRing.latest()
        if tQ is None:
            return None
        return {tSide: np.round(fk(self.mRig.mArms, tSide, tQ[tSide])[:3, 3], 3).tolist() for tSide in SIDES}

    def globalJpg(self):
        _, tLatest = self.mRig.mGlobalCam.mRing.latest()
        if tLatest is None:
            return None
        tImg = cv2.imdecode(np.frombuffer(tLatest[1], np.uint8), cv2.IMREAD_COLOR)
        tCamFromBase, tK = robotCam()
        tCtl = self.mCtl
        # 11-01 user "UI 에서 TCP 가 벗어나": the Orbbec extrinsic is in the label B frame, FK is the robot's -> map FK points to B with
        # height_calib.json (per-arm xy shift from the tag camera ray + table-height plane) before projecting
        tCalib = heightCalib.loadCalibration()
        tSnap = tCtl.mSnap if (tCtl is not None and self.mMode in ("dry", "go")) else None
        _, (tQ, _, _) = self.mRig.mPoller.mRing.latest()
        if tQ is not None:
            for tSide in SIDES:
                tU = project(
                    tCamFromBase, tK, heightCalib.toBaseFrame(tCalib, tSide, fk(self.mRig.mArms, tSide, tQ[tSide])[None, :3, 3])
                )[0]
                if np.isfinite(tU).all():
                    cv2.circle(tImg, tuple(tU.astype(int)), 9, BLACK, 2)
        if tSnap is not None:
            for tSide in SIDES:
                tUv = project(
                    tCamFromBase, tK, heightCalib.toBaseFrame(tCalib, tSide, np.array([tT[:3, 3] for tT in tSnap["Tc"][tSide]]))
                )
                for tR in range(CHUNK_ROWS - 1):
                    if np.isfinite(tUv[tR]).all() and np.isfinite(tUv[tR + 1]).all():
                        cv2.line(tImg, tuple(tUv[tR].astype(int)), tuple(tUv[tR + 1].astype(int)), ARM_COLOURS_BGR[tSide], 2)
                if np.isfinite(tUv[0]).all():
                    cv2.circle(tImg, tuple(tUv[0].astype(int)), 7, ARM_COLOURS_BGR[tSide], -1)
                    cv2.circle(tImg, tuple(tUv[0].astype(int)), 8, WHITE, 1)
                if np.isfinite(tUv[-1]).all():
                    cv2.circle(tImg, tuple(tUv[-1].astype(int)), 5, ARM_COLOURS_BGR[tSide], 2)
        return cv2.imencode(".jpg", tImg, [cv2.IMWRITE_JPEG_QUALITY, GLOBAL_JPEG_QUALITY])[1].tobytes()

    def wristJpg(self, side: str):
        tFrame = self.mRig.mWristCams[side].mRaw
        if tFrame is None:
            return None
        tView = cv2.resize(tFrame, WRIST_VIEW_WH, interpolation=cv2.INTER_AREA)
        if self.mSettings.get("show_umi_ref") and side in self.mUmiRef:
            tRef = cv2.resize(self.mUmiRef[side], WRIST_VIEW_WH)
            tView = np.hstack([tView, tRef])
            cv2.putText(tView, "UMI train ref", (650, 24), 0, 0.7, YELLOW, 2)
            tJaw = self.mJaw.get(side)
            tJawUmi = (self.mJaw.get("umi") or {}).get(side)
            if tJaw:
                cv2.putText(
                    tView, f"robot jaw area {tJaw['area']}% top {tJaw['top']} axis {tJaw['axis']}", (8, 350), 0, 0.55, GREEN, 2
                )
            if tJawUmi:
                cv2.putText(
                    tView,
                    f"UMI jaw area {tJawUmi['area']}% top {tJawUmi['top']} axis {tJawUmi['axis']}",
                    (650, 350),
                    0,
                    0.55,
                    GREEN,
                    2,
                )
        cv2.putText(tView, f"{side.upper()} {self.mRig.mCamsCfg[side]['location']}", (8, 24), 0, 0.7, YELLOW, 2)
        return cv2.imencode(".jpg", tView, [cv2.IMWRITE_JPEG_QUALITY, WRIST_JPEG_QUALITY])[1].tobytes()

    def apiStatus(self) -> dict:
        tCtl = self.mCtl
        tSnap = tCtl.mSnap if tCtl is not None else None
        tSt = self.mBridgeStatus or {}
        if self.mSettings.get("policy") == "xvla":
            try:
                tXs = self.mBridgeRo.s.get(XVLA_STATUS_URL, timeout=XVLA_STATUS_TIMEOUT_S).json()
                self.mXvla = dict(up=True, ckpt=tXs.get("ckpt"))
            except Exception:
                self.mXvla = dict(up=False)
        tMotorsOk = {
            f"{tKey}": all(tM.get("status") == 1 for tM in tV.values())
            for tKey, tV in (self.mMotors or {}).items()
            if isinstance(tV, dict)
        }
        tOut = dict(
            mode=self.mMode,
            msg=self.mMsg,
            busy=self.mBusy,
            settings=self.mSettings,
            dry_only=self.mArgs.dry_only,
            replay=self.mRig.mReplay,
            wave_ok=self.mRig.mWaveOk,
            xvla=self.mXvla,
            step_mode=bool(tCtl is not None and tCtl.mStepMode),
            step_pending=bool(tCtl is not None and tCtl.mStepN > 0),
            steps_done=(tCtl.mStepsDone if tCtl is not None else 0),
            bridge=dict(
                url=self.mRig.mUrl,
                connected=tSt.get("connected"),
                estop=tSt.get("estop"),
                parked=tSt.get("parked"),
                status=tSt.get("status"),
                error=tSt.get("error"),
                read_age_ms={tKey: tV.get("read_age_ms") for tKey, tV in (tSt.get("arms") or {}).items()},
                motors_ok=tMotorsOk,
            ),
            tcp=self.tcp(),
            ckpts={tKind: listCkpts(tKind) for tKind in CKPT_DIRS},
            orbbec_frames=self.mRig.mGlobalCam.mFrames,
            wrist_frames={tSide: self.mRig.mWristCams[tSide].mFrames for tSide in SIDES},
        )
        if tSnap is not None:
            tOut.update(
                t=round(tSnap["t"], 1),
                hz=round(tSnap["hz"], 1),
                lat_pol_ms=round(tSnap["lat_pol_ms"], 1),
                lat_total_ms=round(tSnap["lat_total_ms"], 1),
                k0=tSnap["k0"],
                flags=[tF for tF in tSnap["flags"] if not tF.startswith("LIM")],
                lims=[tF for tF in tSnap["flags"] if tF.startswith("LIM")],
                speed=tSnap["speed"],
                grip=np.round(tSnap["grip"], 3).tolist(),
                g_cmd=tSnap["g_cmd"],
                k1=[[round(tT, 2), round(tL, 3), round(tR, 3)] for tT, tL, tR in tCtl.mK1[-K1_SHOWN:]],
            )
        if self.mLast:
            tOut["last"] = {
                tKey: self.mLast.get(tKey)
                for tKey in (
                    "run",
                    "stop_reason",
                    "plan_hz",
                    "latency_ms",
                    "video",
                    "grip_plot",
                    "violation_cycles",
                    "flag_counts",
                )
            }
        return tOut
