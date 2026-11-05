"""MIT-mode robot service for the bimanual reBot (HTTP, port MIT_PORT 8021), built on the colleague's bh_indy7_LeRobot reBot
plugin (10-01 user: "내 동료가 mit를 구현한게 있어 그걸 사용하자" / "중력보상 말고 integral버전으로 해줘").

Same HTTP contract as robot_service.py (POS_VEL), so clients work unchanged when pointed here. Runs in the colleague's env with
the robot-cockpit checkout (usb_arm_ports) on PYTHONPATH:
    PYTHONPATH=<repo>/src:~/robot-cockpit ~/bh_indy7_LeRobot/.venv/bin/python -m laundry_deploy.robot_service.mit_bridge
The POS_VEL service and this one cannot own the arm ports at the same time: stop the other one first.

Per arm (`hardware.buildArm`): ReBot(arm_control_mode="mit", INTEGRAL law) -> connect -> integral law armed as
bh_indy7_lerobot_gui/infer_cli.py does it (gravity model disarmed, Ki default 28 N*m/(rad*s)) -> arm torque on. Known cost: t_ff
starts at 0, so a loaded joint sags until the integral winds up (8.9 deg at the elbow at kp 64, recovered in ~6.5 s).

ONE control thread owns both buses at MIT_TICK_HZ (100): reads each arm, moves the commanded pose toward the goal
(`profile.profileStep`, at most MIT_MAX_DEG_S per joint, acceleration-limited), sends one MIT frame per arm. HTTP handlers only
swap goals and read the published snapshot. Faults are recovered in place (no automatic E-STOP, 10-01 user); the operator
E-STOP holds both arms. 10-02 goal interpolation (MIT_GOAL_INTERP=1) and PLAN playback track straight segments between goals
(`profile.profileStepInterp`). 10-02 park: with MIT_REST_TORQUE_OFF=1 /rest ends with arm torque off; the next motion command
re-enables in place at the measured pose.

Frames (`jaw`): /execute_step takes the robot_service 14-vector [L j1..j6 rad, L jaw CMD 0..45, R ...] in the LEADER frame and
maps it to the raw motor frame (Seeed joint_directions); jaw raw = cmd * -6 deg in [-270, 0]. MIT_MOCK=1: the plugin's mock
backend
and synthetic camera frames. Deploy stage: robot bridge.
"""

from __future__ import annotations
import base64
import collections
import contextlib
import math
import sys
import threading
import time

import cv2
import numpy as np

from . import config
from .arm_state import ArmState
from .hardware import armMode, armPorts, buildArm, kiJoints, openCameras, readArm
from .jaw import leaderVecToGoals, normToRawJaw
from .profile import profileStep, profileStepInterp

RECOVER_MIN_INTERVAL_S = 1.0  # at most one in-place recovery per second per arm (no busy loop)
RECONNECT_MIN_INTERVAL_S = 5.0  # a USB reconnect is started at most every 5 s per arm
PORT_WAIT_S = 30.0  # a re-enumerated USB-CAN adapter must come back within this
PORT_POLL_S = 1.0
PORT_SETTLE_S = 1.0  # after the port reappears, before the arm is rebuilt
REST_REACHED_DEG = 0.05  # every arm's goal - command below this: /rest has arrived
MAX_TICK_DECAY = 0.999  # slow decay of the published worst tick time
MS_PER_S = 1000.0
JPEG_QUALITY = 80
TICKS_BEFORE_DISCONNECT = 3.0  # the control thread sees `connected` False within this many ticks
CLEAR_COLLISION_SETTLE_S = 0.1
# 10-02 goal interpolation: segment duration = EMA of the goal inter-arrival time, clipped
GAP_EMA_KEEP = 0.6
GAP_EMA_NEW = 0.4
SEG_T_MIN_S = 0.02
SEG_T_MAX_S = 1.0
# PLAN playback limits
ROW_DUR_MIN_S = 0.005
ROW_DUR_MAX_S = 2.0
PACE_MIN = 0.05
PACE_MAX = 1.5
PACE_FLOOR = 1e-3  # division guard of the pace
SEGMENT_CHAIN_SLACK_S = 0.02  # a late tick chains segments without dropping more than this
# /set_speed and /jaw_close_test accepted ranges
MAX_DEG_S_RANGE = (1.0, 90.0)
MAX_ACC_RANGE = (5.0, 1000.0)
GRIP_DEG_S_RANGE = (10.0, 900.0)
INTERP_KP_RANGE = (0.5, 40.0)
CLOSE_DEG_RANGE = (-5.0, 45.0)
# lengths of exception / reason text kept in replies and events
REASON_CHARS = 200
SHORT_REASON_CHARS = 120
GUARD_EVENT_CHARS = 160
RECONNECT_REASON_CHARS = 150
CLOSE_ERROR_CHARS = 80
UNPARK_ERROR_CHARS = 160
COLLISION_ERROR_CHARS = 150


class BridgeState:
    """Service snapshot published by the control thread and read by the HTTP handlers."""

    def __init__(self) -> None:
        self.mConnected = False
        self.mEstop = False
        self.mStatus = "idle"
        self.mError = None
        self.mArms: dict[str, ArmState] = {}
        self.mCams = {}
        self.mTicks = 0
        self.mTickOverruns = 0
        self.mMaxTickMs = 0.0
        self.mRestPoseRaw = None


class PlanSegment:
    """One PLAN knot being played: from mStart to mEnd ({side: (raw deg (6,), jaw norm)}) over mTNom / pace seconds."""

    def __init__(self, row: int, t0: float, tNom: float, start: dict, end: dict) -> None:
        self.mRow = row
        self.mT0 = t0
        self.mTNom = tNom
        self.mStart = start
        self.mEnd = end


class PlanState:
    """10-06 user "추론에서_모터까지": numbered, timed rows (knots) handed over by the UI and played by the control loop. Rows are
    accepted only in order (session + row number); a knot that jumps more than MIT_PLAN_MAX_JUMP_DEG closes the session and
    holds; mPace scales time; when the queue runs dry the arm settles on the last knot. Without an open session nothing here
    runs and /execute_step behaves as before."""

    def __init__(self) -> None:
        self.mSession = None
        self.mKnots = collections.deque()
        self.mLastRow = 0
        self.mExecRow = 0
        self.mPace = 1.0
        self.mSeg: PlanSegment | None = None
        self.mDry = True
        self.mRowsIn = 0
        self.mGlitch = None
        self.mMaxJumpDeg = config.PLAN_MAX_JUMP_DEG


STATE = BridgeState()
PLAN = PlanState()
GLOCK = threading.Lock()  # goals / snapshot only; the bus belongs to the control thread
AUTO_EVENTS = []  # 10-01 user "e stop이나 이런거 없애줘": automatic trips, recovered in place
NEED_REENABLE = {}  # 10-21: side -> motor-fault reason, handled by the control loop
NO_GUARD = {}  # 10-04: last ignored collision reason per arm (log once per reason)


def _event(side: str, reason: str, result) -> None:
    AUTO_EVENTS.append({"t": time.time(), "arm": side, "reason": reason, "result": result})


def _onGuardTrip(side: str, reason) -> None:
    """Collision-guard latch replacement (MIT_COLLISION_GUARD!=1): record the would-be trip, never latch."""
    if config.MOTOR_REENABLE and "MOTOR FAULT" in str(reason) and "disabled" in str(reason):
        NEED_REENABLE[side] = str(reason)[:REASON_CHARS]  # 10-21: handled by the control loop (not from inside the guard)
    if NO_GUARD.get(side) != str(reason)[:SHORT_REASON_CHARS]:
        NO_GUARD[side] = str(reason)[:SHORT_REASON_CHARS]
        _event(side, "collision guard (latch disabled): " + str(reason)[:GUARD_EVENT_CHARS], "ignored")
        tShort = str(reason)[:SHORT_REASON_CHARS]
        print(f"[mit] {side}: collision guard would trip ({tShort}) -- latch disabled, ignored", flush=True)


def _buildArm(side: str, port: str) -> ArmState:
    return buildArm(side, port, _onGuardTrip, config.LIVE.mGripDegS)


def _reconnectArm(side: str, reason) -> None:
    """10-21 user "오른팔 usb도 끊기면 자동으로 리줌": the right arm's USB-CAN adapter re-enumerates (Broken pipe) when the arm starts
    moving. Drop the dead session, wait for the SAME USB location to come back (new /dev path), rebuild the arm with the frozen
    gains and hold the measured pose. Runs in its own thread; the control loop skips the arm meanwhile."""
    tArm = STATE.mArms[side]
    try:
        try:
            tArm.mRebot.disconnect()
        except Exception as tErr:
            print(f"[mit] RECONNECT {side}: old session close failed ({str(tErr)[:CLOSE_ERROR_CHARS]}) -- continuing", flush=True)
        try:  # a failed close keeps this arm's flock -> the new session would be refused
            tTransport = tArm.mRebot.arm_backend._transport
            tTransport._release_process_lock_locked()
            with contextlib.suppress(Exception):
                tTransport._release_controller_locked()
        except Exception:
            pass
        tT0 = time.time()
        tPort = None
        while time.time() - tT0 < PORT_WAIT_S and STATE.mConnected:
            try:
                tPort = armPorts().get(side)
                if tPort:
                    break
            except Exception:
                pass
            time.sleep(PORT_POLL_S)
        if not tPort:
            raise RuntimeError("port did not come back in 30 s")
        time.sleep(PORT_SETTLE_S)
        tNew = _buildArm(side, tPort)
        tQ, tJn = readArm(tNew)
        tNew.seedAt(tQ, tJn, time.time())
        with GLOCK:
            STATE.mArms[side] = tNew
        tResult = f"reconnected on {tPort}"
    except Exception as tErr:
        tArm.mReconnecting = False
        tResult = f"reconnect failed: {str(tErr)[:SHORT_REASON_CHARS]}"
    _event(side, "usb reconnect: " + str(reason)[:RECONNECT_REASON_CHARS], tResult)
    print(f"[mit] RECONNECT {side}: {tResult}", flush=True)


def recoverInPlace(side: str, reason) -> None:
    """No automatic E-STOP (user). The colleague's guard latches and then suppresses every frame, which under MIT lets the drives
    time out and drop the arm. Instead: clear the latch, re-seed the hold at the measured pose (fresh enable if a drive dropped
    out), keep streaming. Logged + counted in /status (auto_events)."""
    tArm = STATE.mArms[side]
    tFol = tArm.mFol
    tUsbGone = "Broken pipe" in str(reason) or "Device not configured" in str(reason)
    if config.USB_RECONNECT and tUsbGone and not tArm.mReconnecting:
        if time.time() - tArm.mLastReconnect > RECONNECT_MIN_INTERVAL_S:
            tArm.mReconnecting = True
            tArm.mLastReconnect = time.time()
            threading.Thread(target=_reconnectArm, args=(side, reason), daemon=True, name=f"mit-reconnect-{side}").start()
        return
    if tArm.mParked:  # 10-02 parked (torque off at rest): never auto re-enable
        return
    tNow = time.time()
    if tNow - tArm.mLastRecover < RECOVER_MIN_INTERVAL_S:
        return
    tArm.mLastRecover = tNow
    try:
        tFol.clear_collision_stop()
        # a drive that dropped out of Enable Mode is NOT seen by fol.arm_torque_enabled (a software flag that stays True), so a
        # MOTOR FAULT / disabled report forces a real re-enable (guard paused so the enable's bus hold cannot latch STALE)
        tReason = str(reason)
        if (not tFol.arm_torque_enabled) or ("FAULT" in tReason) or ("disabled" in tReason) or ("latched" in tReason):
            with tFol.guard_paused():
                tFol.set_arm_torque(True)
        tQ, tJn = readArm(tArm)
        with GLOCK:
            tArm.mQ = tQ
            tArm.mJawNorm = tJn
            tArm.mGoal = tQ.copy()
            tArm.mCmd = tQ.copy()
            tArm.mVel = None
        tArm.mBlocked = None
        tResult = "recovered"
    except Exception as tErr:
        tResult = f"recover failed: {str(tErr)[:SHORT_REASON_CHARS]}"
    _event(side, str(reason)[:REASON_CHARS], tResult)
    print(f"[mit] AUTO-RECOVER {side}: {reason} -> {tResult}", flush=True)


def holdAll(reason: str) -> None:
    """Operator E-STOP: both arms stop and hold the measured pose; the service latches E-STOP."""
    for tArm in STATE.mArms.values():
        try:
            tArm.mFol.stop_motion()
        except Exception as tErr:
            print("[mit] hold failed:", tErr, flush=True)
        if tArm.mQ is not None:
            tArm.mGoal = tArm.mQ.copy()
            tArm.mCmd = tArm.mQ.copy()
            tArm.mVel = None
    STATE.mEstop = True
    STATE.mStatus = "ESTOP"
    STATE.mError = reason
    print(f"[mit] BOTH ARMS HOLD: {reason}", flush=True)


def parkTorqueOff() -> None:
    for tSide, tArm in STATE.mArms.items():
        try:
            with tArm.mFol.guard_paused():
                tArm.mFol.set_arm_torque(False)
            tResult = "torque off"
        except Exception as tErr:
            tResult = f"torque off failed: {str(tErr)[:SHORT_REASON_CHARS]}"
        _event(tSide, "rest reached (park)", tResult)
        print(f"[mit] PARK {tSide}: {tResult}", flush=True)


def unpark(side: str) -> None:
    """Re-enable a parked arm in place: torque on (guard paused), hold re-seeded at the MEASURED pose, then unparked."""
    tArm = STATE.mArms[side]
    if not tArm.mParked:
        return
    tFol = tArm.mFol
    with tFol.guard_paused():
        tFol.set_arm_torque(True)
    tQ, tJn = readArm(tArm)
    with GLOCK:
        tArm.mQ = tQ
        tArm.mJawNorm = tJn
        tArm.mGoal = tQ.copy()
        tArm.mCmd = tQ.copy()
        tArm.mVel = None
        tArm.mSegEnd = None
        tArm.mParked = False
    _event(side, "unpark (motion command)", "torque on, hold at measured pose")
    print(f"[mit] UNPARK {side}: torque on, hold at measured pose", flush=True)


def _tickArm(side: str, arm: ArmState, dt: float) -> None:
    """One control tick of one arm: read, profile step, MIT frame, rest arrival, recovery of a blocked / faulted drive."""
    if arm.mReconnecting:  # 10-21: USB re-enumerated, _reconnectArm owns this arm until it is back
        return
    tQ, tJn = readArm(arm)
    with GLOCK:
        arm.mQ = tQ
        arm.mJawNorm = tJn
        arm.mTRead = time.time()
        if arm.mCmd is None:
            arm.mCmd = tQ.copy()
        if arm.mGoal is None:
            arm.mGoal = tQ.copy()
        if arm.mJawGoal is None:
            arm.mJawGoal = tJn
        if arm.mParked:  # 10-02 torque off at rest: read only, no frames, no recover
            return
        if STATE.mEstop:
            return
        tLive = config.LIVE
        tSegment = arm.mSegEnd is not None and arm.mSegEnd is arm.mGoal
        if (tLive.mGoalInterp or PLAN.mSession is not None) and STATE.mStatus != "resting" and tSegment:
            tVff = profileStepInterp(arm, tLive.mMaxDegS, tLive.mMaxAccDegS2, dt, tLive.mInterpKp, time.time())
        else:
            tVmax = config.REST_DEG_S if STATE.mStatus == "resting" else tLive.mMaxDegS
            tVff = profileStep(arm, tVmax, tLive.mMaxAccDegS2, dt)
        tAction = {f"joint_{tI}.pos": float(arm.mCmd[tI]) for tI in range(config.ARM_DOF)}
        tAction["gripper.pos"] = float(arm.mJawGoal)
    arm.mFol.send_joint_action(tAction, vel_ff_deg_s=[float(tV) for tV in tVff])
    if STATE.mStatus == "resting" and all(
        float(np.abs(tA.mGoal - tA.mCmd).max()) < REST_REACHED_DEG for tA in STATE.mArms.values() if tA.mCmd is not None
    ):
        STATE.mStatus = "connected"  # rest reached: back to the normal speed limit
        if config.LIVE.mRestTorqueOff:
            with GLOCK:
                for tOther in STATE.mArms.values():
                    tOther.mParked = True
            threading.Thread(target=parkTorqueOff, daemon=True).start()
    tBlocked = arm.mFol.streaming_blocked_reason()
    arm.mBlocked = tBlocked
    if tBlocked and not STATE.mEstop:
        recoverInPlace(side, tBlocked)
    tNeed = NEED_REENABLE.pop(side, None)
    if tNeed and not STATE.mEstop:
        recoverInPlace(side, "motor re-enable: " + tNeed)


def _controlLoop() -> None:
    tDt = 1.0 / config.TICK_HZ
    tNext = time.perf_counter()
    while STATE.mConnected:
        tT0 = time.perf_counter()
        try:
            if PLAN.mSession is not None and not STATE.mEstop:
                with GLOCK:
                    planUpdate(time.time())
            for tSide, tArm in list(STATE.mArms.items()):
                _tickArm(tSide, tArm, tDt)
        except Exception as tErr:
            if not STATE.mEstop:
                for tSide in STATE.mArms:
                    recoverInPlace(tSide, f"control tick raised: {tErr}")
        STATE.mTicks += 1
        tElapsedMs = (time.perf_counter() - tT0) * MS_PER_S
        STATE.mMaxTickMs = max(STATE.mMaxTickMs * MAX_TICK_DECAY, tElapsedMs)
        tNext += tDt
        tSleep = tNext - time.perf_counter()
        if tSleep > 0:
            time.sleep(tSleep)
        else:
            STATE.mTickOverruns += 1
            tNext = time.perf_counter()


def _startControlThread() -> None:
    threading.Thread(target=_controlLoop, daemon=True, name="mit-control").start()


# ---- cameras (never on the control thread) ----
def cameraFrame(name: str):
    """(rgb, age_ms): newest frame without waiting for a new one (the colleague's camera_frame.latest_frame peek)."""
    if config.MOCK:
        tImg = np.full((config.CAM_HEIGHT_PX, config.CAM_WIDTH_PX, 3), 60, np.uint8)
        cv2.putText(tImg, f"MOCK {name}", (40, 240), 0, 2, (255, 255, 255), 3)
        return tImg, 0.0
    from bh_indy7_lerobot.camera_frame import latest_frame

    tCam = STATE.mCams.get(name)
    if tCam is None:
        return None, None
    tImg = latest_frame(tCam)
    tTs = getattr(tCam, "latest_timestamp", None)
    return tImg, (None if tTs is None else round((time.perf_counter() - tTs) * MS_PER_S, 1))


def joints14Raw() -> list:
    """The robot_service 14-vector in raw degrees (jaws as physical degrees)."""
    tOut = []
    for tSide in config.ALL_SIDES:
        tArm = STATE.mArms.get(tSide)
        if tArm is None:  # 10-13: arm not connected (MIT_SIDES): zeros keep the layout
            tOut += [0.0] * config.JOINTS_PER_SIDE
            continue
        tOut += [float(tQj) for tQj in tArm.mQ] + [normToRawJaw(tArm.mJawNorm, tSide)]
    return tOut


def _jpeg(img) -> bytes:
    _, tBuf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return tBuf


# ---- PLAN playback ----
def _planSegT(seg: PlanSegment) -> float:
    return seg.mTNom / max(PLAN.mPace, PACE_FLOOR)


def planUpdate(now: float) -> None:
    """(control thread, under GLOCK) advance to the next knot when the current segment is done."""
    if PLAN.mSession is None:
        return
    tSeg = PLAN.mSeg
    if tSeg is not None and now < tSeg.mT0 + _planSegT(tSeg):
        return
    if not PLAN.mKnots:
        PLAN.mDry = True
        return
    tRow, tGoals, tDur = PLAN.mKnots.popleft()
    tT0 = now if tSeg is None else max(now - SEGMENT_CHAIN_SLACK_S, tSeg.mT0 + _planSegT(tSeg))  # chain without dropping time
    if tSeg is not None:
        tStart = {tSide: (tSeg.mEnd[tSide][0].copy(), tSeg.mEnd[tSide][1]) for tSide in STATE.mArms}
    else:
        tStart = {tSide: (tArm.mCmd.copy(), tArm.mJawGoal) for tSide, tArm in STATE.mArms.items()}
    tSeg = PlanSegment(tRow, tT0, float(tDur), tStart, tGoals)
    PLAN.mSeg = tSeg
    PLAN.mExecRow = tRow
    PLAN.mDry = False
    tT = _planSegT(tSeg)
    for tSide, tArm in STATE.mArms.items():
        tRaw, tJn = tGoals[tSide]
        tArm.mSegStart = np.asarray(tStart[tSide][0], float).copy()
        tArm.mSegEnd = np.asarray(tRaw, float).copy()
        tArm.mGoal = tArm.mSegEnd
        tArm.mSegT = tT
        tArm.mTSeg = tT0
        tArm.mJawGoal = tJn


def planClose(reason: str) -> None:
    with GLOCK:
        PLAN.mSession = None
        PLAN.mKnots.clear()
        PLAN.mSeg = None
        PLAN.mDry = True
    print(f"[mit] PLAN closed: {reason}", flush=True)


def holdGoals() -> None:
    with GLOCK:
        for tArm in STATE.mArms.values():
            tArm.mGoal = tArm.mQ.copy()
            tArm.mCmd = tArm.mQ.copy()
            tArm.mVel = None


def _setInterpolatedGoal(arm: ArmState, raw, now: float) -> np.ndarray:
    """10-02 user "브리지로 해줘": a goal within INTERP_MAX_GAP_S of the previous one becomes a segment from where the previous
    segment's reference is now, over the EMA of the goal inter-arrival time; a first goal after a pause is a plain step."""
    tGap = now - arm.mTGoal
    if tGap < config.INTERP_MAX_GAP_S and arm.mSegEnd is not None and arm.mSegEnd is arm.mGoal:
        tT0 = arm.mSegT
        tS0 = min(1.0, (now - arm.mTSeg) / tT0) if tT0 > 0 else 1.0
        tStart = arm.mSegStart + (arm.mSegEnd - arm.mSegStart) * tS0  # the reference where it is now
        tEma = tGap if arm.mGapEma is None else GAP_EMA_KEEP * arm.mGapEma + GAP_EMA_NEW * tGap
    elif tGap < config.INTERP_MAX_GAP_S:
        tStart = arm.mGoal.copy() if arm.mGoal is not None else raw
        tEma = tGap
    else:
        tStart = raw  # first goal after a pause: plain step to it
        tEma = None
    arm.mGapEma = tEma
    arm.mSegStart = np.asarray(tStart, float)
    arm.mSegT = float(np.clip(tEma, SEG_T_MIN_S, SEG_T_MAX_S)) if tEma else 0.0
    arm.mTSeg = now
    arm.mTGoal = now
    tRaw = np.asarray(raw, float)
    arm.mSegEnd = tRaw
    return tRaw


def createApp():
    """The Flask app with the robot_service HTTP contract (+ PLAN playback, park, jaw test)."""
    from flask import Flask, Response, jsonify, request

    tApp = Flask(__name__)

    def notConnected():
        return jsonify({"error": "not connected"}), 400

    @tApp.route("/connect", methods=["POST"])
    def connect():
        if STATE.mConnected:
            return jsonify({"ok": True, "msg": "already connected"})
        try:
            STATE.mStatus = "connecting"
            tPorts = armPorts()
            STATE.mArms = {}
            try:
                for tSide in config.SIDES:
                    STATE.mArms[tSide] = _buildArm(tSide, tPorts[tSide])
            except Exception:
                # 10-01: a partial connect must not leave an energised arm without a control loop: whatever connected is held by
                # streaming its present pose (E-STOP latched), so the operator can still /disconnect it.
                if STATE.mArms:
                    for tArm in STATE.mArms.values():
                        tQ, tJn = readArm(tArm)
                        tArm.seedAt(tQ, tJn, time.time())
                    STATE.mConnected = True
                    STATE.mEstop = True
                    STATE.mStatus = "ESTOP"
                    _startControlThread()
                raise
            STATE.mCams = openCameras()
            for tArm in STATE.mArms.values():
                tQ, tJn = readArm(tArm)
                tArm.seedAt(tQ, tJn, time.time())
            STATE.mRestPoseRaw = joints14Raw()
            STATE.mConnected = True
            STATE.mEstop = False
            STATE.mStatus = "connected"
            STATE.mError = None
            _startControlThread()
            return jsonify(
                {
                    "ok": True,
                    "mode": "mit/integral",
                    "ports": tPorts,
                    "tick_hz": config.TICK_HZ,
                    "max_deg_s": config.LIVE.mMaxDegS,
                }
            )
        except Exception as tErr:
            STATE.mError = str(tErr)
            if not STATE.mConnected:
                STATE.mStatus = "error"
            return jsonify({"ok": False, "error": str(tErr), "held_arms": list(STATE.mArms) if STATE.mConnected else []}), 500

    @tApp.route("/reopen_cameras", methods=["POST"])
    def reopen_cameras():
        """10-03 user: cameras re-plugged (used elsewhere): release the old OpenCV handles and open them again; arms untouched."""
        tOld = STATE.mCams or {}
        for tCam in tOld.values():
            try:
                tCam.disconnect()
            except Exception:
                pass
        try:
            STATE.mCams = openCameras()
            return jsonify({"ok": True, "cams": list(STATE.mCams)})
        except Exception as tErr:
            STATE.mCams = {}
            return jsonify({"ok": False, "error": str(tErr)}), 500

    @tApp.route("/status")
    def status():
        tArms = (
            {
                tS: {
                    "blocked": tA.mBlocked,
                    "mode": armMode(tA.mFol, tA.mCfg),
                    "goal_minus_cmd_deg": None if tA.mCmd is None else round(float(np.abs(tA.mGoal - tA.mCmd).max()), 3),
                    "read_age_ms": None if tA.mTRead is None else round((time.time() - tA.mTRead) * MS_PER_S, 1),
                }
                for tS, tA in STATE.mArms.items()
            }
            if STATE.mArms
            else {}
        )
        tLive = config.LIVE
        return jsonify(
            {
                "connected": STATE.mConnected,
                "estop": STATE.mEstop,
                "status": STATE.mStatus,
                "n_joints": config.N_JOINTS,
                "service": "mit",
                "error": STATE.mError,
                "ticks": STATE.mTicks,
                "tick_overruns": STATE.mTickOverruns,
                "max_tick_ms": round(STATE.mMaxTickMs, 2),
                "tick_hz": config.TICK_HZ,
                "max_deg_s": tLive.mMaxDegS,
                "max_acc_deg_s2": tLive.mMaxAccDegS2,
                "grip_deg_s": tLive.mGripDegS,
                "grip_open_frac": config.GRIP_OPEN_FRAC,
                "auto_events": len(AUTO_EVENTS),
                "last_auto_event": AUTO_EVENTS[-1] if AUTO_EVENTS else None,
                "arms": tArms,
                "goal_interp": tLive.mGoalInterp,
                "rest_torque_off": tLive.mRestTorqueOff,
                "parked": [tSd for tSd, tA in STATE.mArms.items() if tA.mParked],
                "kp": {tSd: list(tA.mCfg.arm_mit_kp) for tSd, tA in STATE.mArms.items() if tA.mCfg is not None},
                "ki": {tSd: tA.mCfg.arm_integral_ki_nm_per_rad_s for tSd, tA in STATE.mArms.items() if tA.mCfg is not None},
                "ki_joints": kiJoints(),
            }
        )

    @tApp.route("/observe")
    def observe():
        if not STATE.mConnected:
            return notConnected()
        with GLOCK:
            tJoints = joints14Raw()
        tImgs, tAges = {}, {}
        for tName, tMapped in ({} if config.NO_CAMERAS else config.CAM_MAP).items():
            tImg, tAge = cameraFrame(tName)
            if tImg is None:
                return jsonify({"error": f"camera {tName} has no frame"}), 500
            tImgs[tMapped] = base64.b64encode(_jpeg(tImg).tobytes()).decode()
            tAges[tMapped] = tAge
        tRad = list(tJoints)
        for tI in config.ARM_IDX:
            tRad[tI] = math.radians(tJoints[tI])
        return jsonify({"joints_deg": tJoints, "joints_rad": tRad, "images": tImgs, "cam_age_ms": tAges, "t_wall": time.time()})

    @tApp.route("/frame/<cam>")
    def frame(cam):
        tInverse = {tV: tK for tK, tV in config.CAM_MAP.items()}
        tImg, _ = cameraFrame(tInverse.get(cam, cam)) if STATE.mConnected else (None, None)
        if tImg is None:
            tImg = np.zeros((config.CAM_HEIGHT_PX, config.CAM_WIDTH_PX, 3), np.uint8)
        return Response(_jpeg(tImg).tobytes(), mimetype="image/jpeg")

    @tApp.route("/motor_states")
    def motor_states():
        if not STATE.mConnected:
            return notConnected()
        tOut = {}
        with GLOCK:
            for tSide in config.SIDES:
                tArm = STATE.mArms[tSide]
                tMotors = {tName: {"status": 1, "pos_deg": round(float(tArm.mQ[tI]), 2)} for tI, tName in enumerate(config.MOTORS)}
                tMotors["gripper"] = {"status": 1, "pos_deg": round(normToRawJaw(tArm.mJawNorm, tSide), 1)}
                tOut[f"{tSide}_arm"] = tMotors
        return jsonify(tOut)

    @tApp.route("/execute_step", methods=["POST"])
    def execute_step():
        if not STATE.mConnected:
            return notConnected()
        if STATE.mEstop:
            return jsonify({"error": f"E-STOP ({STATE.mError})"}), 400
        if PLAN.mSession is not None:
            return jsonify({"error": f"PLAN session {PLAN.mSession} open -- /plan/close first"}), 409
        tAction = (request.json or {}).get("action")
        if not tAction or len(tAction) != config.N_JOINTS:
            return jsonify({"error": "bad action"}), 400
        tGoals = leaderVecToGoals(tAction)
        for tSide in tGoals:  # 10-02: a motion command re-enables a parked arm in place first
            if STATE.mArms[tSide].mParked:
                try:
                    unpark(tSide)
                except Exception as tErr:
                    return jsonify({"error": f"unpark {tSide} failed: {str(tErr)[:UNPARK_ERROR_CHARS]}"}), 500
        with GLOCK:
            tNow = time.time()
            for tSide, (tRaw, tJn) in tGoals.items():
                tArm = STATE.mArms[tSide]
                if config.LIVE.mGoalInterp:
                    tRaw = _setInterpolatedGoal(tArm, tRaw, tNow)
                tArm.mGoal = tRaw
                tArm.mJawGoal = tJn
        tSent = list(tAction)
        for tI in config.ARM_IDX:
            tSent[tI] = math.degrees(tAction[tI])
        return jsonify({"ok": True, "sent_deg": tSent, "error": None})

    @tApp.route("/stop", methods=["POST"])
    def stop():
        if not STATE.mConnected:
            return notConnected()
        holdGoals()
        STATE.mStatus = "connected"
        return jsonify({"ok": True, "msg": "stop (hold the current pose)"})

    @tApp.route("/plan/open", methods=["POST"])
    def plan_open():
        if not STATE.mConnected:
            return notConnected()
        if STATE.mEstop:
            return jsonify({"error": f"E-STOP ({STATE.mError})"}), 400
        tSession = (request.json or {}).get("session")
        if tSession is None:
            return jsonify({"error": "session id required"}), 400
        for tSide in STATE.mArms:
            if STATE.mArms[tSide].mParked:
                unpark(tSide)
        with GLOCK:
            if STATE.mStatus == "resting":
                STATE.mStatus = "connected"
            for tArm in STATE.mArms.values():
                tArm.mGoal = tArm.mQ.copy()
                tArm.mCmd = tArm.mQ.copy()
                tArm.mVel = None
                tArm.mSegEnd = None
            PLAN.mSession = tSession
            PLAN.mLastRow = 0
            PLAN.mExecRow = 0
            PLAN.mPace = 1.0
            PLAN.mSeg = None
            PLAN.mDry = True
            PLAN.mRowsIn = 0
            PLAN.mGlitch = None
            PLAN.mKnots.clear()
        print(f"[mit] PLAN open: session {tSession}", flush=True)
        return jsonify({"ok": True, "session": tSession, "row0": "the measured pose"})

    @tApp.route("/plan/rows", methods=["POST"])
    def plan_rows():
        tBody = request.json or {}
        if PLAN.mSession is None or tBody.get("session") != PLAN.mSession:
            return jsonify({"error": f"no open session {tBody.get('session')} (open: {PLAN.mSession})"}), 409
        tAccepted = 0
        for tRowSpec in tBody.get("rows", []):
            tRow = int(tRowSpec["row"])
            tAction = tRowSpec["action"]
            tDur = float(tRowSpec["dur_s"])
            if tRow != PLAN.mLastRow + 1:
                return jsonify({"error": f"row {tRow} out of order (expected {PLAN.mLastRow + 1})", "accepted": tAccepted}), 409
            if not (ROW_DUR_MIN_S <= tDur <= ROW_DUR_MAX_S) or not tAction or len(tAction) != config.N_JOINTS:
                return jsonify({"error": f"bad row {tRow}", "accepted": tAccepted}), 400
            tGoals = leaderVecToGoals(tAction)
            with GLOCK:
                if PLAN.mKnots:
                    tPrev = PLAN.mKnots[-1][1]
                elif PLAN.mSeg is not None:
                    tPrev = PLAN.mSeg.mEnd
                else:
                    tPrev = {tSide: (tArm.mCmd, tArm.mJawGoal) for tSide, tArm in STATE.mArms.items()}
                tJump = max(float(np.abs(np.asarray(tGoals[tSide][0]) - np.asarray(tPrev[tSide][0])).max()) for tSide in tGoals)
                if tJump > PLAN.mMaxJumpDeg:
                    PLAN.mGlitch = f"row {tRow}: jump {tJump:.1f} deg > {PLAN.mMaxJumpDeg}"
                else:
                    PLAN.mKnots.append((tRow, tGoals, tDur))
                    PLAN.mLastRow = tRow
                    PLAN.mRowsIn += 1
                    tAccepted += 1
            if PLAN.mGlitch:
                tGlitch = PLAN.mGlitch
                planClose(tGlitch)
                holdGoals()
                return jsonify({"error": f"GLITCH -- session closed, holding: {tGlitch}", "accepted": tAccepted}), 409
        return jsonify({"ok": True, "accepted": tAccepted, "last_row": PLAN.mLastRow})

    @tApp.route("/plan/pace", methods=["POST"])
    def plan_pace():
        tBody = request.json or {}
        if PLAN.mSession is None or tBody.get("session") != PLAN.mSession:
            return jsonify({"error": "no such session"}), 409
        tNew = float(np.clip(float(tBody.get("pace", 1.0)), PACE_MIN, PACE_MAX))
        tNow = time.time()
        with GLOCK:
            tSeg = PLAN.mSeg
            if tSeg is not None:
                tTOld = _planSegT(tSeg)
                tS = min(1.0, (tNow - tSeg.mT0) / tTOld) if tTOld > 0 else 1.0
                PLAN.mPace = tNew
                tTNew = _planSegT(tSeg)
                tSeg.mT0 = tNow - tS * tTNew
                for tArm in STATE.mArms.values():
                    tArm.mSegT = tTNew
                    tArm.mTSeg = tSeg.mT0
            else:
                PLAN.mPace = tNew
        return jsonify({"ok": True, "pace": PLAN.mPace})

    @tApp.route("/plan/close", methods=["POST"])
    def plan_close():
        tSession = (request.json or {}).get("session")
        if PLAN.mSession is not None and (tSession is None or tSession == PLAN.mSession):
            planClose("operator / UI")
            holdGoals()
        return jsonify({"ok": True})

    @tApp.route("/plan/progress")
    def plan_progress():
        tNow = time.time()
        with GLOCK:
            tSeg = PLAN.mSeg
            tLeftSeg = 0.0
            tLeftSeg1x = 0.0
            if tSeg is not None:
                tT = _planSegT(tSeg)
                tLeftSeg = max(0.0, tSeg.mT0 + tT - tNow)
                tLeftSeg1x = tLeftSeg * PLAN.mPace
            tQueued1x = sum(tKnot[2] for tKnot in PLAN.mKnots)
            return jsonify(
                {
                    "session": PLAN.mSession,
                    "exec_row": PLAN.mExecRow,
                    "last_row": PLAN.mLastRow,
                    "queued": len(PLAN.mKnots),
                    "remaining_s": tLeftSeg + tQueued1x / max(PLAN.mPace, PACE_FLOOR),
                    "remaining_1x_s": tLeftSeg1x + tQueued1x,
                    "pace": PLAN.mPace,
                    "dry": PLAN.mDry,
                    "rows_in": PLAN.mRowsIn,
                    "glitch": PLAN.mGlitch,
                    "estop": STATE.mEstop,
                    "t": tNow,
                }
            )

    @tApp.route("/estop", methods=["POST"])
    def estop():
        if STATE.mConnected:
            if PLAN.mSession is not None:
                planClose("operator E-STOP")
            holdGoals()
            holdAll("operator E-STOP")
        return jsonify({"ok": True, "msg": "E-STOP - both arms hold"})

    @tApp.route("/clear_estop", methods=["POST"])
    def clear_estop():
        tBlocked = {tS: tA.mBlocked for tS, tA in STATE.mArms.items() if tA.mBlocked}
        if tBlocked:
            tError = f"collision guard still latched: {tBlocked} -- /reenable after checking the arm"
            return jsonify({"ok": False, "error": tError}), 400
        holdGoals()
        STATE.mEstop = False
        STATE.mStatus = "connected"
        STATE.mError = None
        return jsonify({"ok": True})

    @tApp.route("/reenable", methods=["POST"])
    def reenable():
        if not STATE.mConnected:
            return notConnected()
        tDone = {}
        for tSide, tArm in STATE.mArms.items():
            try:
                if tArm.mParked:
                    unpark(tSide)
                    tDone[tSide] = "unparked"
                    continue
                if tArm.mFol.arm_torque_enabled:
                    tDone[tSide] = "already enabled (no-op: a re-enable holds the bus > 0.5 s and stalls the control loop)"
                else:
                    with tArm.mFol.guard_paused():
                        tArm.mFol.set_arm_torque(True)
                    tDone[tSide] = True
            except Exception as tErr:
                tDone[tSide] = str(tErr)[:SHORT_REASON_CHARS]
        holdGoals()
        return jsonify({"ok": True, "reenabled": tDone})

    @tApp.route("/clear_collision", methods=["POST"])
    def clear_collision():
        """10-01: the colleague's guard latches and then SUPPRESSES every frame, so under MIT the drives time out and drop to
        disabled (the left arm went limp). Clear the latch on every arm, re-enable (the plugin's enable re-seeds the live pose),
        hold the present pose, release E-STOP."""
        if not STATE.mConnected:
            return notConnected()
        tDone = {}
        for tSide, tArm in STATE.mArms.items():
            try:
                tArm.mFol.clear_collision_stop()
                with tArm.mFol.guard_paused():
                    tArm.mFol.set_arm_torque(True)
                tArm.mBlocked = None
                tDone[tSide] = "cleared+enabled"
            except Exception as tErr:
                tDone[tSide] = f"ERR {str(tErr)[:COLLISION_ERROR_CHARS]}"
        time.sleep(CLEAR_COLLISION_SETTLE_S)
        holdGoals()
        if all(tV == "cleared+enabled" for tV in tDone.values()):
            STATE.mEstop = False
            STATE.mStatus = "connected"
            STATE.mError = None
        return jsonify({"ok": not STATE.mEstop, "arms": tDone})

    @tApp.route("/torque", methods=["POST"])
    def torque():
        """robot_service compat: {arm: left|right|both, on: bool}. Off = the arm goes limp (support it first)."""
        if not STATE.mConnected:
            return notConnected()
        tBody = request.get_json(force=True) or {}
        tWhich = tBody.get("arm", "both")
        tOn = bool(tBody.get("on", False))
        tSides = {"left": ["left"], "right": ["right"], "both": ["left", "right"]}.get(tWhich, [])
        tDone = {}
        for tSide in tSides:
            tFol = STATE.mArms[tSide].mFol
            try:
                if tOn and STATE.mArms[tSide].mParked:
                    unpark(tSide)
                    tDone[tSide] = "unparked: torque on, hold at measured pose"
                    continue
                if not tOn:
                    STATE.mArms[tSide].mParked = True  # 10-02: off = parked, so auto-recover does not re-enable it
                if tOn and tFol.arm_torque_enabled:
                    # 10-01: the UI calls /torque on=True at every run start only to power the JAWS. Re-running the arm enable
                    # here held the bus > 0.5 s and the guard latched ARM STATE STALE. Arm already on -> jaw only.
                    tFol.set_gripper_torque(True)
                    tDone[tSide] = "arm already on; jaw on"
                else:
                    with tFol.guard_paused():
                        tFol.set_arm_torque(tOn)
                    tDone[tSide] = tOn
            except Exception as tErr:
                tDone[tSide] = f"ERR {str(tErr)[:SHORT_REASON_CHARS]}"
        holdGoals()
        return jsonify({"ok": True, "torque_on": tOn, "arms": tDone})

    @tApp.route("/rest", methods=["POST"])
    def rest():
        if not STATE.mConnected:
            return notConnected()
        if STATE.mEstop:
            return jsonify({"error": "E-STOP"}), 400
        tRestPose = STATE.mRestPoseRaw
        for tSide in config.SIDES:
            if STATE.mArms[tSide].mParked:
                unpark(tSide)
        with GLOCK:
            for tSide in config.SIDES:
                tR = config.ALL_SIDES.index(tSide) * config.JOINTS_PER_SIDE
                STATE.mArms[tSide].mGoal = np.asarray(tRestPose[tR : tR + config.ARM_DOF], float)
        STATE.mStatus = "resting"
        return jsonify({"ok": True, "msg": f"rest: ramping to the connect pose at {config.REST_DEG_S} deg/s"})

    @tApp.route("/set_speed", methods=["POST"])
    def set_speed():
        """10-01 user "움직임이 너무 빠른데": per-joint ramp limit (deg/s) and friends, changed live: {"max_deg_s": 30}."""
        tLive = config.LIVE
        tBody = request.json or {}
        if "goal_interp" in tBody:
            tLive.mGoalInterp = bool(tBody["goal_interp"])
            print(f"[mit] goal interpolation -> {tLive.mGoalInterp}", flush=True)
        if "rest_torque_off" in tBody:
            tLive.mRestTorqueOff = bool(tBody["rest_torque_off"])
            print(f"[mit] rest torque off -> {tLive.mRestTorqueOff}", flush=True)
        if "interp_kp" in tBody:
            tLive.mInterpKp = float(np.clip(float(tBody["interp_kp"]), *INTERP_KP_RANGE))
        tV = float(tBody.get("max_deg_s", tLive.mMaxDegS))
        tAcc = float(tBody.get("max_acc_deg_s2", tLive.mMaxAccDegS2))
        tGrip = float(tBody.get("grip_deg_s", tLive.mGripDegS))
        tOk = (
            MAX_DEG_S_RANGE[0] <= tV <= MAX_DEG_S_RANGE[1]
            and MAX_ACC_RANGE[0] <= tAcc <= MAX_ACC_RANGE[1]
            and GRIP_DEG_S_RANGE[0] <= tGrip <= GRIP_DEG_S_RANGE[1]
        )
        if not tOk:
            tError = "max_deg_s in [1, 90], max_acc_deg_s2 in [5, 1000], grip_deg_s in [10, 900]"
            return jsonify({"ok": False, "error": tError}), 400
        tLive.mMaxDegS = tV
        tLive.mMaxAccDegS2 = tAcc
        tLive.mGripDegS = tGrip
        for tArm in STATE.mArms.values():  # the transport reads cfg.gripper_velocity_deg_s on every FORCE_POS frame
            tArm.mCfg.gripper_velocity_deg_s = tLive.mGripDegS
        print(f"[mit] profile -> {tLive.mMaxDegS} deg/s, {tLive.mMaxAccDegS2} deg/s^2, jaw {tLive.mGripDegS} deg/s", flush=True)
        return jsonify(
            {
                "ok": True,
                "max_deg_s": tLive.mMaxDegS,
                "max_acc_deg_s2": tLive.mMaxAccDegS2,
                "grip_deg_s": tLive.mGripDegS,
                "goal_interp": tLive.mGoalInterp,
                "interp_kp": tLive.mInterpKp,
                "rest_torque_off": tLive.mRestTorqueOff,
            }
        )

    @tApp.route("/gripper_zero", methods=["POST"])
    def gripper_zero():
        """release = jaw motor torque off (the UI uses it at rest so the jaw does not buzz against its stop); set (re-zero) is
        not implemented here -- use robot_service.py (POS_VEL)."""
        tBody = request.get_json(force=True) or {}
        tSide, tStage = tBody.get("arm"), tBody.get("stage")
        if tStage == "release" and tSide in STATE.mArms:
            try:
                STATE.mArms[tSide].mFol.set_gripper_torque(False)
                return jsonify({"ok": True, "msg": f"{tSide} jaw torque off"})
            except Exception as tErr:
                return jsonify({"ok": False, "error": str(tErr)[:REASON_CHARS]}), 500
        tError = "gripper re-zero (stage=set) is not implemented in the MIT bridge -- use robot_service.py (POS_VEL)"
        return jsonify({"ok": False, "error": tError}), 501

    @tApp.route("/jaw_close_test", methods=["POST"])
    def jaw_close_test():
        """11-02 user "3도씩 더 꽉 닫아볼래?": {"arm": "left", "close_deg": 19}: set that jaw's close endpoint live and command it
        fully closed, the arm held at its current goal (a parked arm is re-enabled in place). Not persisted -- the launch env
        MIT_GRIP_CLOSE_DEG_<SIDE> decides the value after a restart."""
        if not STATE.mConnected:
            return notConnected()
        tBody = request.get_json(force=True) or {}
        tSide, tDeg = tBody.get("arm"), float(tBody.get("close_deg", 0.0))
        if tSide not in STATE.mArms or not CLOSE_DEG_RANGE[0] <= tDeg <= CLOSE_DEG_RANGE[1]:
            return jsonify({"error": "arm must be left/right, close_deg in [-5, 45]"}), 400
        tArm = STATE.mArms[tSide]
        if tArm.mParked:
            unpark(tSide)
        tArm.mRebot.arm_backend.set_gripper_endpoint("close", math.radians(tDeg))
        config.GRIP_CLOSE_EXTRA[tSide] = tDeg
        with GLOCK:
            tArm.mJawGoal = 1.0
        print(f"[mit] {tSide} jaw close test -> {tDeg:+.1f} deg, closing", flush=True)
        return jsonify({"ok": True, "arm": tSide, "close_deg": tDeg})

    @tApp.route("/disconnect", methods=["POST"])
    def disconnect():
        STATE.mConnected = False
        time.sleep(TICKS_BEFORE_DISCONNECT / config.TICK_HZ)
        for tArm in STATE.mArms.values():
            try:
                tArm.mFol.stop_motion()
                tArm.mFol.disconnect()
            except Exception as tErr:
                print("[mit] disconnect:", tErr, flush=True)
        for tCam in STATE.mCams.values():
            try:
                tCam.disconnect()
            except Exception:
                pass
        STATE.mArms = {}
        STATE.mCams = {}
        STATE.mStatus = "disconnected"
        return jsonify({"ok": True, "msg": "disconnected (torque off)"})

    return tApp


def main(argv: list[str] | None = None) -> int:
    tBanner = (
        f"[mit] robot_service_mit on :{config.PORT} mock={config.MOCK} tick={config.TICK_HZ} Hz max={config.LIVE.mMaxDegS} deg/s"
    )
    print(tBanner, flush=True)
    createApp().run(host="0.0.0.0", port=config.PORT, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
