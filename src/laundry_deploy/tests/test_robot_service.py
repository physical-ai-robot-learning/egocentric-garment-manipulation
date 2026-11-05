"""MIT robot bridge (Flask service): jaw command / norm mapping, leader-vector goals, the acceleration-limited and segment
profiles, gain / Ki overrides from the environment, and the HTTP routes against fake arms (no hardware, no plugin)."""

from __future__ import annotations
import importlib
import os
import sys
import types

import numpy as np
import pytest

from laundry_deploy.tests import fakes
from laundry_deploy.tests.golden_record import checkGolden

PACKAGE = "laundry_deploy.robot_service"


def _freshImport():
    """Re-import every module of the bridge package, so the environment switches are read again and the state is empty."""
    for tName in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
        sys.modules.pop(tName, None)
    return types.SimpleNamespace(
        **{
            tShort: importlib.import_module(f"{PACKAGE}.{tMod}")
            for tShort, tMod in (
                ("B", "mit_bridge"),
                ("C", "config"),
                ("J", "jaw"),
                ("P", "profile"),
                ("H", "hardware"),
                ("A", "arm_state"),
            )
        }
    )


def _dropImports():
    for tName in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
        sys.modules.pop(tName, None)


@pytest.fixture
def M(monkeypatch):
    for tKey in list(os.environ):
        if tKey.startswith(("MIT_", "CAM_", "LEFT_LOC", "RIGHT_LOC")):
            monkeypatch.delenv(tKey)
    yield _freshImport()
    _dropImports()


def test_jaw_mapping(M, monkeypatch):
    J, C = M.J, M.C
    tRaw = [-300.0, -270.0, -200.0, -45.0, -30.0, -20.0, -5.0, 0.0, 10.0]
    tOut = dict(
        plain=[J.rawJawToNorm(r) for r in tRaw],
        plain_side=[J.rawJawToNorm(r, "left") for r in tRaw],
        inv=[J.normToRawJaw(n) for n in (0.0, 0.25, 0.5, 0.9, 1.0, 1.2)],
    )
    monkeypatch.setitem(C.GRIP_CLOSE_EXTRA, "left", 19.0)
    tOut["squeeze"] = [J.rawJawToNorm(r, "left") for r in tRaw]
    tOut["squeeze_inv"] = [J.normToRawJaw(n, "left") for n in (0.0, 0.5, 0.9, 0.95, 1.0)]
    tOut["roundtrip"] = [J.normToRawJaw(J.rawJawToNorm(r, "left"), "left") for r in tRaw]
    tAct = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 45.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 7.5]
    tOut["goals"] = J.leaderVecToGoals(tAct)
    monkeypatch.setitem(C.GRIP_OPEN_FRAC, "left", 0.5)
    tOut["goals_half"] = J.leaderVecToGoals([3.0, -3.5, -4.0, 2.0, 2.0, 3.0, 60.0] + [0.0] * 6 + [-2.0])
    tOut["consts"] = dict(
        SEEED_DIR=C.SEEED_DIR,
        SEEED_LIM=C.SEEED_LIM_DEG,
        GRIP=[C.GRIP_OPEN_DEG, C.GRIP_CLOSE_DEG, C.GRIP_CMD_SCALE],
        ARM_IDX=C.ARM_IDX,
        SIDES=C.SIDES,
        MOTORS=C.MOTORS,
        CAM_MAP=C.CAM_MAP,
        CAMERAS=C.CAMERAS,
        env=[
            C.MOCK,
            C.PORT,
            C.TICK_HZ,
            C.LIVE.mMaxDegS,
            C.REST_DEG_S,
            C.LIVE.mMaxAccDegS2,
            C.LIVE.mGoalInterp,
            C.LIVE.mInterpKp,
            C.INTERP_MAX_GAP_S,
            C.LIVE.mRestTorqueOff,
            C.LIVE.mGripDegS,
            C.NO_CAMERAS,
            C.MOTOR_REENABLE,
            C.USB_RECONNECT,
            C.GRIP_SQUEEZE_RAMP_DEG,
            M.B.PLAN.mMaxJumpDeg,
        ],
    )
    checkGolden("robot_service/jaw", tOut)


def test_env_switches(monkeypatch):
    tEnv = {
        "MIT_MAX_DEG_S": "30",
        "MIT_GOAL_INTERP": "1",
        "MIT_SIDES": "left",
        "MIT_NO_CAMERAS": "1",
        "MIT_GRIP_OPEN_FRAC_LEFT": "0.5",
        "MIT_GRIP_CLOSE_DEG_LEFT": "25",
        "MIT_PLAN_MAX_JUMP_DEG": "45",
        "MIT_TICK_HZ": "200",
        "CAM_LEFT": "4",
    }
    for tKey in list(os.environ):
        if tKey.startswith(("MIT_", "CAM_")):
            monkeypatch.delenv(tKey)
    for tKey, tValue in tEnv.items():
        monkeypatch.setenv(tKey, tValue)
    tM = _freshImport()
    try:
        checkGolden(
            "robot_service/env",
            dict(
                sides=tM.C.SIDES,
                max_deg_s=tM.C.LIVE.mMaxDegS,
                interp=tM.C.LIVE.mGoalInterp,
                no_cams=tM.C.NO_CAMERAS,
                open_frac=tM.C.GRIP_OPEN_FRAC,
                close=tM.C.GRIP_CLOSE_EXTRA,
                jump=tM.B.PLAN.mMaxJumpDeg,
                tick=tM.C.TICK_HZ,
                cams=tM.C.CAMERAS,
                goals=tM.J.leaderVecToGoals([0.1] * 6 + [45.0] + [0.2] * 6 + [45.0]),
            ),
        )
    finally:
        _dropImports()


def test_gain_overrides(M, monkeypatch):
    H = M.H
    tOut = dict(none=[H.gainOverrides(), H.kiJoints()])
    monkeypatch.setenv("MIT_KP", "64,64,64,28,28,28")
    monkeypatch.setenv("MIT_KD", "3,3,3,2,2,2")
    monkeypatch.setenv("MIT_KI", "40")
    monkeypatch.setenv("MIT_KI_JOINTS", "28,28,40,10,10,10")
    tOut["set"] = [H.gainOverrides(), H.kiJoints()]
    for tEnv, tValue in (
        ("MIT_KP", "1,2,3"),
        ("MIT_KD", "1,1,1,1,1,9"),
        ("MIT_KI", "0"),
        ("MIT_KI", "201"),
        ("MIT_KI_JOINTS", "1,1,1,1,1,0"),
    ):
        monkeypatch.setenv("MIT_KP", "64,64,64,28,28,28")
        monkeypatch.setenv("MIT_KD", "3,3,3,2,2,2")
        monkeypatch.setenv("MIT_KI", "40")
        monkeypatch.setenv("MIT_KI_JOINTS", "28,28,40,10,10,10")
        monkeypatch.setenv(tEnv, tValue)
        try:
            tOut[f"{tEnv}={tValue}"] = [H.gainOverrides(), H.kiJoints()]
        except RuntimeError as e:
            tOut[f"{tEnv}={tValue}"] = str(e)
    checkGolden("robot_service/gains", tOut)


def _arm(M, **fields):
    tArm = M.A.ArmState(None, None, None)
    for tKey, tValue in fields.items():
        setattr(tArm, tKey, tValue)
    return tArm


def test_profiles(M, monkeypatch):
    tOut = {}
    for tName, tGoal, tVel in (
        ("pos", np.array([10.0, -5.0, 3.0, 0.0, 1.0, -2.0]), None),
        ("neg", np.array([-20.0, 0.0, 0.0, 0.0, 0.0, 0.0]), None),
        ("moving", np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0]), np.array([-30.0, 10.0, 0.0, 0.0, 0.0, 5.0])),
    ):
        tArm = _arm(M, mGoal=tGoal, mCmd=np.zeros(6), mVel=tVel)
        tTrace = []
        for _ in range(150):
            tV = M.P.profileStep(tArm, 45.0, 180.0, 0.01)
            tTrace.append([tArm.mCmd.copy(), np.asarray(tV).copy()])
        tOut[tName] = tTrace[::10] + [tTrace[-1]]
    tClock = fakes.FakeClock(500.0)
    tArm = _arm(
        M, mCmd=np.zeros(6), mVel=None, mSegStart=np.zeros(6), mSegEnd=np.array([4.0, -2.0, 1.0, 0, 0, 0]), mSegT=0.1, mTSeg=500.0
    )
    tArm.mGoal = tArm.mSegEnd
    tTrace = []
    for _ in range(60):
        tV = M.P.profileStepInterp(tArm, 45.0, 1000.0, 0.01, M.C.LIVE.mInterpKp, tClock.time())
        tTrace.append([tArm.mCmd.copy(), np.asarray(tV).copy()])
        tClock.advance(0.01)
    tOut["interp"] = tTrace[::5] + [tTrace[-1]]
    tArm0 = _arm(M, mCmd=np.ones(6), mVel=None, mSegStart=np.ones(6), mSegEnd=np.ones(6), mSegT=0.0, mTSeg=0.0)
    tOut["interp_zero"] = [M.P.profileStepInterp(tArm0, 45.0, 1000.0, 0.01, M.C.LIVE.mInterpKp, tClock.time()), tArm0.mCmd]
    checkGolden("robot_service/profiles", tOut)


class _Fol:
    def __init__(self, log, side):
        self.mLog = log
        self.mSide = side
        self.arm_torque_enabled = True
        self.arm_control_mode = "mit"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*a, **k):
            self.mLog.append([self.mSide, name, list(a), {kk: vv for kk, vv in k.items()}])
            if name == "streaming_blocked_reason":
                return None
            return None

        return call

    def guard_paused(self):
        tLog, tSide = self.mLog, self.mSide

        class Ctx:
            def __enter__(self):
                tLog.append([tSide, "guard_paused"])

            def __exit__(self, *a):
                return False

        return Ctx()

    def refresh_arm_state(self):
        self.mLog.append([self.mSide, "refresh"])

    def read_observation(self):
        tQ = {"left": [1.0, -2.0, 3.0, -4.0, 5.0, -6.0], "right": [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]}[self.mSide]
        tObs = {f"joint_{i}.pos": v for i, v in enumerate(tQ)}
        tObs["gripper.pos"] = 0.4
        return tObs


def _connect(M, monkeypatch):
    tLog = []
    B = M.B
    monkeypatch.setattr(B, "time", fakes.FakeClock(700.0))
    for tSide in M.C.SIDES:
        tFol = _Fol(tLog, tSide)
        tRebot = types.SimpleNamespace(
            arm_backend=types.SimpleNamespace(set_gripper_endpoint=lambda *a: tLog.append(["endpoint", list(a)]))
        )
        tCfg = types.SimpleNamespace(
            arm_mit_kp=(64,) * 6, arm_integral_ki_nm_per_rad_s=28.0, gripper_velocity_deg_s=90.0, arm_control_mode="mit"
        )
        tArm = M.A.ArmState(tRebot, tFol, tCfg)
        tQ, tJn = M.H.readArm(tArm)
        tArm.seedAt(tQ, tJn, 699.9)
        B.STATE.mArms[tSide] = tArm
    B.STATE.mConnected = True
    B.STATE.mStatus = "connected"
    B.STATE.mRestPoseRaw = B.joints14Raw()
    return tLog


def _j(r):
    return [r.status_code, r.get_json(silent=True)]


def test_routes_not_connected(M):
    tApp = M.B.createApp()
    tC = tApp.test_client()
    tOut = {}
    for tMethod, tPath, tBody in (
        ("get", "/status", None),
        ("get", "/observe", None),
        ("get", "/motor_states", None),
        ("post", "/execute_step", {"action": [0] * 14}),
        ("post", "/stop", None),
        ("post", "/plan/open", {"session": 1}),
        ("post", "/plan/rows", {"session": 1, "rows": []}),
        ("post", "/plan/pace", {"session": 1}),
        ("post", "/plan/close", {}),
        ("get", "/plan/progress", None),
        ("post", "/estop", None),
        ("post", "/clear_estop", None),
        ("post", "/reenable", None),
        ("post", "/clear_collision", None),
        ("post", "/torque", {"arm": "left", "on": True}),
        ("post", "/rest", None),
        ("post", "/gripper_zero", {"arm": "left", "stage": "set"}),
        ("post", "/jaw_close_test", {"arm": "left", "close_deg": 10}),
        ("post", "/reopen_cameras", None),
        ("post", "/set_speed", {"max_deg_s": 30}),
        ("post", "/set_speed", {"max_deg_s": 120}),
        (
            "post",
            "/set_speed",
            {"goal_interp": True, "interp_kp": 100, "rest_torque_off": True, "grip_deg_s": 900, "max_acc_deg_s2": 1000},
        ),
    ):
        tR = getattr(tC, tMethod)(tPath, json=tBody) if tBody is not None else getattr(tC, tMethod)(tPath)
        tJ = tR.get_json(silent=True)
        if isinstance(tJ, dict) and "t" in tJ:
            tJ["t"] = "<t>"
        tOut[f"{tMethod} {tPath} {tBody}"] = [tR.status_code, tJ]
    tOut["url_map"] = sorted((r.rule, sorted(r.methods)) for r in tApp.url_map.iter_rules())
    checkGolden("robot_service/routes_idle", tOut)


def _seg(seg):
    return dict(T_nom=seg.mTNom, end=seg.mEnd, row=seg.mRow, t0=seg.mT0)


def test_routes_connected(M, monkeypatch):
    B = M.B
    tLog = _connect(M, monkeypatch)
    monkeypatch.setattr(M.C, "NO_CAMERAS", True)
    tC = B.createApp().test_client()
    tOut = dict(status=_j(tC.get("/status")), motors=_j(tC.get("/motor_states")), raw14=B.joints14Raw())
    tObs = tC.get("/observe").get_json()
    tObs["t_wall"] = "<t>"
    tOut["observe"] = tObs
    tAct = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 20.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 7.5]
    tOut["exec"] = _j(tC.post("/execute_step", json={"action": tAct}))
    tOut["exec_goals"] = {s: [a.mGoal, a.mJawGoal] for s, a in B.STATE.mArms.items()}
    tOut["exec_bad"] = _j(tC.post("/execute_step", json={"action": [0] * 13}))
    tOut["stop"] = _j(tC.post("/stop"))
    tOut["after_stop"] = {s: [a.mGoal, a.mCmd, a.mVel] for s, a in B.STATE.mArms.items()}
    # PLAN playback
    tOut["plan_open_none"] = _j(tC.post("/plan/open", json={}))
    tOut["plan_open"] = _j(tC.post("/plan/open", json={"session": "s1"}))
    tOut["exec_during_plan"] = _j(tC.post("/execute_step", json={"action": tAct}))
    tRows = [
        {"row": i + 1, "action": [0.01 * (i + 1)] * 6 + [10.0] + [-0.01 * (i + 1)] * 6 + [30.0], "dur_s": 0.05} for i in range(4)
    ]
    tOut["rows"] = _j(tC.post("/plan/rows", json={"session": "s1", "rows": tRows}))
    tOut["rows_wrong_session"] = _j(tC.post("/plan/rows", json={"session": "s2", "rows": tRows}))
    tOut["rows_order"] = _j(tC.post("/plan/rows", json={"session": "s1", "rows": [dict(tRows[0], row=9)]}))
    tOut["rows_bad"] = _j(tC.post("/plan/rows", json={"session": "s1", "rows": [dict(tRows[0], row=5, dur_s=3.0)]}))
    tOut["pace"] = _j(tC.post("/plan/pace", json={"session": "s1", "pace": 3.0}))
    tOut["progress"] = _j(tC.get("/plan/progress"))
    B.planUpdate(700.0)
    tOut["seg1"] = _seg(B.PLAN.mSeg)
    tOut["arm_after_seg"] = {s: [a.mSegStart, a.mSegEnd, a.mSegT, a.mTSeg, a.mJawGoal] for s, a in B.STATE.mArms.items()}
    tOut["pace2"] = _j(tC.post("/plan/pace", json={"session": "s1", "pace": 0.5}))
    tOut["progress2"] = _j(tC.get("/plan/progress"))
    for tT in (700.01, 700.2, 700.5, 701.0, 702.0):
        B.planUpdate(tT)
        tOut[f"plan_{tT}"] = [B.PLAN.mExecRow, B.PLAN.mDry, len(B.PLAN.mKnots)]
    tGlitch = [{"row": 5, "action": [3.0] * 6 + [10.0] + [0.0] * 6 + [30.0], "dur_s": 0.05}]
    tOut["glitch"] = _j(tC.post("/plan/rows", json={"session": "s1", "rows": tGlitch}))
    tOut["after_glitch"] = [B.PLAN.mSession, B.PLAN.mGlitch]
    tOut["plan_open2"] = _j(tC.post("/plan/open", json={"session": "s3"}))
    tOut["plan_close_other"] = _j(tC.post("/plan/close", json={"session": "zz"}))
    tOut["plan_after_close_other"] = B.PLAN.mSession
    tOut["plan_close"] = _j(tC.post("/plan/close", json={"session": "s3"}))
    # rest / torque / reenable / estop / clear
    tOut["rest"] = _j(tC.post("/rest"))
    tOut["rest_goals"] = {s: a.mGoal for s, a in B.STATE.mArms.items()}
    tOut["status_rest"] = B.STATE.mStatus
    tOut["torque_off"] = _j(tC.post("/torque", json={"arm": "left", "on": False}))
    tOut["parked"] = {s: a.mParked for s, a in B.STATE.mArms.items()}
    tOut["torque_on"] = _j(tC.post("/torque", json={"arm": "both", "on": True}))
    tOut["torque_bogus"] = _j(tC.post("/torque", json={"arm": "x", "on": True}))
    B.STATE.mArms["right"].mFol.arm_torque_enabled = False
    tOut["reenable"] = _j(tC.post("/reenable"))
    tOut["estop"] = _j(tC.post("/estop"))
    tOut["exec_estop"] = _j(tC.post("/execute_step", json={"action": tAct}))
    tOut["rest_estop"] = _j(tC.post("/rest"))
    B.STATE.mArms["left"].mBlocked = "ARM STATE STALE"
    tOut["clear_blocked"] = _j(tC.post("/clear_estop"))
    tOut["clear_collision"] = _j(tC.post("/clear_collision"))
    tOut["clear"] = _j(tC.post("/clear_estop"))
    tOut["gripper_release"] = _j(tC.post("/gripper_zero", json={"arm": "left", "stage": "release"}))
    tOut["jaw_test_bad"] = _j(tC.post("/jaw_close_test", json={"arm": "left", "close_deg": 50}))
    tOut["jaw_test"] = _j(tC.post("/jaw_close_test", json={"arm": "left", "close_deg": 19}))
    tOut["after_jaw_test"] = [M.C.GRIP_CLOSE_EXTRA, B.STATE.mArms["left"].mJawGoal]
    tOut["speed"] = _j(tC.post("/set_speed", json={"max_deg_s": 60, "grip_deg_s": 400}))
    tOut["cfg_grip"] = [a.mCfg.gripper_velocity_deg_s for a in B.STATE.mArms.values()]
    tOut["status_end"] = _j(tC.get("/status"))
    tOut["fol_log"] = tLog
    tOut["auto_events"] = [{k: v for k, v in e.items() if k != "t"} for e in B.AUTO_EVENTS]
    checkGolden("robot_service/routes_connected", tOut)


def test_recover_and_hold(M, monkeypatch):
    B = M.B
    tLog = _connect(M, monkeypatch)
    tOut = {}
    B.recoverInPlace("left", "MOTOR FAULT joint_5 disabled")
    B.recoverInPlace("left", "again within a second")
    B.time.advance(1.5)
    B.STATE.mArms["right"].mParked = True
    B.recoverInPlace("right", "parked arm")
    B.STATE.mArms["right"].mParked = False
    B.recoverInPlace("right", "ARM STATE STALE")
    tOut["after_recover"] = {s: [a.mGoal, a.mCmd, a.mVel, a.mLastRecover] for s, a in B.STATE.mArms.items()}
    B.holdAll("test hold")
    tOut["hold"] = [B.STATE.mEstop, B.STATE.mStatus, B.STATE.mError]
    B.unpark("left")
    B.STATE.mArms["left"].mParked = True
    B.unpark("left")
    tOut["unpark"] = B.STATE.mArms["left"].mParked
    B.parkTorqueOff()
    tOut["log"] = tLog
    tOut["auto_events"] = [{k: v for k, v in e.items() if k != "t"} for e in B.AUTO_EVENTS]
    checkGolden("robot_service/recover", tOut)
