"""Web UI run control: settings persistence, the per-run control object, how UI settings become run arguments (the GO
relaxations), stop / E-STOP / clear / park handling against a fake bridge, the READY move checks and ramp, and the camera views."""

from __future__ import annotations
import argparse
import json
import threading
import types

import cv2
import numpy as np

from laundry_deploy.core import height_calib as heightCalib
from laundry_deploy.core.errors import DeployError
from laundry_deploy.session.cli import buildParser
from laundry_deploy.ui import app as control_mod
from laundry_deploy.ui import ready_move as ready_mod
from laundry_deploy.ui import run_control as run_control_mod
from laundry_deploy.ui import settings as settings_mod
from laundry_deploy.ui.run_control import UiRunControl
from laundry_deploy.tests import fakes
from laundry_deploy.tests.golden_record import checkGolden, sha256

SIDES = ("left", "right")
Q0 = {"left": np.array([0.1, -0.3, 0.4, 0.2, -0.1, 0.05]), "right": np.array([-0.1, -0.25, 0.35, -0.2, 0.1, -0.05])}


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "SETTINGS_FILE", tmp_path / "s.json")
    tOut = dict(nofile=settings_mod.loadSettings())
    (tmp_path / "s.json").write_text('{"speed": 0.9, "policy": "dit", "extra": 1}')
    tOut["file"] = settings_mod.loadSettings()
    settings_mod.saveSettings({"a": 1, "한글": "값"})
    tOut["saved"] = (tmp_path / "s.json").read_text()
    (tmp_path / "s.json").write_text("{bad")
    tOut["broken"] = settings_mod.loadSettings()
    tDir = tmp_path / "ck"
    (tDir / "0100000" / "pretrained_model").mkdir(parents=True)
    (tDir / "0050000" / "pretrained_model").mkdir(parents=True)
    (tDir / "notes").mkdir()
    monkeypatch.setattr(settings_mod, "CKPT_DIRS", {"act": [tDir, tmp_path / "missing"]})
    tOut["ckpts"] = [p.replace(str(tmp_path), "<TMP>") for p in settings_mod.listCkpts("act")]
    tOut["HB_TIMEOUT_S"] = settings_mod.HB_TIMEOUT_S
    tOut["default_keys"] = sorted(settings_mod.DEFAULTS)
    checkGolden("ui/settings", tOut)


def test_uictl(monkeypatch):
    tClock = fakes.FakeClock(50.0)
    monkeypatch.setattr(run_control_mod, "time", tClock)
    tC = UiRunControl(True)
    tOut = [tC.takeStep(), tC.poll(), tC.mStepMode, tC.mStepsDone]
    tC.mStepN = 2
    tOut += [tC.takeStep(), tC.takeStep(), tC.takeStep(), tC.mStepsDone, tC.mStepN]
    tClock.advance(2.9)
    tOut.append(tC.poll())
    tClock.advance(0.2)
    tOut.append(tC.poll())
    tOut.append(tC.poll())
    tDry = UiRunControl(False)
    tClock.advance(100.0)
    tOut.append(tDry.poll())
    for tI in range(405):
        tDry.publish({"t": tI * 0.1, "grip": np.array([[0.01 * (tI % 50), 1.0 - 0.01 * (tI % 30)]])})
    tOut += [len(tDry.mK1), tDry.mK1[0], tDry.mK1[-1], tDry.mSnap["t"]]
    checkGolden("ui/uictl", tOut)


def _app(dryOnly=False, settings=None, bridgeStatus=None):
    tA = object.__new__(control_mod.UiApp)
    tA.mArgs = argparse.Namespace(
        dry_only=dryOnly, replay_global="", global_url="http://127.0.0.1:8092", bridge_url=None, port=8770
    )
    tA.mSettings = dict(settings_mod.DEFAULTS, **(settings or {}))
    tA.mRunArgs = buildParser().parse_args(["run"])
    tArms, _ = fakes.fakeArms()
    tA.mRig = types.SimpleNamespace(
        mUrl="http://bridge",
        mReplay=False,
        mWaveOk=True,
        mArms=tArms,
        mQReady=Q0,
        mCollision=fakes.FakeCollision(),
        mCamsCfg={s: {"location": f"0x{s}"} for s in SIDES},
        mWristCams={s: types.SimpleNamespace(mRaw=None, mFrames=3) for s in SIDES},
        mGlobalCam=types.SimpleNamespace(mFrames=5, mRing=types.SimpleNamespace(latest=lambda: (0, None))),
        mPoller=types.SimpleNamespace(mRing=types.SimpleNamespace(latest=lambda: (0, (Q0, None, False)))),
    )
    tA.mLock = threading.Lock()
    tA.mCtl = None
    tA.mThread = None
    tA.mMode = "idle"
    tA.mMsg = "준비됨"
    tA.mLast = None
    tA.mBusy = None
    tA.mBridgeStatus = {}
    tA.mMotors = {}
    tA.mXvla = None
    tA.mBridgeRo = fakes.FakeBridge(Q0, status=bridgeStatus or {})
    tA.mBridgeRo.mReplies["/status"] = bridgeStatus or {}
    tA.mUmiRef = {}
    tA.mJaw = {}
    return tA


class _NoThread:
    mStarted = []

    def __init__(self, target=None, args=(), daemon=None, **k):
        self.mTarget = target
        self.mArgs = args

    def start(self):
        _NoThread.mStarted.append(self)

    def is_alive(self):
        return True


def test_start_run_argument_mapping(monkeypatch):
    monkeypatch.setattr(control_mod.threading, "Thread", _NoThread)
    monkeypatch.setattr(control_mod, "time", fakes.FakeClock())
    tOut = {}
    tCases = [
        ("dry", False, False, {}),
        ("go", True, False, {}),
        ("gostep", True, True, dict(policy="dit", rtc="nolook", tens=True)),
        (
            "go_xvla",
            True,
            False,
            dict(
                policy="xvla",
                grip_mode="sigmoid",
                grip_gain=4.0,
                z_offset_mm=5.0,
                amp=0.8,
                height_calib=True,
                active_arms="right",
                grip_fake_open=True,
                rtc="guided",
            ),
        ),
        ("go_binary", True, False, dict(grip_mode="binary", grip_cont=False)),
        ("dry_legacy", False, False, dict(grip_mode=None)),
        ("go_estop", True, False, {}),
    ]
    for tName, tGo, tStep, tSettings in tCases:
        tApp = _app(settings=tSettings, bridgeStatus={"estop": True} if tName == "go_estop" else {})
        if tSettings.get("grip_mode", "x") is None:
            del tApp.mSettings["grip_mode"]
        _NoThread.mStarted.clear()
        tRes = tApp.startRun(tGo, step=tStep)
        tRa = _NoThread.mStarted[-1].mArgs[0] if _NoThread.mStarted else None
        tOut[tName] = dict(
            res=tRes,
            mode=tApp.mMode,
            msg=tApp.mMsg,
            ctl_go=tApp.mCtl.mGo if tApp.mCtl else None,
            ctl_step=tApp.mCtl.mStepMode if tApp.mCtl else None,
            ra=vars(tRa) if tRa is not None else None,
            bridge=tApp.mBridgeRo.mLog,
        )
    # refusals
    tBusy = _app()
    tBusy.mBusy = "READY로 이동"
    tOut["busy"] = tBusy.startRun(False)
    tOut["dry_only_go"] = _app(dryOnly=True).startRun(True)
    tReplay = _app()
    tReplay.mRig.mReplay = True
    tOut["replay_go"] = tReplay.startRun(True)
    tNoWave = _app()
    tNoWave.mRig.mWaveOk = False
    tOut["nowave_go"] = tNoWave.startRun(True)
    tRunning = _app()
    tRunning.mThread = _NoThread()
    tOut["running"] = tRunning.startRun(False)
    checkGolden("ui/start_run", tOut)


def test_run_thread_outcomes(monkeypatch):
    tOut = {}
    for tName, tResult in (
        ("ok", {"stop_reason": "max time 60s"}),
        ("none", None),
        ("exit", SystemExit("로봇 상태 E-STOP")),
        ("err", ValueError("x")),
    ):
        tApp = _app()

        def fake(ra, rig, ctl, _r=tResult):
            if isinstance(_r, BaseException):
                raise _r
            return _r

        monkeypatch.setattr(control_mod, "runSession", fake)
        for tGo in (False, True):
            tRa = argparse.Namespace(go=tGo)
            tApp._run(tRa, None)
            tOut[f"{tName}_{tGo}"] = dict(msg=tApp.mMsg, mode=tApp.mMode, last=tApp.mLast)
    checkGolden("ui/run_thread", tOut)


def test_stop_buttons(monkeypatch):
    tOut = {}
    for tDry in (False, True):
        for tMode in ("idle", "go", "hold", "dry"):
            tApp = _app(dryOnly=tDry)
            tApp.mMode = tMode
            tApp.mCtl = UiRunControl(tMode == "go")
            tH = tApp.hold()
            tOut[f"hold_{tDry}_{tMode}"] = dict(res=tH, mode=tApp.mMode, stop=tApp.mCtl.mStopReq, log=tApp.mBridgeRo.mLog)
            tApp2 = _app(dryOnly=tDry)
            tApp2.mMode = tMode
            tApp2.mCtl = UiRunControl(True)
            tApp2.mCtl.mStopReq = "earlier"
            tE = tApp2.estop()
            tOut[f"estop_{tDry}_{tMode}"] = dict(res=tE, mode=tApp2.mMode, stop=tApp2.mCtl.mStopReq, log=tApp2.mBridgeRo.mLog)
    tStep = _app()
    tOut["step_idle"] = tStep.step()
    tStep.mMode = "go"
    tStep.mCtl = UiRunControl(True)
    tStep.mCtl.mStepMode = True
    tStep.mCtl.mStepsDone = 3
    tOut["step_go"] = [tStep.step(), tStep.mCtl.mStepN, tStep.step(), tStep.mCtl.mStepN]
    # clear E-STOP: ok, refused + clear_collision ok, refused twice, exception, busy
    for tName, tReplies in (
        ("ok", {"/clear_estop": {"ok": True}}),
        ("collision_ok", {"/clear_estop": {"ok": False, "error": "latched"}, "/clear_collision": {"ok": True, "arms": {}}}),
        (
            "collision_fail",
            {
                "/clear_estop": {"ok": False, "error": "latched"},
                "/clear_collision": {"ok": False, "error": "torque", "arms": {"left": "ERR x"}},
            },
        ),
        ("error_field", {"/clear_estop": {"error": "x"}, "/clear_collision": {"ok": True}}),
    ):
        tApp = _app()
        tApp.mMode = "hold"
        tApp.mBridgeRo.mReplies.update(tReplies)
        tOut[f"clear_{tName}"] = dict(res=tApp.clearEstop(), mode=tApp.mMode, log=tApp.mBridgeRo.mLog)
    tBusy = _app()
    tBusy.mMode = "go"
    tOut["clear_busy"] = tBusy.clearEstop()
    tRaise = _app()

    def boom(*a, **k):
        raise ConnectionError("down")

    tRaise.mBridgeRo.s.post = boom
    tOut["clear_raise"] = tRaise.clearEstop()
    checkGolden("ui/stop_buttons", tOut)


def test_motions(monkeypatch):
    import laundry_shared.visual_teleop.robot.bridge as vb

    tBridges = []

    def mkBridge(url, live=False, timeout_s=0.5):
        tB = fakes.FakeBridge(Q0, status={"estop": True})
        tB.mReplies["/rest"] = {"ok": True}
        tB.mInit = [url, live, timeout_s]
        tBridges.append(tB)
        return tB

    monkeypatch.setattr(vb, "HttpBridge", mkBridge)
    monkeypatch.setattr(
        control_mod.threading, "Thread", lambda target=None, daemon=None, **k: types.SimpleNamespace(start=target)
    )
    tStarts = []
    monkeypatch.setattr(control_mod, "startCmd", lambda a: tStarts.append(vars(a)))
    tReady = []
    monkeypatch.setattr(control_mod, "moveReady", lambda br, rig: tReady.append(br.mInit))
    tOut = {}
    tApp = _app(settings=dict(height_calib=True, start_root="flatS05C_train149"))
    tOut["ready"] = [tApp.goReady(), tApp.mMsg, tApp.mBusy, tApp.mMode]
    tOut["start"] = [tApp.goStart(), tApp.mMsg]
    tOut["park"] = [tApp.park(), tApp.mMsg]
    tOut["start_args"] = tStarts
    tOut["ready_bridges"] = tReady
    tOut["bridges"] = [[b.mInit, b.mLog] for b in tBridges]
    tApp2 = _app()
    tOut["start_default"] = tApp2.goStart()
    tOut["start_args2"] = tStarts[-1]
    tRefuse = _app()
    tRefuse.mBusy = "x"
    tOut["refuse_busy"] = tRefuse.park()
    tOut["refuse_dry"] = _app(dryOnly=True).park()
    tPark = _app()
    tBridges.clear()

    def mkRefuse(url, live=False, timeout_s=0.5):
        tB = fakes.FakeBridge(Q0)
        tB.mReplies["/rest"] = {"ok": False, "error": "estop"}
        tBridges.append(tB)
        return tB

    monkeypatch.setattr(vb, "HttpBridge", mkRefuse)
    tOut["park_refused"] = [tPark.park(), tPark.mMsg, tBridges[-1].mLog]
    checkGolden("ui/motions", tOut)


def test_move_ready(monkeypatch):
    import laundry_shared.visual_teleop.tools.robot_moves as rm

    tUnparks = []
    monkeypatch.setattr(rm, "unpark", lambda br, s: tUnparks.append(s))
    monkeypatch.setattr(ready_mod, "time", fakes.FakeClock())
    tArms, _ = fakes.fakeArms()
    tOut = {}
    tQr = {s: Q0[s] * 0.5 for s in SIDES}
    tCases = (
        ("ok", Q0, tQr, {}, {}),
        ("parked", Q0, tQr, {"parked": ["left"], "estop": True}, {"/torque": {"ok": True, "arms": {"left": True}}}),
        (
            "torque_err",
            Q0,
            tQr,
            {"parked": ["right"]},
            {"/torque": {"ok": True, "arms": {"right": "ERR no reply from motor 0x05 ..."}}},
        ),
        ("low_tcp", {s: Q0[s] * 4 for s in SIDES}, tQr, {}, {}),
        ("limits", Q0, {s: Q0[s] + 2.0 for s in SIDES}, {}, {}),
        ("same", tQr, tQr, {}, {}),
    )
    for tName, tQ, tQrCase, tStatus, tReplies in tCases:
        tBr = fakes.FakeBridge(tQ, g={"left": 1.0, "right": 0.0}, status=tStatus)
        tBr.mReplies.update(tReplies)
        tRig = types.SimpleNamespace(mArms=tArms, mQReady=tQrCase, mCollision=fakes.FakeCollision())
        try:
            ready_mod.moveReady(tBr, tRig)
            tRes = "ok"
        except DeployError as e:
            tRes = str(e)
        tOut[tName] = dict(res=tRes, log=tBr.mLog, n_sends=len(tBr.mSends), first=tBr.mSends[:2], last=tBr.mSends[-1:])
    tBrRej = fakes.FakeBridge(Q0, reject_at=3)
    try:
        ready_mod.moveReady(tBrRej, types.SimpleNamespace(mArms=tArms, mQReady=tQr, mCollision=fakes.FakeCollision()))
    except DeployError as e:
        tOut["reject"] = [str(e), tBrRej.mLog[-2:], len(tBrRej.mSends)]
    tBrE = fakes.FakeBridge(Q0, estop_at=30)
    try:
        ready_mod.moveReady(tBrE, types.SimpleNamespace(mArms=tArms, mQReady=tQr, mCollision=fakes.FakeCollision()))
    except DeployError as e:
        tOut["estop_mid"] = [str(e), len(tBrE.mSends)]
    tOut["unparks"] = tUnparks
    checkGolden("ui/move_ready", tOut)


def test_views(monkeypatch, tmp_path):
    tApp = _app(settings=dict(show_umi_ref=True))
    tRng = np.random.default_rng(16)
    tApp.mRig.mWristCams["left"].mRaw = (tRng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    tApp.mUmiRef = {"left": (tRng.random((720, 1280, 3)) * 255).astype(np.uint8)}
    tApp.mJaw = {"left": dict(area=12.5, top=400, axis=3.0), "umi": {"left": dict(area=10.0, top=380, axis=-2.0)}}
    tOut = dict(wrist_none=tApp.wristJpg("right"))
    tJ = tApp.wristJpg("left")
    tOut["wrist"] = sha256(cv2.imdecode(np.frombuffer(tJ, np.uint8), cv2.IMREAD_COLOR).tobytes())
    tApp.mSettings["show_umi_ref"] = False
    tOut["wrist_plain"] = sha256(cv2.imdecode(np.frombuffer(tApp.wristJpg("left"), np.uint8), cv2.IMREAD_COLOR).tobytes())
    tOut["tcp"] = tApp.tcp()
    tApp.mRig.mPoller = types.SimpleNamespace(mRing=types.SimpleNamespace(latest=lambda: (0, (None, None, True))))
    tOut["tcp_none"] = tApp.tcp()
    # global view: FK ring + the running plan projected through the camera model and the height calibration
    monkeypatch.setattr(heightCalib, "CALIB_FILE", tmp_path / "hc.json")
    (tmp_path / "hc.json").write_text(
        json.dumps({"fit": {"left": {"plane_dz": {"a": 0.01, "b": 0.0, "c": 0.002}, "xy_cam_minus_robot": [0.003, -0.002]}}})
    )
    monkeypatch.setattr(
        control_mod,
        "robotCam",
        lambda: (
            np.array([[0, -1, 0, 0.0], [0, 0, -1, 0.4], [1, 0, 0, 0.3], [0, 0, 0, 1.0]]),
            {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0},
        ),
    )
    tG = (tRng.random((480, 848, 3)) * 255).astype(np.uint8)
    tJpg = cv2.imencode(".jpg", tG, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    tApp.mRig.mPoller = types.SimpleNamespace(mRing=types.SimpleNamespace(latest=lambda: (0, (Q0, None, False))))
    tApp.mRig.mGlobalCam = types.SimpleNamespace(mRing=types.SimpleNamespace(latest=lambda: (0, (None, tJpg))))
    tOut["global_idle"] = sha256(cv2.imdecode(np.frombuffer(tApp.globalJpg(), np.uint8), cv2.IMREAD_COLOR).tobytes())
    tApp.mMode = "dry"
    tApp.mCtl = UiRunControl(False)
    tApp.mCtl.mSnap = {"Tc": {s: [fakes.fakeFk(s, Q0[s] + 0.01 * k) for k in range(32)] for s in SIDES}}
    tOut["global_plan"] = sha256(cv2.imdecode(np.frombuffer(tApp.globalJpg(), np.uint8), cv2.IMREAD_COLOR).tobytes())
    tApp.mRig.mGlobalCam = types.SimpleNamespace(mRing=types.SimpleNamespace(latest=lambda: (0, None)))
    tOut["global_none"] = tApp.globalJpg()
    checkGolden("ui/views", tOut)


def test_dep_parser_is_the_deploy_cli():
    tP = buildParser()
    checkGolden("ui/dep_parser", dict(run=vars(tP.parse_args(["run"])), prog_is_set=bool(tP.prog)))
