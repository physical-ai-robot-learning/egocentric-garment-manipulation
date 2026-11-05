"""Deploy session side: run-argument checks, the dry receding-horizon loop on a fake rig (fake clock, fake arms, fixed synthetic
chunks), the streamer timeline, ring buffer, preview, poller, gripper-limit guard, READY ramp, start pose, keys, reports and
the camera-mapping helpers. Printed messages are part of the record."""

from __future__ import annotations
import argparse
import json
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from laundry_shared.contracts.action_contract import T_to_pose9

from laundry_deploy.core import height_calib as heightCalib
from laundry_deploy.core.contract import GRIP_INDEX, POSE_SLICE
from laundry_deploy.core.errors import DeployError
from laundry_deploy.session import bridge as bridge_mod
from laundry_deploy.session import cameras as cameras_mod
from laundry_deploy.session import keys as keys_mod
from laundry_deploy.session import preview as preview_mod
from laundry_deploy.session import report as report_mod
from laundry_deploy.session import rig as rig_mod
from laundry_deploy.session import ring as ring_mod
from laundry_deploy.session import run_loop as session_mod
from laundry_deploy.session import start_pose as start_mod
from laundry_deploy.session import streamer as streamer_mod
from laundry_deploy.session.cli import buildParser
from laundry_deploy.tests import fakes
from laundry_deploy.tests.golden_record import checkGolden, sha256

SIDES = ("left", "right")
Q0 = {"left": np.array([0.1, -0.3, 0.4, 0.2, -0.1, 0.05]), "right": np.array([-0.1, -0.25, 0.35, -0.2, 0.1, -0.05])}
LATENCY_S = 0.12


def _parse(argv):
    return buildParser().parse_args(argv)


# ------------------------------------------------------------------------------------------------ run arguments / helpers
def test_check_run_args(capsys):
    tOut = {}
    tCases = [
        {},
        dict(policy="dit"),
        dict(policy="dit", speed=2.0, commit=16),
        dict(policy="dit", speed=3.0, commit=20),
        dict(policy="dit", speed=0.5, commit=20),
        dict(policy="label"),
        dict(policy="label", label="x.npz"),
        dict(commit=40),
        dict(commit=0),
        dict(go=True, assume_ready=True),
        dict(go=True, assume_start=True),
        dict(go=True, replay_global="d"),
        dict(go=True),
        dict(policy="xvla", commit=None),
    ]
    for tI, tKw in enumerate(tCases):
        tA = argparse.Namespace(
            **{
                **dict(
                    policy="act",
                    commit=None,
                    label="",
                    speed=0.5,
                    go=False,
                    assume_ready=False,
                    assume_start=False,
                    replay_global="",
                ),
                **tKw,
            }
        )
        try:
            session_mod.checkRunArgs(tA)
            tOut[str(tI)] = ["ok", tA.commit]
        except DeployError as e:
            tOut[str(tI)] = ["exit", str(e)]
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("session/check_run_args", tOut)


def _pose(rng, scale=0.2):
    tT = np.eye(4)
    tT[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.3).as_matrix()
    tT[:3, 3] = rng.normal(size=3) * scale
    return tT


def test_ensemble_and_scale():
    tRng = np.random.default_rng(10)
    tEns = [(0.05 * i, {s: [_pose(tRng) for _ in range(32)] for s in SIDES}) for i in range(4)]
    tD = _pose(tRng)
    checkGolden(
        "session/ensemble_scale",
        dict(
            ens=[session_mod.ensemble(tEns, t, s, m) for t in (0.2, 0.4, 1.0) for s in SIDES for m in (0.01, 0.5)],
            scale=[session_mod.scaleTransform(tD, s) for s in (0.0, 0.5, 1.0, 1.7)],
        ),
    )


def test_ring():
    tRing = ring_mod.Ring(4)
    tOut = [tRing.latest(), tRing.near(5, 1.0)]
    for tI in range(6):
        tRing.put(int(tI * 1e8), tI)
    tOut += [
        tRing.latest(),
        tRing.near(int(2.6e8), 0.05),
        tRing.near(int(2.6e8), 0.01),
        tRing.near(int(9e8), 0.5),
        len(tRing.mBuf),
    ]
    checkGolden("session/ring", tOut)


def test_streamer(monkeypatch):
    tClock = fakes.FakeClock(100.0)
    monkeypatch.setattr(streamer_mod, "time", tClock)
    tBr = fakes.FakeBridge(Q0)
    tSt = streamer_mod.Streamer(tBr, {s: np.zeros(6) for s in SIDES}, {"left": 0.0, "right": 1.0}, 30.0, dt=0.05)
    tRows = [{s: np.full(6, 0.01 * (j + 1)) * (1 if s == "left" else -1) for s in SIDES} for j in range(10)]
    tOut = dict(dt=tSt.mDt, tick0=tSt.tick(), executed0=tSt.executed(), last_idx0=tSt.lastIndex)
    tSt.setTimeline(tRows[:4], 0.05, {"left": 1.0, "right": 1.0})
    tClock.advance(0.12)
    tOut["anchor"] = tSt.anchor(tClock.monotonic())
    tOut["merge1"] = list(tSt.merge(tSt.executed(), tRows, {"left": 0.0, "right": 0.0}))
    tOut["tl1"] = {str(k): v for k, v in sorted(tSt.mTimeline.items())}
    tClock.advance(0.07)
    tOut["q1"] = tSt._currentTarget()
    tOut["merge2"] = list(tSt.merge(2, tRows[3:], {"left": 0.5, "right": 0.5}))
    tOut["tl2"] = {str(k): v for k, v in sorted(tSt.mTimeline.items())}
    tClock.advance(2.0)
    tOut["q_dry"] = tSt._currentTarget()
    tOut["n_dry"] = tSt.mDryTicks
    tOut["q_at"] = [tSt.qAt(x) for x in (-3.0, 0.0, 1.5, 2.25, 100.0)]
    tSt.mLast = {s: np.full(6, 0.3) for s in SIDES}
    tSt.setTimeline([], tSt.mDt, tSt.mG)
    tOut["tl_soft"] = {str(k): v for k, v in sorted(tSt.mTimeline.items())}
    tOut["g"] = tSt.mG
    # run(): sends at stream rate until the bridge rejects
    tBr2 = fakes.FakeBridge(Q0, reject_at=5)
    tSt2 = streamer_mod.Streamer(tBr2, {s: np.zeros(6) for s in SIDES}, {"left": 0.0, "right": 1.0}, 30.0)
    tSt2.setTimeline(tRows[:3], 0.05, {"left": 1.0, "right": 0.0})
    tSt2.run()
    tOut["run_sends"] = tBr2.mSends
    tOut["run_error"] = tSt2.mError
    tOut["run_n"] = tSt2.mSent
    tOut["run_clock"] = tClock.monotonic()
    checkGolden("session/streamer", tOut)


def test_poller_grip_limits_and_ready(monkeypatch, capsys):
    tOut = {}
    tPoll = bridge_mod.RobotPoller(None, Q0, True)
    tOut["fake_read"] = tPoll.read()
    tBr = fakes.FakeBridge(Q0, g={"left": 0.3, "right": 0.9})
    tPoll2 = bridge_mod.RobotPoller(tBr, Q0, False, hz=50.0)
    tOut["live_read"] = tPoll2.read()
    tOut["lat"] = tPoll2.mLat
    tOut["hz"] = [tPoll.mHz, tPoll2.mHz, len(tPoll.mRing.mBuf), tPoll.mRing.mBuf.maxlen]
    # run(): one good read, one failing read, one good read clears the error
    tCalls = {"n": 0}

    def flaky():
        tCalls["n"] += 1
        if tCalls["n"] == 2:
            raise RuntimeError("/observe: not connected")
        if tCalls["n"] >= 3:
            tPoll2.mStop = True
        return ({s: Q0[s] * tCalls["n"] for s in SIDES}, {s: 0.0 for s in SIDES}, False)

    tErrors = []
    tClock = fakes.FakeClock()

    def sleep(s):
        tErrors.append(tPoll2.mError)
        tClock.sleep(s)

    monkeypatch.setattr(
        bridge_mod, "time", types.SimpleNamespace(monotonic_ns=tClock.monotonic_ns, sleep=sleep, monotonic=tClock.monotonic)
    )
    tPoll2.read = flaky
    tPoll2.run()
    tOut["run_errors"] = tErrors
    tOut["run_ring"] = list(tPoll2.mRing.mBuf)
    tOut["run_clock"] = tClock.monotonic()
    # gripper-opening guard
    import requests

    for tName, tReply in (
        ("ok", {"grip_open_frac": {"left": 0.5, "right": 1.0}}),
        ("lower", {"grip_open_frac": {"left": 0.3, "right": 0.2}}),
        ("left_open", {"grip_open_frac": {"left": 1.0, "right": 1.0}}),
        ("missing", {"grip_open_frac": {"left": 0.5}}),
        ("none", {}),
        ("raise", None),
    ):

        def get(url, timeout=None, _r=tReply):
            tOut.setdefault("grip_urls", []).append([url, timeout])
            if _r is None:
                raise requests.ConnectionError("down")
            return types.SimpleNamespace(json=lambda: _r)

        monkeypatch.setattr(requests, "get", get)
        try:
            bridge_mod.checkGripLimits("http://b:8021/")
            tOut[f"grip_{tName}"] = "ok"
        except DeployError as e:
            tOut[f"grip_{tName}"] = str(e)
    tOut["GRIP_OPEN_FRAC_MAX"] = bridge_mod.GRIP_OPEN_FRAC_MAX
    # READY ramp: refused on a close pair, refused on a failed clear_estop, ramped otherwise
    import laundry_shared.visual_teleop.tools.umi_replay_bi as ub

    tRamps = []
    monkeypatch.setattr(ub, "ramp_both", lambda br, q, qr, g, deg_s=10.0: tRamps.append((q, qr, g, deg_s)))
    tArms, _ = fakes.fakeArms()
    tQr = {s: Q0[s] * 0.5 for s in SIDES}
    for tName, tQ, tReply in (
        ("ok", Q0, {"ok": True}),
        ("refused", Q0, {"ok": False, "error": "latched"}),
        ("close", {"left": np.zeros(6), "right": np.zeros(6)}, {"ok": True}),
    ):
        tBrR = fakes.FakeBridge(tQ, g={"left": 1.0, "right": 0.0})
        tBrR.mReplies["/clear_estop"] = tReply
        bridge_mod.goReady(tBrR, tArms, tQr)
        tOut[f"ready_{tName}"] = tBrR.mLog
    tOut["ramps"] = tRamps
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("session/poller_grip_ready", tOut)


def test_preview_run(monkeypatch):
    tArms, _ = fakes.fakeArms()
    tWs = {s: (np.array([0.0, -0.6, 0.0]), np.array([0.6, 0.6, 0.5])) for s in SIDES}
    tRng = np.random.default_rng(11)
    tTc = {s: np.stack([fakes.fakeFk(s, Q0[s] + tRng.normal(size=6) * 0.1 * (k + 1) / 8) for k in range(32)]) for s in SIDES}
    tOut = {}
    for tName, tKw in (
        ("default", {}),
        ("hold", dict(preview_hold=True, arm_clear=0.3, max_ik_mm=0.01)),
        ("stride8", dict(preview_stride=8)),
    ):
        tA = _parse(["run"])
        for k, v in tKw.items():
            setattr(tA, k, v)
        tClock = fakes.FakeClock()
        tPrev = preview_mod.Preview(tArms, fakes.FakeCollision(), tWs, tA, 0.003)

        def sleep(s, _p=tPrev, _c=tClock):
            _c.sleep(s)
            if _p.mJob is None:
                _p.mStop = True

        monkeypatch.setattr(preview_mod, "time", types.SimpleNamespace(monotonic=tClock.monotonic, sleep=sleep))
        tPrev.submit(1.25, tTc, Q0)
        tPrev.run()
        tOut[tName] = dict(res=tPrev.mResult, log=tPrev.mLog, clock=tClock.monotonic())
    checkGolden("session/preview", tOut)


def test_keys_ctl():
    tKeys = object.__new__(keys_mod.Keys)
    tKeys.mQueue = list(" xQ")
    tKeys.mFd = None
    tCtl = object.__new__(keys_mod.KeysCtl)
    tCtl.mKeys = tKeys
    tOut = [tCtl.poll(), tCtl.poll()]
    tKeys.mQueue = list("abq")
    tOut.append(tCtl.poll())
    tKeys.mQueue = list("zY")
    tOut.append(tKeys.wait("yn", timeout=1.0))
    tOut.append(tKeys.wait("yn", timeout=0.0))
    tOut.append(tCtl.publish({}))
    checkGolden("session/keys", tOut)


# ------------------------------------------------------------------------------------------------ dry receding-horizon loop
class _Ring:
    def __init__(self, clock, value, fixedT=None, history=None):
        self.mClock = clock
        self.mValue = value
        self.mFixedT = fixedT
        self.mHistory = history

    def latest(self):
        return (self.mFixedT if self.mFixedT is not None else self.mClock.monotonic_ns()), self.mValue

    def near(self, t, tol):
        if self.mHistory is None:
            return None
        return (t + int(0.004e9), self.mHistory)


class _Runner:
    def __init__(self, clock, kind="act", fakeOpen=True):
        self.mKind = kind
        self.mClock = clock
        self.mCalls = []
        self.mItems = []
        self.pol = types.SimpleNamespace(objective=types.SimpleNamespace(num_inference_steps=10))
        if fakeOpen:
            self.mGripFakeOpen = False
            self.mGripMask = False

    def setDitSteps(self, steps):
        self.pol.objective.num_inference_steps = int(steps)

    def workspace(self, margin=0.05):
        return {s: (np.array([0.0, -0.7, -0.2]) - margin, np.array([0.8, 0.7, 0.6]) + margin) for s in SIDES}

    def item(self, imgs, st, prev=None):
        self.mItems.append(
            dict(state=np.asarray(st).copy(), prev=None if prev is None else np.asarray(prev[1]).copy(), imgs=sorted(imgs))
        )
        return len(self.mItems)

    def predict(self, item, seed=0, rtc=None):
        self.mCalls.append([item, seed, rtc is not None])
        self.mClock.advance(LATENCY_S)
        tRng = np.random.default_rng(1000 + seed)
        tRows = np.zeros((32, 20))
        for tSide in SIDES:
            for tK in range(32):
                tT = np.eye(4)
                tT[:3, :3] = Rotation.from_rotvec(
                    np.array([0.01, -0.02, 0.015]) * (tK + 1) / 8 + tRng.normal(size=3) * 0.002
                ).as_matrix()
                tT[:3, 3] = (
                    np.array([0.004, 0.002 if tSide == "left" else -0.002, -0.003]) * (tK + 1) + tRng.normal(size=3) * 0.0005
                )
                tRows[tK, POSE_SLICE[tSide]] = T_to_pose9(tT)
            tRows[:, GRIP_INDEX[tSide]] = np.clip(np.linspace(0.1, 0.9, 32) + tRng.normal(size=32) * 0.05, 0, 1)
        return tRows * 0.5, tRows


def _labelFile(tmp_path):
    tRng = np.random.default_rng(12)
    tN = 80
    tData = {"t_ns": (np.arange(tN) * 0.05e9).astype(np.int64)}
    for tSide in SIDES:
        tP0 = fakes.fakeFk(tSide, Q0[tSide])[:3, 3]
        tData[f"{tSide}_p_rob"] = tP0 + np.cumsum(tRng.normal(size=(tN, 3)) * 0.002, 0)
        tData[f"{tSide}_R_rob"] = (
            Rotation.from_rotvec(np.cumsum(tRng.normal(size=(tN, 3)) * 0.01, 0)).as_matrix()
            @ fakes.fakeFk(tSide, Q0[tSide])[:3, :3]
        )
        tData[f"{tSide}_g"] = (np.arange(tN) > 30).astype(float)
    tPath = tmp_path / "demo07.npz"
    np.savez(tPath, **tData)
    return tPath


def _runDry(
    tmp_path,
    monkeypatch,
    capsys,
    argv,
    cycles=4,
    attrs=None,
    history=False,
    staleImages=False,
    previewRes=None,
    calib=None,
    startPose=False,
    kind=None,
):
    tClock = fakes.FakeClock(5000.0)
    tCaptured = {}
    tSubmits = []

    class FakePreview:
        def __init__(self, arms, col, ws, a, table_z):
            self.mResult = previewRes
            self.mStop = False
            self.mLog = []
            tCaptured["preview_args"] = [ws, table_z]

        def start(self):
            pass

        def submit(self, t, Tc, q):
            tSubmits.append([t, {s: Tc[s][[0, 31]] for s in SIDES}, q])

    def fakeSummarize(L, a, stop_reason, n_viol, run_dir, WS, off, cfg, prev, ckpt, runner):
        tCaptured.update(L=dict(L), stop=stop_reason, n_viol=n_viol, run_dir=run_dir.name, off=off, ckpt=ckpt, cfg=cfg)
        return {"run": str(run_dir.name)}

    monkeypatch.setattr(session_mod, "time", tClock)
    monkeypatch.setattr(session_mod, "Preview", FakePreview)
    monkeypatch.setattr(session_mod, "summarize", fakeSummarize)
    monkeypatch.setattr(session_mod, "RUN_OUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(session_mod, "START_FILE", tmp_path / "start_pose.json")
    if startPose:
        (tmp_path / "start_pose.json").write_text(json.dumps({"q": {s: (Q0[s] + 0.01).tolist() for s in SIDES}, "episode": 104}))
    monkeypatch.setattr(heightCalib, "CALIB_FILE", tmp_path / "height_calib.json")
    if calib is not None:
        (tmp_path / "height_calib.json").write_text(json.dumps(calib))
    tArms, tRa = fakes.fakeArms()
    tQr = {s: Q0[s] - 0.02 for s in SIDES}
    tA = _parse(argv)
    for k, v in (attrs or {}).items():
        setattr(tA, k, v)
    tRunner = None if tA.policy == "label" else _Runner(tClock, kind=kind or ("multi_task_dit" if tA.policy == "dit" else "act"))
    tG = {"left": 0.7, "right": 0.2}
    tHist = (Q0, tG, False) if history else None
    tImg = np.zeros((216, 384, 3), np.uint8)
    tRob = types.SimpleNamespace(read=lambda: (Q0, tG, False), mRing=_Ring(tClock, (Q0, tG, False), history=tHist), mError=None)
    tFixed = int(5000.0 * 1e9) if staleImages else None
    tGl = types.SimpleNamespace(
        mRing=_Ring(tClock, (tImg + 1, b"jpg-global"), fixedT=tFixed, history=(tImg + 2, b"old") if history else None)
    )
    tWr = {
        s: types.SimpleNamespace(
            mRing=_Ring(tClock, tImg + (3 if s == "left" else 4), fixedT=tFixed, history=tImg + 5 if history else None)
        )
        for s in SIDES
    }
    tCfg = types.SimpleNamespace(robot={"base_url": "http://127.0.0.1:8021", "request_timeout_s": 0.2})
    tRig = types.SimpleNamespace(
        mArms=tArms,
        mRobotArgs=tRa,
        mCollision=fakes.FakeCollision(),
        mPoller=tRob,
        mGlobalCam=tGl,
        mWristCams=tWr,
        mQReady=tQr,
        mCamsCfg={"left": {}, "right": {}},
        mUrl="http://127.0.0.1:8031",
        mTeleopCfg=tCfg,
        mWaveOk=False,
        mReplay=False,
        mFakeQ=Q0,
        runner=lambda ckpt, device, n: (tCaptured.setdefault("runner_args", [ckpt, device, n]), tRunner)[1],
    )
    tSnaps = []
    tPolls = {"n": 0}

    def poll():
        tPolls["n"] += 1
        return "정지 (테스트)" if tPolls["n"] > cycles else None

    tCtl = types.SimpleNamespace(poll=poll, publish=lambda d: tSnaps.append(d), takeStep=lambda: False)
    tSumm = session_mod.runSession(tA, tRig, tCtl)
    tL = tCaptured.pop("L")
    tOut = dict(
        summ=tSumm,
        stdout=capsys.readouterr().out.replace(str(tmp_path), "<TMP>"),
        captured=tCaptured,
        a=vars(tA),
        submits=tSubmits,
        tip_clear=[tArms[s].mArgs.tip_clear for s in SIDES] + [tArms[s].mArgs.tip_clear_left_extra for s in SIDES],
        L={k: v for k, v in tL.items() if k not in ("gimg_jpg", "wl", "wr")},
        L_img=[len(tL["wl"]), len(tL["wr"]), tL["gimg_jpg"][:1]],
        snaps=[{k: v for k, v in s.items() if k not in ("Tm", "Tc")} for s in tSnaps],
        snap_keys=sorted(tSnaps[0]) if tSnaps else [],
        files=sorted(p.name for p in (tmp_path / "runs").rglob("*")),
        runner=None
        if tRunner is None
        else dict(
            calls=tRunner.mCalls,
            items=tRunner.mItems,
            dit_steps=tRunner.pol.objective.num_inference_steps,
            fake_open=getattr(tRunner, "mGripFakeOpen", None),
        ),
    )
    return tOut


BASE = ["run", "--ckpt", "fake", "--secs", "0"]
CALIB = {
    "created": "synthetic",
    "fit": {
        "left": {"n": 5, "dz_mean_mm": -3.0, "plane_dz": {"a": 0.08, "b": -0.03, "c": -0.004}},
        "right": {"n": 5, "dz_mean_mm": 9.0, "plane_dz": {"a": -0.05, "b": 0.02, "c": 0.011}},
    },
}


def test_run_dry_act_default(tmp_path, monkeypatch, capsys):
    checkGolden("session/run_act_default", _runDry(tmp_path, monkeypatch, capsys, BASE + ["--policy", "act"]))


def test_run_dry_act_commit4_amp_z_one_arm(tmp_path, monkeypatch, capsys):
    checkGolden(
        "session/run_act_c4",
        _runDry(
            tmp_path,
            monkeypatch,
            capsys,
            BASE
            + [
                "--commit",
                "4",
                "--speed",
                "0.7",
                "--z-offset-mm",
                "12",
                "--grip-mode",
                "sigmoid",
                "--temporal-ensemble",
                "0",
                "--init-grip",
                "-1",
            ],
            attrs=dict(amp=0.6, active_arms="left", grip_fake_open=True),
            startPose=True,
        ),
    )


def test_run_dry_dit_history_ensemble(tmp_path, monkeypatch, capsys):
    checkGolden(
        "session/run_dit_hist",
        _runDry(
            tmp_path,
            monkeypatch,
            capsys,
            BASE
            + [
                "--policy",
                "dit",
                "--commit",
                "4",
                "--speed",
                "0.5",
                "--grip-continuous",
                "--dit-steps",
                "6",
                "--dit-fixed-noise",
                "0",
            ],
            history=True,
            cycles=5,
        ),
    )


def test_run_dry_height_calib_no_safety(tmp_path, monkeypatch, capsys):
    checkGolden(
        "session/run_hc",
        _runDry(
            tmp_path,
            monkeypatch,
            capsys,
            BASE
            + [
                "--height-calib",
                "--commit",
                "8",
                "--grip-mode",
                "relu",
                "--grip-gain",
                "2",
                "--max-ik-mm",
                "0.1",
                "--max-ik-stop-mm",
                "0.5",
                "--tip-clear",
                "0.02",
                "--max-tcp-mps",
                "0.05",
                "--arm-clear",
                "0.4",
            ],
            attrs=dict(no_safety=True, no_floor=True),
            calib=CALIB,
            previewRes=dict(
                t=0.0,
                ik_ok=[1.0, 0.5],
                ik_max_mm=[2.0, 30.0],
                min_clear=0.05,
                min_table=0.01,
                flags=["HOLD:future_table_k8", "WARN:future_ik_l_30mm"],
                ms=3.0,
            ),
        ),
    )


def test_run_dry_height_calib_missing(tmp_path, monkeypatch, capsys):
    with pytest.raises(DeployError) as tExc:
        _runDry(tmp_path, monkeypatch, capsys, BASE + ["--height-calib"])
    checkGolden("session/run_hc_missing", str(tExc.value))


def test_run_dry_label(tmp_path, monkeypatch, capsys):
    tPath = _labelFile(tmp_path)
    tOut = _runDry(
        tmp_path,
        monkeypatch,
        capsys,
        ["run", "--policy", "label", "--label", str(tPath), "--label-rate", "0.5", "--commit", "6"],
        cycles=40,
        attrs=dict(no_safety=True),
    )
    tOut["a"]["label"] = "demo07.npz"
    tOut["stdout"] = tOut["stdout"].replace(str(tPath), "demo07.npz")
    tOut["captured"]["ckpt"] = "demo07.npz"
    checkGolden("session/run_label", tOut)


def test_run_dry_stale_images_grip_max(tmp_path, monkeypatch, capsys):
    checkGolden(
        "session/run_stale",
        _runDry(
            tmp_path,
            monkeypatch,
            capsys,
            BASE + ["--max-img-age-ms", "200", "--grip-at", "max", "--grip-thr", "0.3", "--commit", "2", "--print-every", "1"],
            staleImages=True,
        ),
    )


def test_run_dry_dit_fast_xvla_kind(tmp_path, monkeypatch, capsys):
    checkGolden(
        "session/run_dit_fast",
        _runDry(
            tmp_path,
            monkeypatch,
            capsys,
            BASE + ["--policy", "dit", "--speed", "2.0", "--commit", "20", "--grip-at", "last", "--rtc", "guided"],
        ),
    )
    checkGolden(
        "session/run_xvla_kind",
        _runDry(
            tmp_path / "x",
            monkeypatch,
            capsys,
            BASE + ["--policy", "act", "--speed", "0.4", "--commit", "3"],
            kind="xvla",
            attrs=dict(time_scale=None),
        ),
    )


def test_run_refusals(tmp_path, monkeypatch, capsys):
    tOut = {}
    tA = _parse(BASE + ["--go"])
    tRig = types.SimpleNamespace(
        mWaveOk=False,
        mReplay=False,
        mFakeQ=None,
        mArms=fakes.fakeArms()[0],
        mRobotArgs=None,
        mCollision=None,
        mPoller=None,
        mGlobalCam=None,
        mWristCams=None,
        mQReady=Q0,
        mCamsCfg={},
    )
    for tName, tPatch in (
        ("no_wave", {}),
        ("replay", dict(mWaveOk=True, mReplay=True)),
        ("fake_q", dict(mWaveOk=True, mFakeQ=Q0)),
    ):
        for k, v in tPatch.items():
            setattr(tRig, k, v)
        try:
            session_mod.runSession(tA, tRig, types.SimpleNamespace(poll=lambda: None))
        except DeployError as e:
            tOut[tName] = str(e)
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("session/run_refusals", tOut)


# ------------------------------------------------------------------------------------------------ start pose (fake arms, label start)
def test_start_cmd_label(tmp_path, monkeypatch, capsys):
    tTarget = {s: fakes.fakeFk(s, Q0[s]) for s in SIDES}
    monkeypatch.setattr(start_mod, "LabelRunner", lambda path, *a: types.SimpleNamespace(poseAt=lambda s, t: tTarget[s].copy()))
    monkeypatch.setattr(start_mod, "makeArms", lambda **k: fakes.fakeArms(**k))
    monkeypatch.setattr(start_mod, "Collision", lambda arms: fakes.FakeCollision())
    import laundry_shared.visual_teleop.tools.robot_moves as rm

    monkeypatch.setattr(rm, "READY_Q_DEG", {s: np.degrees(Q0[s] * 0.2) for s in SIDES})
    monkeypatch.setattr(start_mod, "START_FILE", tmp_path / "start_pose.json")
    monkeypatch.setattr(start_mod, "time", fakes.FakeClock())
    monkeypatch.setattr(heightCalib, "CALIB_FILE", tmp_path / "height_calib.json")
    (tmp_path / "height_calib.json").write_text(json.dumps({"fit": {"left": {"xy_B_minus_robot": [0.006, -0.004]}, "right": {}}}))
    import sys

    tLerobot = types.ModuleType("lerobot.datasets.lerobot_dataset")
    tLerobot.LeRobotDataset = None
    for tName in ("lerobot", "lerobot.datasets"):
        monkeypatch.setitem(sys.modules, tName, sys.modules.get(tName) or types.ModuleType(tName))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", tLerobot)
    tOut = {}
    for tName, tExtra in (("plain", []), ("hc", ["--height-calib"]), ("lift", ["--start-lift", "0.05", "--deg-s", "20"])):
        tA = _parse(["start", "--label", "x.npz"] + tExtra)
        try:
            start_mod.startCmd(tA)
            tOut[tName] = json.loads((tmp_path / "start_pose.json").read_text())
        except DeployError as e:
            tOut[tName] = "exit: " + str(e)
    tOut["stdout"] = capsys.readouterr().out.replace(str(tmp_path), "<TMP>")
    checkGolden("session/start_cmd", tOut)


# ------------------------------------------------------------------------------------------------ reports
def _logL(n=6):
    tRng = np.random.default_rng(13)
    tL = {
        k: []
        for k in (
            "t",
            "lat_obs_ms",
            "lat_pol_ms",
            "lat_ctrl_ms",
            "lat_total_ms",
            "obs_to_commit_ms",
            "k0",
            "hist_used",
            "T_meas",
            "T_chunk",
            "grip_raw",
            "ik_mm",
            "ws_chunk_out",
            "arm_clear",
            "link_gap",
            "img_age_ms",
            "flags",
            "gimg_jpg",
            "wl",
            "wr",
            "dchunk_k1_mm",
            "dchunk_k8_mm",
            "dchunk_k16_mm",
            "dchunk_k32_mm",
            "speed",
            "g_pred",
            "pv_ik_ok",
        )
    }
    for tI in range(n):
        tL["t"].append(0.25 * tI + 0.1)
        for k in ("lat_obs_ms", "lat_pol_ms", "lat_ctrl_ms", "lat_total_ms", "obs_to_commit_ms"):
            tL[k].append(float(tRng.random() * 100))
        tL["k0"].append(int(tI % 3))
        tL["hist_used"].append(bool(tI % 2))
        tL["T_meas"].append(np.stack([_pose(tRng) for _ in SIDES]))
        tL["T_chunk"].append(np.stack([[_pose(tRng) for _ in range(32)] for _ in SIDES]))
        tL["grip_raw"].append(tRng.random((32, 2)) * 1.2 - 0.1)
        tL["ik_mm"].append(tRng.random((2, 2)).tolist())
        tL["ws_chunk_out"].append(tRng.random(2).tolist())
        tL["arm_clear"].append(float(tRng.random()))
        tL["link_gap"].append(float(tRng.random()))
        tL["img_age_ms"].append(tRng.random(3).tolist())
        tL["flags"].append(["LIM:tcp_speed_l", "STOP:ik_l_30mm_5deg", "WARN:link_gap_100mm", "nocolon"][: 1 + tI % 4])
        tL["gimg_jpg"].append(b"x")
        tL["wl"].append(np.zeros((2, 2, 3), np.uint8))
        tL["wr"].append(np.zeros((2, 2, 3), np.uint8))
        for k in (1, 8, 16, 32):
            tL[f"dchunk_k{k}_mm"].append(np.nan if tI == 0 else float(tRng.random() * 10))
        tL["speed"].append(0.5)
        tL["g_pred"].append([0.2, 0.8])
        tL["pv_ik_ok"].append([1.0, 1.0])
    return tL


def test_report_summarize_and_topdown(tmp_path, capsys):
    tRng = np.random.default_rng(14)
    tWs = {s: (np.array([-0.1, -0.5, 0.0]), np.array([0.6, 0.5, 0.4])) for s in SIDES}
    tA = _parse(["run", "--no-video", "--replay-global", "/takes/x"])
    tA.commit = 1
    tPrev = types.SimpleNamespace(
        mLog=[
            dict(t=0.1, ik_ok=[1.0, 0.75], ik_max_mm=[2.0, 11.0], min_clear=0.03, min_table=0.01, flags=[], ms=4.0),
            dict(t=0.3, ik_ok=[1.0, 1.0], ik_max_mm=[1.0, 3.0], min_clear=0.05, min_table=0.02, flags=[], ms=6.0),
        ]
    )
    tRun = tmp_path / "act_dry_20261104_120000"
    tRun.mkdir()
    tS = report_mod.summarize(
        _logL(),
        tA,
        "max time 60s",
        2,
        tRun,
        tWs,
        {"left": 1.0, "right": 2.0},
        {"left": {"location": "0x1"}, "right": {"location": "0x2"}, "wave_test": {"confirmed": True}},
        tPrev,
        "ck",
        None,
    )
    tZ = np.load(tRun / "log.npz", allow_pickle=True)
    tEmpty = report_mod.summarize({"t": []}, tA, "x", 0, tRun, tWs, {}, {}, tPrev, "ck", None)
    tImg = report_mod.topDownView(
        {s: _pose(tRng) for s in SIDES},
        {s: [_pose(tRng) for _ in range(32)] for s in SIDES},
        tWs,
        tRng.normal(size=(20, 3)) * 0.2,
    )
    checkGolden(
        "session/report",
        dict(
            S={k: v for k, v in tS.items() if k != "run"},
            files=sorted(p.name for p in tRun.iterdir()),
            npz={k: (tZ[k].shape, str(tZ[k].dtype)) for k in sorted(tZ.files)},
            npz_flags=tZ["flags"],
            npz_preview=tZ["preview"],
            empty=tEmpty,
            stdout=capsys.readouterr().out.replace(str(tmp_path), "<TMP>"),
            topdown=sha256(tImg.tobytes()),
            COL=report_mod.ARM_COLOURS_BGR,
        ),
    )


def test_report_grip_plot(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "DESKTOP", tmp_path)
    tL = _logL()
    tArr = {"t": np.asarray(tL["t"]), "grip_raw": np.asarray(tL["grip_raw"])}
    tP = report_mod.gripPlot(tArr, tmp_path / "act_dry_x")
    import cv2

    checkGolden("session/grip_plot", dict(name=tP.name, img=sha256(cv2.imread(str(tP)).tobytes())))


# ------------------------------------------------------------------------------------------------ cameras: mapping helpers, white balance
UVC_LIST = """Index  Vend:Prod   LocationID UVC  Name
0      0x0c45:0x6366 0x01100000 1.00 Arducam 1080P Low Light
1      0x0c45:0x6366 0x01200000 1.00 FisheyeCamLeft
2      0x046d:0x085c 0x02100000 1.00 C922 Pro Stream Webcam"""


def test_camera_mapping(tmp_path, monkeypatch, capsys):
    tOut = {}
    tCfgFile = tmp_path / "act_tagrig_cams.json"
    monkeypatch.setattr(cameras_mod, "CAMS_FILE", tCfgFile)
    monkeypatch.setattr(cameras_mod, "time", fakes.FakeClock())
    monkeypatch.setattr(cameras_mod, "uvcList", lambda: UVC_LIST)
    tOut["cam_key"] = cameras_mod.camKey()
    import laundry_shared.visual_teleop.tools.stereo_check as sc

    monkeypatch.setattr(sc, "index_for_location", lambda loc, candidates=None: {"0x01100000": 3, "0x01200000": 1}[loc])
    tCfg = {
        "left": {"location": "0x0aaaaaaa", "name": "Arducam 1080P Low Light", "index": 0},
        "right": {"location": "0x01200000", "name": "FisheyeCamLeft", "index": 2},
        "cam_key": "old",
        "wave_test": {"confirmed": True},
    }
    tOut["remap"] = cameras_mod.autoRemapCams(tCfg, UVC_LIST)
    tOut["remap_file"] = json.loads(tCfgFile.read_text())
    tOut["remap_bak"] = json.loads((tmp_path / "act_tagrig_cams.json.bak_auto").read_text())
    tSame = dict(tOut["remap"])
    tOut["remap_noop"] = cameras_mod.autoRemapCams(tSame, UVC_LIST) is tSame
    try:
        cameras_mod.autoRemapCams(
            {"left": {"location": "0xdead", "name": "nope"}, "right": {"location": "0xbeef", "name": "FisheyeCamLeft"}}, UVC_LIST
        )
    except DeployError as e:
        tOut["remap_missing"] = str(e)
    import laundry_shared.visual_teleop.tools.umi_jaw as uj

    monkeypatch.setattr(uj, "LOCATIONS", {"left": "0x0110", "right": "0x0300"})
    tOut["umi_warn"] = cameras_mod.umiLocationWarning({"left": {"location": "0x01100000"}, "right": {"location": "0x01200000"}})
    tOut["umi_none"] = cameras_mod.umiLocationWarning({"left": {"location": "0x0900"}, "right": {}})
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            stdout="111 python tag_rig_record.py\n222 umi_jaw infer x\n333 python umi_jaw record --hand left\n"
        ),
    )
    monkeypatch.setattr(cameras_mod.os, "getpid", lambda: 111)
    tOut["recorder"] = cameras_mod.recorderRunning()
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("session/camera_mapping", tOut)


def test_white_balance():
    tRng = np.random.default_rng(15)
    tCam = object.__new__(cameras_mod.WristCam)
    tCam.mWb = None
    tOut = []
    for tI in range(12):
        tCam.mFrames = tI
        tF = np.clip(tRng.normal(140, 50, size=(240, 320, 3)) * [1.0, 1.1, 0.8], 0, 255).astype(np.uint8)
        tF[:60] = np.clip(tF[:60] * 0.25, 0, 255).astype(np.uint8)
        tOut.append(sha256(tCam.whiteBalance(tF).tobytes()))
        tOut.append(None if tCam.mWb is None else tCam.mWb.tolist())
    checkGolden("session/white_balance", tOut)


def test_rig_runner_cache(tmp_path, monkeypatch):
    tMade = []
    monkeypatch.setattr(rig_mod, "XvlaRunner", lambda ckpt: tMade.append(("xvla", str(ckpt))) or "xvla")
    monkeypatch.setattr(
        rig_mod,
        "PolicyRunner",
        lambda ckpt, device, nSamples=1: tMade.append(("policy", str(ckpt), device, nSamples)) or "policy",
    )
    (tmp_path / "x" / "pretrained_model").mkdir(parents=True)
    (tmp_path / "x" / "pretrained_model" / "config.json").write_text('{"type": "xvla"}')
    (tmp_path / "d" / "pretrained_model").mkdir(parents=True)
    (tmp_path / "d" / "pretrained_model" / "config.json").write_text('{"type": "multi_task_dit"}')
    tRig = object.__new__(rig_mod.Rig)
    tRig.mRunners = {}
    tOut = [
        tRig.runner(tmp_path / "x", "mps", 1),
        tRig.runner(tmp_path / "d", "cpu", 2),
        tRig.runner(tmp_path / "d", "cpu", 2),
        tRig.runner(tmp_path / "missing", "mps", 1),
        len(tRig.mRunners),
    ]
    checkGolden(
        "session/rig_runner", dict(out=tOut, made=[[m[0], m[1].replace(str(tmp_path), "<TMP>")] + list(m[2:]) for m in tMade])
    )
