"""Deterministic stand-ins for the hardware side of the deploy (arms, bridge, clock, cameras) used by the characterization
tests: no URDF, no robot, no camera, no network. The arm model is a fixed linear map q -> TCP, so every number depends only
on the code under test."""

from __future__ import annotations
import argparse
import runpy
import sys
import types
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
_RNG = np.random.default_rng(1234)
TRANS_MAP = {s: _RNG.normal(size=(3, 6)) * 0.08 for s in SIDES}
ROT_MAP = {s: _RNG.normal(size=(3, 6)) * 0.3 for s in SIDES}
BASE_P = {"left": np.array([0.35, 0.25, 0.20]), "right": np.array([0.35, -0.25, 0.20])}
IK_MAX_STEP_RAD = 0.2


def fakeFk(side, q):
    q = np.asarray(q, np.float64)
    tT = np.eye(4)
    tT[:3, :3] = Rotation.from_rotvec(ROT_MAP[side] @ q).as_matrix()
    tT[:3, 3] = BASE_P[side] + TRANS_MAP[side] @ q
    return tT


class FakeIk:
    def __init__(self, side, maxStep=IK_MAX_STEP_RAD):
        self.mSide = side
        self.mMaxStep = maxStep
        self.mCalls = 0

    def ik(self, qPrev, target):
        self.mCalls += 1
        tQ = np.asarray(qPrev, np.float64).copy()
        tDp = np.asarray(target)[:3, 3] - fakeFk(self.mSide, tQ)[:3, 3]
        tDq = np.linalg.pinv(TRANS_MAP[self.mSide]) @ tDp
        tM = float(np.abs(tDq).max())
        if tM > self.mMaxStep:
            tDq = tDq * self.mMaxStep / tM
        tQ = tQ + tDq
        tT = fakeFk(self.mSide, tQ)
        tPe = float(np.linalg.norm(tT[:3, 3] - np.asarray(target)[:3, 3]))
        tRe = float(np.degrees(np.arccos(np.clip((np.trace(tT[:3, :3].T @ np.asarray(target)[:3, :3]) - 1) / 2, -1, 1))))
        return tQ, tPe, tRe


class FakePk:
    def __init__(self, side):
        self.mSide = side

    def fk(self, q):
        return fakeFk(self.mSide, q)


class FakeArm:
    """laundry_dataset.Arm surface used by the deploy: limits, two IK solvers, FK, finger-corner floor, link origins."""

    def __init__(self, side, args):
        self.mSide = side
        self.mArgs = args
        self.mLo = np.full(6, -1.2)
        self.mHi = np.full(6, 1.2)
        self.mLimMargin = np.radians(float(getattr(args, "lim_margin_deg", 0.5)))
        self.mFree = FakeIk(side)
        self.mF0 = FakeIk(side, maxStep=3.0)
        self.mPk = FakePk(side)

    def cornerLift(self, T, g):
        tLow = T[2, 3] - 0.03 * (1.0 - float(g)) - 0.01
        return max(
            0.0,
            self.mArgs.table_z + self.mArgs.tip_clear + (self.mArgs.tip_clear_left_extra if self.mSide == "left" else 0.0) - tLow,
        )

    def links(self, q6):
        tP = fakeFk(self.mSide, q6)[:3, 3]
        return np.stack([tP, tP + [0.0, 0.0, 0.1], tP + [-0.1, 0.0, 0.2]])


def fakeArms(**over):
    tArgs = types.SimpleNamespace(
        **{
            **dict(
                ik_step=7.5,
                ori_cost=0.1,
                table_z=0.003,
                tip_clear=0.005,
                tip_clear_left_extra=0.005,
                jaw_half=0.020,
                level_max=0.0,
                roll_thr=115.0,
                roll_relax=1.5,
            ),
            **over,
        }
    )
    return {s: FakeArm(s, tArgs) for s in SIDES}, tArgs


class FakeCollision:
    """Collision.check surface: arm-arm clearance from the TCP distance, table clearance from the lowest TCP."""

    def check(self, q, tableZ=0.003, armClear=0.010, tableClear=0.0):
        tP = {s: fakeFk(s, q[s])[:3, 3] for s in SIDES}
        tArm = float(np.linalg.norm(tP["left"] - tP["right"]) - 0.30)
        tTab = float(min(tP[s][2] for s in SIDES) - 0.05 - tableZ)
        return dict(
            arm_clear_m=tArm,
            pair="L.link5-link6 / R.link5-link6",
            table_clear_m=tTab,
            finger_low_m=tTab - 0.01,
            ok_arm=bool(tArm >= armClear),
            ok_table=bool(tTab >= tableClear),
        )


class FakeClock:
    """time module stand-in: the clock only moves on sleep() and advance(); strftime is fixed."""

    def __init__(self, t0=1000.0):
        self.mT = float(t0)

    def monotonic(self):
        return self.mT

    def monotonic_ns(self):
        return int(round(self.mT * 1e9))

    def time(self):
        return self.mT

    def perf_counter(self):
        return self.mT

    def sleep(self, seconds):
        self.mT += max(0.0, float(seconds))

    def advance(self, seconds):
        self.mT += float(seconds)

    def strftime(self, fmt, *a):
        return "20261104_120000" if "%Y%m%d" in fmt else ("2026-11-04 12:00:00" if "%Y" in fmt else "120000")


class Result:
    def __init__(self, accepted=True, error=None):
        self.accepted = accepted
        self.error = error


class Observation:
    def __init__(self, q, g=None, estop=False):
        self.q = {s: np.asarray(v, np.float64).copy() for s, v in q.items()}
        self.mG = g or {s: 0.0 for s in SIDES}
        self.estop = estop
        self.latency_ms = 3.0
        self.tau = None  # 11-06 MIT bridge /observe tau_nm; the fake robot reports none

    def g(self, side):
        return self.mG[side]


class FakeHttpSession:
    def __init__(self, owner):
        self.mOwner = owner

    def post(self, url, json=None, timeout=None):
        self.mOwner.mLog.append(("POST", url.replace(self.mOwner.base_url, ""), json))
        tReply = self.mOwner.mReplies.get(url.replace(self.mOwner.base_url, ""), {"ok": True})
        return types.SimpleNamespace(json=lambda: tReply)

    def get(self, url, timeout=None):
        self.mOwner.mLog.append(("GET", url.replace(self.mOwner.base_url, ""), None))
        tReply = self.mOwner.mReplies.get(url.replace(self.mOwner.base_url, ""), {})
        return types.SimpleNamespace(json=lambda: tReply)


class FakeBridge:
    """HttpBridge surface: observe / send / estop / status / close and the raw session `s` for POSTs; logs every call."""

    base_url = "http://bridge"

    def __init__(self, q, g=None, status=None, reject_at=None, estop_at=None):
        self.mQ = {s: np.asarray(v, np.float64).copy() for s, v in q.items()}
        self.mG = g
        self.mStatus = status or {}
        self.mLog = []
        self.mSends = []
        self.mRejectAt = reject_at
        self.mEstopAt = estop_at
        self.mReplies = {}
        self.s = FakeHttpSession(self)

    def observe(self):
        self.mLog.append(("observe",))
        return Observation(self.mQ, self.mG, estop=self.mEstopAt is not None and len(self.mSends) >= self.mEstopAt)

    def send(self, q, g):
        self.mSends.append(({s: np.asarray(v).copy() for s, v in q.items()}, dict(g)))
        self.mQ = {s: np.asarray(v, np.float64).copy() for s, v in q.items()}
        if self.mRejectAt is not None and len(self.mSends) >= self.mRejectAt:
            return Result(False, "rejected by test")
        return Result(True)

    def estop(self):
        self.mLog.append(("estop",))

    def status(self):
        self.mLog.append(("status",))
        return dict(self.mStatus)

    def close(self):
        self.mLog.append(("close",))


class _Captured(Exception):
    pass


def captureParser(moduleName: str, argv: list[str], parseArgv: list[str] | None = None):
    """Run `python -m moduleName argv...` until its first ArgumentParser.parse_args call and return what that parser makes
    of parseArgv (default: argv). Nothing after the parse runs."""
    tHolder = {}
    tOrig = argparse.ArgumentParser.parse_args

    def _grab(self, args=None, namespace=None):
        tHolder["parser"] = self
        raise _Captured()

    tSavedArgv = sys.argv
    argparse.ArgumentParser.parse_args = _grab
    sys.argv = [moduleName] + list(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # runpy: the module is already imported by the tests
            runpy.run_module(moduleName, run_name="__main__", alter_sys=False)
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = tOrig
        sys.argv = tSavedArgv
    return tOrig(tHolder["parser"], list(argv if parseArgv is None else parseArgv))
