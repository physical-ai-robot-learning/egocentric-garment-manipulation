"""Core of the deploy: image / state / chunk contract, RTC weights, retiming, the policy runners' pure parts, geometry and the
capsule collision model (on a small synthetic pinocchio chain)."""

from __future__ import annotations
import io
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")

from laundry_shared.contracts.action_contract import T_to_pose9, make_T

from laundry_deploy.core import camera_model, chunk, collision, contract, images, kinematics, plot
from laundry_deploy.core.label_runner import LabelRunner
from laundry_deploy.core.policy_runner import PolicyRunner
from laundry_deploy.core.xvla_runner import XvlaRunner
from laundry_deploy.tests import fakes
from laundry_deploy.tests.golden_record import checkGolden, sha256

SIDES = ("left", "right")


def _rng(seed=0):
    return np.random.default_rng(seed)


def _pose(rng):
    tT = np.eye(4)
    tT[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.3).as_matrix()
    tT[:3, 3] = rng.normal(size=3) * 0.2
    return tT


def _chunk(rng, scale=0.01):
    tRows = np.zeros((32, 20))
    for tSide in SIDES:
        for tK in range(32):
            tT = np.eye(4)
            tT[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * scale * 3).as_matrix()
            tT[:3, 3] = rng.normal(size=3) * scale * (tK + 1) / 4
            tRows[tK, contract.POSE_SLICE[tSide]] = T_to_pose9(tT)
        tRows[:, contract.GRIP_INDEX[tSide]] = rng.random(32)
    return tRows


def test_contract_constants():
    checkGolden(
        "core/constants",
        dict(
            IMG_WH=contract.IMG_WH,
            CAMS=contract.CAMS,
            TASK=contract.TASK,
            SIDES=contract.SIDES,
            DT=contract.ROW_DT_S,
            SL={k: [v.start, v.stop] for k, v in contract.POSE_SLICE.items()},
            GI=contract.GRIP_INDEX,
            ROBOTIZE_ARGS=kinematics.ROBOTIZE_ARGS,
            CAPSULES=collision.CAPSULES,
            FINGER_R=collision.FINGER_RADIUS_M,
            TABLE_R_WRIST=collision.WRIST_TABLE_RADIUS_M,
            XVLA_RY=XvlaRunner.RY,
            XVLA_ROW_DT=XvlaRunner.ROW_DT_S,
        ),
    )


def test_images():
    tRng = _rng(1)
    tGlobal = (tRng.random((480, 848, 3)) * 255).astype(np.uint8)
    tWrist = (tRng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    tWrist720 = (tRng.random((720, 1280, 3)) * 255).astype(np.uint8)
    tOut = dict(
        global_=sha256(images.prepGlobal(tGlobal).tobytes()),
        wrist=sha256(images.prepWrist(tWrist).tobytes()),
        wrist_noq=sha256(images.prepWrist(tWrist, jpegQ=0).tobytes()),
        wrist720=sha256(images.prepWrist(tWrist720, jpegQ=80).tobytes()),
    )
    for tName, tAlign in (
        ("rot", dict(rot180=True)),
        ("flip", dict(hflip=True)),
        ("zoom_out", dict(scale=0.9, dx=5, dy=-3)),
        ("zoom_in", dict(scale=1.1)),
        ("shift", dict(dx=4.0)),
        ("none", {}),
    ):
        tOut[f"align_{tName}"] = sha256(images.applyAlign(tWrist720, tAlign).tobytes())
        tOut[f"wrist_align_{tName}"] = sha256(images.prepWrist(tWrist, align=tAlign).tobytes())
    checkGolden("core/images", tOut)


def test_items():
    tRng = _rng(2)
    tImgs = {c: (tRng.random((216, 384, 3)) * 255).astype(np.uint8) for c in contract.CAMS}
    tOld = {c: (tRng.random((216, 384, 3)) * 255).astype(np.uint8) for c in contract.CAMS}
    tState, tStateOld = tRng.normal(size=30), tRng.normal(size=30)

    def summary(item):
        return {
            k: (v if isinstance(v, str) else [list(v.shape), str(v.dtype), round(float(v.double().sum()), 6)])
            for k, v in item.items()
        }

    checkGolden(
        "core/items",
        dict(
            one=summary(images.makeItem(tImgs, tState)),
            two_none=summary(images.makeItemPair(tImgs, tState)),
            two_prev=summary(images.makeItemPair(tImgs, tState, (tOld, tStateOld))),
        ),
    )


def test_state_decode_rtc_retime():
    tRng = _rng(3)
    tTl, tTr, tAl, tAr = (_pose(tRng) for _ in range(4))
    tChunk = _chunk(tRng)
    tDec = chunk.decodeChunk(tChunk, tTl, tTr)
    tOut = dict(
        state=chunk.buildState(tTl, tTr, tAl, tAr, 1.0, 0.0),
        state_same=chunk.buildState(tTl, tTr, tTl, tTr, 0.0, 1.0),
        decode={s: np.asarray(tDec[s]) for s in SIDES},
        rtc=[chunk.rtcWeights(32, d, n) for d, n in ((2, 32), (5, 20), (0, 0), (40, 32), (3, 3), (0, 10), (-2, 50))],
        retime={
            f"{dt}_{ts}": chunk.retime(tChunk, dt, ts) for dt, ts in ((0.05, 0.5), (1 / 15, 1.0), (0.05, 2.0), (1 / 15, 0.2))
        },
    )
    checkGolden("core/state_decode_rtc_retime", tOut)


def test_label_runner(tmp_path, monkeypatch):
    tRng = _rng(4)
    tN = 60
    tT = (np.arange(tN) * 0.05e9).astype(np.int64) + 123
    tData = {"t_ns": tT}
    for tSide in SIDES:
        tData[f"{tSide}_p_rob"] = np.cumsum(tRng.normal(size=(tN, 3)) * 0.005, 0) + [0.3, 0.2 if tSide == "left" else -0.2, 0.1]
        tData[f"{tSide}_R_rob"] = Rotation.from_rotvec(np.cumsum(tRng.normal(size=(tN, 3)) * 0.02, 0)).as_matrix()
        tData[f"{tSide}_g"] = (np.arange(tN) % 17 > 8).astype(float)
    tPath = tmp_path / "demo00.npz"
    np.savez(tPath, **tData)
    tRunner = LabelRunner(tPath, rate=0.5)
    tOut = dict(
        kind=tRunner.mKind,
        n_obs=tRunner.mNObs,
        dur=tRunner.mDurS,
        tau_none=tRunner.tau(),
        item=tRunner.item(None, None),
        T_at={f"{s}_{tau}": tRunner.poseAt(s, tau) for s in SIDES for tau in (-1.0, 0.0, 0.73, 2.9, 99.0)},
        g_at={f"{s}_{tau}": tRunner.gripAt(s, tau) for s in SIDES for tau in (-1.0, 0.0, 0.51, 1.0, 2.94, 99.0)},
        workspace=tRunner.workspace(0.05),
        workspace0=tRunner.workspace(0.0),
        done_none=tRunner.done(),
    )
    tRunner.mCursor = 0.7
    tOut["predict_cursor"] = tRunner.predict(None)[0]
    tRunner.mCursor = tRunner.mDurS
    tOut["done_end"] = tRunner.done()
    tRunner.mCursor = None
    tRunner.mT0 = 100.0
    monkeypatch.setattr("time.monotonic", lambda: 101.0)
    tOut["tau_clock"] = tRunner.tau()
    tOut["predict_clock"] = tRunner.predict(None, seed=3)[1]
    checkGolden("core/label_runner", tOut)


class _FakeSession:
    def __init__(self, chunk):
        self.mChunk = chunk
        self.mPosted = []

    def post(self, url, data=None, json=None, timeout=None):
        self.mPosted.append((url, data, json, timeout))
        tBuf = io.BytesIO()
        np.savez(tBuf, chunk=self.mChunk, ms=np.float64(1.0))
        return types.SimpleNamespace(content=tBuf.getvalue(), raise_for_status=lambda: None)


def _xvla(rng, ckpt="ck/XVLA"):
    tRunner = object.__new__(XvlaRunner)
    tRunner.mUrl = "http://x"
    tRunner.mCkpt = ckpt
    tRunner.mRh = make_T(XvlaRunner.RY, np.zeros(3))
    tRunner.mStats = {"pose_abs.min": rng.normal(size=18) - 1, "pose_abs.max": rng.normal(size=18) + 1}
    return tRunner


def test_xvla_runner():
    tRng = _rng(5)
    tRunner = _xvla(tRng)
    tState = np.r_[_chunk(tRng)[3, :18], 0.3, 0.8, np.zeros(10)]
    tImgs = {c: (tRng.random((216, 384, 3)) * 255).astype(np.uint8) for c in contract.CAMS}
    tOut = dict(
        item=tRunner.item(tImgs, tState)["state20"],
        time_scale_default=XvlaRunner.mTimeScale,
        grip_fake_open_default=XvlaRunner.mGripFakeOpen,
        workspace=tRunner.workspace(0.05),
    )
    tRunner.mGripFakeOpen = True
    tOut["item_fake_open"] = tRunner.item(tImgs, tState)["state20"]
    tRunner.mCkpt = "ck/XVLA-GSM"
    tOut["item_gsm"] = tRunner.item(tImgs, tState)["state20"]
    tP9 = _chunk(tRng)[5, :9]
    tOut["to_x"] = tRunner._toX(tP9)
    tOut["from_x"] = tRunner._fromX(tP9)
    tOut["roundtrip"] = tRunner._fromX(tRunner._toX(tP9))
    tRaw = np.zeros((32, 32))
    tRaw[:, 0:9] = np.stack([tRunner._toX(r) for r in _chunk(tRng)[:, :9]])
    tRaw[:, 10:19] = np.stack([tRunner._toX(r) for r in _chunk(tRng)[:, 9:18]])
    tRaw[:, 9] = tRng.random(32)
    tRaw[:, 19] = tRng.random(32)
    tSession = _FakeSession(tRaw)
    tRunner.mSession = tSession
    tItem = tRunner.item(tImgs, tState)
    for tName, tScale, tRtc in (
        ("ts1", 1.0, None),
        ("ts05", 0.5, None),
        ("raw", None, None),
        ("rtc", 1.0, dict(Y=_chunk(tRng), W=chunk.rtcWeights(32, 3, 20))),
    ):
        tRunner.mTimeScale = tScale
        tN, tD = tRunner.predict(tItem, seed=7, rtc=tRtc)
        tZ = np.load(io.BytesIO(tSession.mPosted[-1][1]))
        tOut[f"predict_{tName}"] = dict(
            norm=tN,
            chunk=tD,
            url=tSession.mPosted[-1][0],
            timeout=tSession.mPosted[-1][3],
            posted={k: tZ[k] for k in sorted(tZ.files) if k not in contract.CAMS},
            posted_imgs={k: sha256(tZ[k].tobytes()) for k in contract.CAMS},
        )
    checkGolden("core/xvla_runner", tOut)


class _FakeScheduler:
    def __init__(self):
        self.alphas_cumprod = torch.linspace(0.999, 0.05, 100, dtype=torch.float64)
        self.timesteps = []

    def set_timesteps(self, n):
        self.timesteps = torch.arange(99, -1, -(100 // n))[:n]

    def step(self, eps, t, x):
        tA = self.alphas_cumprod[int(t)]
        return types.SimpleNamespace(prev_sample=(x - (1 - tA).sqrt() * eps) * 0.9 + 0.05 * eps)


class _FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mW = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 20, dtype=torch.float64))

    def forward(self, x, t, conditioning_vec=None):
        return torch.tanh(x * self.mW) * 0.7 + conditioning_vec.sum() * 0.01 + t.double().view(-1, 1, 1) * 1e-3


class _FakeDiT:
    def __init__(self):
        self.noise_predictor = _FakeNet()
        self.objective = types.SimpleNamespace(
            horizon=33, action_dim=20, num_inference_steps=10, noise_scheduler=_FakeScheduler(), conditional_sample=self._sample
        )

    def _sample(self, net, k, cond):
        return torch.ones((k, 33, 20), dtype=torch.float64) * 0.25 + torch.randn((k, 33, 20), dtype=torch.float64) * 0.01

    def _prepare_batch(self, batch):
        return batch

    observation_encoder = types.SimpleNamespace(encode=lambda b: b["observation.state"].double()[..., :8].reshape(1, -1))


def _policyRunner(rng, kind="multi_task_dit", n_obs=2):
    tRunner = object.__new__(PolicyRunner)
    tRunner.mDevice = "cpu"
    tRunner.mNSamples = 1
    tRunner.mKind = kind
    tRunner.mNObs = n_obs
    tRunner.mGripMask = False
    tRunner.mGripFakeOpen = False
    tRunner.mStats = {
        "action.min": rng.normal(size=20) - 2,
        "action.max": rng.normal(size=20) + 2,
        "pose_abs.min": rng.normal(size=18) - 1,
        "pose_abs.max": rng.normal(size=18) + 1,
    }
    tRunner.mPre = lambda b: b
    tRunner.mPost = lambda n: n * 2.0 + 0.1
    return tRunner


def test_policy_runner_pure_parts():
    tRng = _rng(6)
    tRunner = _policyRunner(tRng)
    tImgs = {c: (tRng.random((216, 384, 3)) * 255).astype(np.uint8) for c in contract.CAMS}
    tState = tRng.normal(size=30)
    tOut = dict(
        jaw_plain=tRunner._jawState(tState),
        normalize=tRunner.normalize(tRng.normal(size=(33, 20))),
        workspace=tRunner.workspace(),
        workspace_m=tRunner.workspace(0.02),
    )
    tRunner.mGripFakeOpen = True
    tOut["jaw_fake_open"] = tRunner._jawState(tState)
    tRunner.mGripMask = True
    tOut["jaw_mask"] = tRunner._jawState(tState)
    tItem = tRunner.item(tImgs, tState, (tImgs, tState + 1))
    tOut["item2_state"] = tItem["observation.state"]
    tRunner.mNObs = 1
    tOut["item1_state"] = tRunner.item(tImgs, tState)["observation.state"]
    checkGolden("core/policy_runner_pure", tOut)


def test_policy_runner_predict_and_rtc():
    tRng = _rng(7)
    tRunner = _policyRunner(tRng)
    tRunner.mPolicy = _FakeDiT()
    tImgs = {c: (tRng.random((216, 384, 3)) * 255).astype(np.uint8) for c in contract.CAMS}
    tItem = tRunner.item(tImgs, tRng.normal(size=30))
    tRtc = dict(Y=_chunk(tRng), W=chunk.rtcWeights(32, 3, 24), g0=np.array([1.0, 0.0]))
    tOut = {}
    tOut["plain"] = tRunner.predict(tItem, seed=3)
    tOut["guided"] = tRunner.predict(tItem, seed=3, rtc=dict(tRtc, mode="guided", beta=5.0))
    tOut["blend"] = tRunner.predict(tItem, seed=3, rtc=dict(tRtc, mode="blend"))
    tOut["guided_default"] = tRunner.predict(tItem, seed=4, rtc=tRtc)
    tRunner.mNSamples = 3
    tOut["k3"] = tRunner.predict(tItem, seed=3, rtc=dict(tRtc, mode="guided"))
    tRunner.mNSamples = 1
    tRunner.mPolicy.grip_head = True
    tRunner.mPolicy.grip_logits = lambda cond: torch.linspace(-2, 2, 33 * 2, dtype=torch.float64).reshape(1, 33, 2)
    tOut["grip_head"] = tRunner.predict(tItem, seed=3)
    tAct = _policyRunner(tRng, kind="act", n_obs=1)
    tAct.mPolicy = types.SimpleNamespace(
        predict_action_chunk=lambda b: torch.arange(32 * 20, dtype=torch.float64).reshape(1, 32, 20) / 640.0
    )
    tOut["act"] = tAct.predict(tAct.item(tImgs, tRng.normal(size=30)), seed=1)
    checkGolden("core/policy_runner_predict", tOut)


def test_geometry_helpers():
    tRng = _rng(8)
    tRa, tRb = _pose(tRng)[:3, :3], _pose(tRng)[:3, :3]
    tSegs = [
        (np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1.0, 0]), np.array([0.5, 1.0, 1.0])),
        (np.zeros(3), np.zeros(3), np.ones(3), np.ones(3)),
        (np.zeros(3), np.zeros(3), np.array([1.0, 0, 0]), np.array([1.0, 2, 0])),
        (np.zeros(3), np.array([0, 0, 2.0]), np.array([1.0, 1, 1]), np.array([1.0, 1, 1])),
        (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1, 0])),
        (np.zeros(3), np.array([1.0, 0, 0]), np.array([2.0, 1, 0]), np.array([3.0, 1, 0])),
        (np.zeros(3), np.array([1.0, 0, 0]), np.array([-2.0, 1, 0]), np.array([-1.0, 1, 0])),
    ]
    tK = {"fx": 600.0, "fy": 610.0, "cx": 424.0, "cy": 240.0}
    tArms, _ = fakes.fakeArms()
    tQ = {s: tRng.normal(size=6) * 0.3 for s in SIDES}
    tPlot = plot.cvPlot(
        [
            (np.arange(10.0), np.sin(np.arange(10.0)), (0, 0, 0), "a"),
            (np.arange(5.0), [0.1, np.nan, 0.3, 0.2, 2.0], (0, 0, 255), "b"),
        ],
        (-1.2, 1.2),
        "t",
        hline=0.5,
        xLabel="s",
    )
    checkGolden(
        "core/geometry",
        dict(
            rot_interp=[kinematics.rotInterp(tRa, tRb, s) for s in (0.0, 0.3, 1.0)],
            rot_angle=kinematics.rotAngleDeg(tRa, tRb),
            rot_angle_same=kinematics.rotAngleDeg(tRa, tRa),
            seg=[collision.segDist(*s) for s in tSegs],
            project=camera_model.project(_pose(tRng), tK, tRng.normal(size=(6, 3)) + [0, 0, 1.0]),
            project_behind=camera_model.project(np.eye(4), tK, np.array([[0.1, 0.0, 1.0], [0.0, 0.1, 0.01], [0.2, -0.1, 2.0]])),
            plot=sha256(tPlot.tobytes()),
            plot_shape=tPlot.shape,
            plot_small=sha256(plot.cvPlot([(np.arange(3.0), np.arange(3.0), (1, 2, 3), "x")], (0, 2), size=(300, 120)).tobytes()),
            fk={s: kinematics.fk(tArms, s, tQ[s]) for s in SIDES},
            link_gap=kinematics.linkGap(tArms, tQ),
        ),
    )


def test_make_arms(monkeypatch):
    import laundry_dataset

    monkeypatch.setattr(laundry_dataset, "Arm", fakes.FakeArm, raising=False)
    tArms, tArgs = kinematics.makeArms()
    tArms2, tArgs2 = kinematics.makeArms(tip_clear=-0.01, roll_thr=90.0)
    checkGolden(
        "core/make_arms",
        dict(
            args=vars(tArgs),
            args2=vars(tArgs2),
            sides=sorted(tArms),
            side_names=[tArms[s].mSide for s in SIDES],
            shared=tArms["left"].mArgs is tArms["right"].mArgs,
        ),
    )


def _chainModel():
    import pinocchio as pin

    tModel = pin.Model()
    tQidx, tJoff = {}, {}
    tAxes = [pin.JointModelRZ, pin.JointModelRY, pin.JointModelRY, pin.JointModelRY, pin.JointModelRZ, pin.JointModelRX]
    tOffsets = [(0, 0, 0.1), (0, 0, 0.1), (0.2, 0, 0), (0.2, 0, 0), (0.05, 0, 0), (0.05, 0, 0)]
    for tSide, tY in (("left", 0.25), ("right", -0.25)):
        tParent = 0
        tIdx = []
        for tJ, (tAxis, tOff) in enumerate(zip(tAxes, tOffsets)):
            tPlace = pin.SE3(np.eye(3), np.array(tOff, float) + (np.array([0.0, tY, 0.0]) if tJ == 0 else 0.0))
            tParent = tModel.addJoint(tParent, tAxis(), tPlace, f"{tSide}_joint{tJ + 1}")
            tIdx.append(tModel.joints[tParent].idx_q)
            tModel.addFrame(pin.Frame(f"{tSide}_link{tJ + 1}", tParent, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
        tRy = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        for tName, tPos, tRot in (
            ("gripper_link", (0.08, 0, 0), np.eye(3)),
            ("gripper_left", (0.06, 0.02, 0), np.eye(3)),
            ("gripper_right", (0.06, -0.02, 0), np.eye(3)),
            ("tcp", (0.10, 0, 0), tRy),
        ):
            tModel.addFrame(
                pin.Frame(f"{tSide}_{tName}", tParent, 0, pin.SE3(tRot, np.array(tPos, float)), pin.FrameType.OP_FRAME)
            )
        tQidx[tSide] = np.array(tIdx)
        tJoff[tSide] = np.linspace(-0.1, 0.1, 6) * (1 if tSide == "left" else -1)
    return tModel, tQidx, tJoff


def test_collision_capsules():
    pytest.importorskip("pinocchio")
    tModel, tQidx, tJoff = _chainModel()
    tPk = types.SimpleNamespace(full=tModel, q_full_pin=np.zeros(tModel.nq), qidx=tQidx, joff=tJoff)
    tArms = {s: types.SimpleNamespace(mPk=tPk) for s in SIDES}
    tCol = collision.Collision(tArms)
    tRng = _rng(9)
    tOut = {}
    for tCase in range(4):
        tQ = {s: tRng.normal(size=6) * 0.6 for s in SIDES}
        tCaps = tCol.capsules(tQ)
        tOut[f"caps{tCase}"] = {s: [[a, b, r, n] for a, b, r, n in tCaps[s]] for s in SIDES}
        tOut[f"check{tCase}"] = tCol.check(tQ)
        tOut[f"check{tCase}_args"] = tCol.check(tQ, tableZ=0.05, armClear=0.2, tableClear=-0.5)
    checkGolden("core/collision", tOut)
