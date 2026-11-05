"""Stand-alone deploy tools: height calibration (fit, plane, B-frame overlay, interactive / one-touch recording, xy shifts),
offline evaluation (metrics on a fake checkpoint + dataset), wrist domain gap measurements, camera-alignment similarity, the
X-VLA server's letterbox and RTC patch, return-to-start, N-step runs, and every command line's arguments and defaults."""

from __future__ import annotations
import builtins
import importlib
import json
import types

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")

from laundry_deploy.core import height_calib as heightCalib
from laundry_deploy.core.errors import DeployError
from laundry_deploy.tests import fakes
from laundry_deploy.tests.golden_record import checkGolden, normalise, sha256

SIDES = ("left", "right")
Q0 = {"left": np.array([0.1, -0.3, 0.4, 0.2, -0.1, 0.05]), "right": np.array([-0.1, -0.25, 0.35, -0.2, 0.1, -0.05])}


# ------------------------------------------------------------------------------------------------ height calibration
@pytest.fixture
def HC(tmp_path, monkeypatch):
    """The height-calibration tool with its files (core.height_calib) in tmp_path and a fixed clock."""
    from laundry_deploy.tools import height_calib as tModule

    monkeypatch.setattr(heightCalib, "CALIB_FILE", tmp_path / "height_calib.json")
    monkeypatch.setattr(heightCalib, "PENDING_FILE", tmp_path / "height_calib_pending.json")
    tAnchors = tmp_path / "anchors.json"
    tAnchors.write_text(json.dumps({"tags": {"400": {"p_B": [0.30, 0.22, 0.001]}, "401": {"p_B": [0.31, -0.21, 0.001]}}}))
    monkeypatch.setattr(heightCalib, "ANCHORS_FILE", tAnchors)
    monkeypatch.setattr(tModule, "time", fakes.FakeClock())
    return tModule


def _points():
    tRng = np.random.default_rng(20)
    tPts = []
    for tI in range(7):
        tSide = SIDES[tI % 2]
        tTarget = ("400", "401", "table")[tI % 3]
        tP = [0.2 + 0.05 * tI, (0.2 if tSide == "left" else -0.2) + 0.03 * (tI % 3), 0.001 + 0.004 * tRng.normal()]
        tExp = [0.3, 0.22, 0.001] if tTarget != "table" else [tP[0], tP[1], 0.001]
        tPts.append(dict(arm=tSide, target=tTarget, p_robot=tP, p_expected=tExp))
    return tPts


def test_height_calib_fit_and_plane(HC, capsys):
    tPts = _points()
    tFit = heightCalib.fitCalibration(tPts)
    tCollinear = heightCalib.fitCalibration(
        [dict(arm="left", target="table", p_robot=[0.1 * i, 0.2 * i, 0.01 * i], p_expected=[0, 0, 0]) for i in range(4)]
    )
    tTwo = heightCalib.fitCalibration(
        [
            dict(arm="right", target="401", p_robot=[0.3, -0.2, 0.004], p_expected=[0.31, -0.21, 0.001]),
            dict(arm="right", target="table", p_robot=[0.4, -0.1, 0.006], p_expected=[0.4, -0.1, 0.001]),
        ]
    )
    tCal = {"fit": tFit}
    tOut = dict(
        fit=tFit,
        collinear=tCollinear,
        two=tTwo,
        empty=heightCalib.fitCalibration([]),
        dz=[heightCalib.dzAt(tCal, s, xy) for s in SIDES for xy in ([0.3, 0.1], [0.5, -0.2])],
        dz_none=[heightCalib.dzAt(None, "left", [0, 0]), heightCalib.dzAt({"fit": {}}, "right", [1, 1])],
        to_B=heightCalib.toBaseFrame(
            dict(tCal, fit=dict(tFit, left=dict(tFit["left"], xy_cam_minus_robot=[0.003, -0.002]))),
            "left",
            [[0.3, 0.1, 0.05], [0.4, 0.2, 0.0]],
        ),
        to_B_none=heightCalib.toBaseFrame(None, "left", [0.3, 0.1, 0.05]),
        load_missing=heightCalib.loadCalibration(),
        TABLE_Z=heightCalib.TAG_TABLE_Z_M,
    )
    HC.show()
    heightCalib.CALIB_FILE.write_text(json.dumps(dict(created="c", points=tPts, fit=tFit)))
    tOut["load"] = heightCalib.loadCalibration()["created"]
    HC.show()
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("tools/height_calib_fit", tOut)


def _patchBridge(monkeypatch, q):
    import laundry_shared.visual_teleop.robot.bridge as vb
    import laundry_shared.visual_teleop.config as vcfg

    monkeypatch.setattr(
        vb,
        "HttpBridge",
        lambda url, live=False, timeout_s=0.5: fakes.FakeBridge(q, status={"status": "connected", "parked": ["left"]}),
    )
    monkeypatch.setattr(vcfg, "load", lambda: types.SimpleNamespace(robot={"base_url": "http://b"}))


def test_height_calib_recording(HC, tmp_path, monkeypatch, capsys):
    _patchBridge(monkeypatch, Q0)
    monkeypatch.setattr(HC, "makeArms", lambda **k: fakes.fakeArms(**k))
    tOut = {}
    HC.point("l", "400")
    HC.point("r", "t")
    HC.point("r", "p401")
    tOut["pending"] = json.loads(heightCalib.PENDING_FILE.read_text())
    heightCalib.CALIB_FILE.write_text(
        json.dumps(dict(created="c", points=tOut["pending"][:1], fit=heightCalib.fitCalibration(tOut["pending"])))
    )
    HC.anchorXyShift()
    tOut["anchorxy"] = json.loads(heightCalib.CALIB_FILE.read_text())["fit"]
    HC.finish()
    tOut["finish"] = json.loads(heightCalib.CALIB_FILE.read_text())
    tOut["finished_files"] = sorted(p.name for p in tmp_path.iterdir())
    HC.finish()
    tInputs = iter(["", "x", "l 400", "u", "u", "r t", "l 401", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(tInputs))
    HC.measure()
    tOut["measure"] = json.loads(heightCalib.CALIB_FILE.read_text())
    tOut["files"] = sorted(p.name for p in tmp_path.iterdir())
    tInputs2 = iter([])

    def eof(prompt=""):
        return next(tInputs2)

    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    HC.measure()
    tOut["stdout"] = capsys.readouterr().out
    checkGolden(
        "tools/height_calib_recording",
        json.loads(json.dumps(normalise(tOut), ensure_ascii=False).replace(str(tmp_path), "<TMP>")),
    )


# ------------------------------------------------------------------------------------------------ offline evaluation
@pytest.fixture
def offlineEval(monkeypatch):
    """laundry_training exports its datasets lazily and they need lerobot (not in the test environment): the contract names are
    stubbed in the package namespace for the duration of the test. setitem on the module dict, because monkeypatch.setattr reads
    the old value first and that read would trigger the lazy lerobot import."""
    import laundry_training

    monkeypatch.setitem(vars(laundry_training), "ActRelDataset", None)
    monkeypatch.setitem(vars(laundry_training), "DitRelDataset", None)
    monkeypatch.setitem(vars(laundry_training), "installGripBce", lambda md: None)
    return importlib.import_module("laundry_deploy.tools.offline_eval")


def _p9(rng, n, scale):
    from laundry_shared.contracts.action_contract import T_to_pose9

    tOut = []
    for _ in range(n):
        tT = np.eye(4)
        tT[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * scale).as_matrix()
        tT[:3, 3] = rng.normal(size=3) * scale
        tOut.append(T_to_pose9(tT))
    return np.array(tOut)


def test_offline_eval_metrics(offlineEval):
    tRng = np.random.default_rng(21)
    tP = np.c_[_p9(tRng, 32, 0.05), _p9(tRng, 32, 0.05), tRng.random((32, 2))]
    tG = np.c_[_p9(tRng, 32, 0.05), _p9(tRng, 32, 0.05), tRng.random((32, 2))]
    tB = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], bool)
    tGh = {8: [(True, True, False), (False, True, True), (True, False, False)], 16: [], 32: [(True, True, True)] * 3}
    checkGolden(
        "tools/offline_eval_metrics",
        dict(
            err_l=offlineEval.poseError(tP, tG, slice(0, 9)),
            err_r=offlineEval.poseError(tP, tG, slice(9, 18)),
            events=offlineEval.gripEvents(tB),
            events_none=offlineEval.gripEvents(np.zeros(5, bool)),
            match=offlineEval.matchEvents(
                [(3, "open"), (9, "open"), (14, "close")], [(2, "open"), (8, "close"), (30, "open")], 10.0
            ),
            match_tol=offlineEval.matchEvents([(3, "open")], [(2, "open")], 10.0, tolS=0.05),
            gh=offlineEval.gripHorizonReport(tGh),
            KS=offlineEval.KS,
        ),
    )


class _Hf:
    def __init__(self, cols):
        self.mCols = cols

    def with_format(self, fmt):
        return self

    def __getitem__(self, key):
        return self.mCols[key]


class _Ds:
    def __init__(self, repo, root=None, delta_timestamps=None, **kw):
        tRng = np.random.default_rng(22)
        self.mDts = delta_timestamps
        tEpi = np.repeat([0, 1, 2], [40, 30, 45])
        tN = len(tEpi)
        self.mN = tN
        tPose = np.c_[_p9(tRng, tN, 0.2), _p9(tRng, tN, 0.2)]
        self.hf_dataset = _Hf(
            {
                "episode_index": tEpi,
                "pose_abs": tPose,
                "pos_conf": (tRng.random((tN, 2)) > 0.3).astype(float),
                "grasp_win": (tRng.random((tN, 2)) > 0.6).astype(float),
            }
        )
        self.features = {"pose_abs", "pos_conf", "grasp_win", "action"}
        self.mHorizon = len(delta_timestamps["action"])
        self.mRng = tRng

    def __len__(self):
        return self.mN

    def __getitem__(self, i):
        tRng = np.random.default_rng(1000 + int(i))
        tA = np.c_[
            _p9(tRng, self.mHorizon, 0.03), _p9(tRng, self.mHorizon, 0.03), (tRng.random((self.mHorizon, 2)) > 0.5).astype(float)
        ]
        return {
            "action": torch.from_numpy(tA).float(),
            "observation.state": torch.from_numpy(tRng.normal(size=30)).float(),
            "task": "laundry folding",
            "action_is_pad": torch.zeros(self.mHorizon, dtype=torch.bool),
        }


def test_offline_eval_main(offlineEval, tmp_path, monkeypatch, capsys):
    tOut = {}
    for tKind in ("act", "multi_task_dit"):
        tCfg = types.SimpleNamespace(type=tKind, n_obs_steps=1 if tKind == "act" else 2, horizon=33, image_features=[])

        class Pol:
            def predict_action_chunk(self, b):
                tS = b["observation.state"].double()
                return (
                    torch.tanh(tS[..., :20].reshape(-1, 1, 20) * torch.linspace(0.1, 1.0, 32, dtype=torch.float64).view(1, 32, 1))
                    * 0.05
                ).float()

            def _prepare_batch(self, b):
                return b

            observation_encoder = types.SimpleNamespace(encode=lambda b: b["observation.state"].double()[..., :4].reshape(1, -1))
            noise_predictor = None
            objective = types.SimpleNamespace(
                conditional_sample=lambda net, k, cond: (
                    torch.ones((1, 33, 20), dtype=torch.float64) * cond.sum() * 0.001
                    + torch.linspace(0, 0.05, 33, dtype=torch.float64).view(1, 33, 1)
                )
            )

            def forward(self, b):
                return b["action"].double().abs().mean(), {}

        import laundry_training

        monkeypatch.setattr(offlineEval, "loadPolicy", lambda ckpt, device, _c=tCfg: (_c, Pol(), lambda b: b, lambda a: a))
        monkeypatch.setitem(vars(laundry_training), "ActRelDataset", _Ds)
        monkeypatch.setitem(vars(laundry_training), "DitRelDataset", _Ds)
        tData = tmp_path / f"data_{tKind}"
        tData.mkdir()
        (tData / "episode_table.csv").write_text("episode_key\nA\nB\n")
        tRep = tmp_path / f"rep_{tKind}"
        offlineEval.main(
            [
                "--ckpt",
                str(tmp_path / "ck"),
                "--data",
                str(tData),
                "--out",
                str(tRep),
                "--stride",
                "3",
                "--val-loss-batches",
                "2",
                "--device",
                "cpu",
            ]
        )
        tJ = json.loads((tRep / "offline_eval.json").read_text())
        tJ["ckpt"] = "<CK>"
        tJ["data"] = "<DATA>"
        tOut[tKind] = tJ
    capsys.readouterr()
    checkGolden("tools/offline_eval_main", tOut)


# ------------------------------------------------------------------------------------------------ wrist domain gap
def _wristImage(rng, jawW=300, shift=0):
    tImg = np.clip(rng.normal(170, 20, size=(720, 1280, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(tImg, (500 + shift, 300), (800 + shift, 600), (90, 140, 200), -1)  # beige cloth
    cv2.rectangle(
        tImg, (640 - jawW // 2 + shift, 450), (640 + jawW // 2 + shift, 719), (20, 20, 20), -1
    )  # dark jaw from the bottom edge
    return tImg


def test_domain_gap_measures(tmp_path, monkeypatch):
    from laundry_deploy.tools import domain_gap as DG

    tRng = np.random.default_rng(23)
    tImgs = [_wristImage(tRng), _wristImage(tRng, 250, 20), _wristImage(tRng, 340, -15)]
    tMs = [DG.measureFrame(x) for x in tImgs]
    tYaml = tmp_path / "fisheye_left_v002.yaml"
    tYaml.write_text(
        "K: [[700.0, 0.0, 960.0], [0.0, 700.0, 540.0], [0.0, 0.0, 1.0]]\nD: [0.05, -0.01, 0.002, -0.0003]\nimage_size: [1920, 1080]\n"
    )
    monkeypatch.setattr(DG, "FISHEYE", {"left": tYaml, "right": tYaml})
    checkGolden(
        "tools/domain_gap",
        dict(
            measure=[m[0] for m in tMs],
            jaw=[sha256(m[1].tobytes()) for m in tMs],
            cloth=[sha256(m[2].tobytes()) for m in tMs],
            agg=DG.aggregate([m[0] for m in tMs]),
            agg_one=DG.aggregate([{"grey": tMs[0][0]["grey"], "bgr_mean": [1, 2, 3]}]),
            overlay=sha256(DG.overlay(tImgs[0], tMs[0][1], tMs[0][2], "UMI LEFT").tobytes()),
            blank=DG.measureFrame(np.full((720, 1280, 3), 200, np.uint8))[0],
            fov=DG.fisheyeFov("left"),
        ),
    )


def test_domain_gap_umi_frames(tmp_path, monkeypatch):
    from laundry_deploy.tools import domain_gap as DG

    tLdt = tmp_path / "train"
    tMeta = tLdt / "data/lerobot/tagrig_laundry_v1_fused_train"
    tMeta.mkdir(parents=True)
    tTakes = tmp_path / "takes"
    tAssign = {}
    for tTake in ("20261026_100000", "20261027_110000"):
        tTd = tTakes / tTake
        tAssign[tTake] = {"cam_of_arm": {"left": "wrist_left", "right": "wrist"}}
        tT0 = 10_000_000_000
        (tTd).mkdir(parents=True)
        (tTd / "meta.json").write_text(
            json.dumps(
                {
                    "t_start_monotonic_ns": tT0,
                    "protocol": [
                        {"text": "자유: 시작", "t0": 1.0, "dur": 4.0},
                        {"text": "접기", "t0": 6.0, "dur": 2.0},
                        {"text": "자유: 끝", "t0": 9.0, "dur": 3.0},
                    ],
                }
            )
        )
        for tCam in ("wrist_left", "wrist"):
            (tTd / tCam).mkdir()
            tTs = tT0 + (np.arange(0, 14, 1 / 15) * 1e9).astype(np.int64)
            np.save(tTd / tCam / "t_ns.npy", tTs)
            for tI in range(0, len(tTs), 2):
                (tTd / tCam / f"wrist_{tI:05d}.jpg").write_bytes(b"")
    (tMeta / "source_meta.json").write_text(json.dumps({"assignments": tAssign}))
    monkeypatch.setattr(DG, "TRAIN_ROOT", tLdt)
    monkeypatch.setattr(DG, "TAG_RIG", tTakes)
    tOut = {}
    for tName, tN, tStart in (("start3", 3, True), ("whole2", 2, False), ("start1", 1, True)):
        tF = DG.umiFrames(tN, startOnly=tStart)
        tOut[tName] = {s: [str(p.relative_to(tTakes)) for p in tF[s]] for s in SIDES}
    checkGolden("tools/domain_gap_umi_frames", tOut)


# ------------------------------------------------------------------------------------------------ camera alignment
def _scene(rng):
    tImg = np.full((480, 848, 3), 120, np.uint8)
    for _ in range(140):
        tC = tuple(int(x) for x in rng.integers(0, 255, 3))
        tP = (int(rng.integers(0, 848)), int(rng.integers(0, 300)))
        cv2.circle(tImg, tP, int(rng.integers(3, 18)), tC, -1)
        cv2.rectangle(
            tImg, tP, (tP[0] + int(rng.integers(5, 40)), tP[1] + int(rng.integers(5, 30))), tuple(255 - c for c in tC), 2
        )
    return tImg


def test_cam_align(tmp_path, monkeypatch):
    from laundry_deploy.tools import cam_align as CA

    tRng = np.random.default_rng(24)
    tRef = _scene(tRng)
    tOut = {}
    for tName, (tDx, tDy, tRot, tScale) in (
        ("same", (0, 0, 0, 1)),
        ("shift", (6, -4, 0, 1)),
        ("rot", (0, 0, 1.0, 1)),
        ("zoom", (2, 1, -0.5, 1.02)),
    ):
        tM = cv2.getRotationMatrix2D((424, 240), tRot, tScale)
        tM[:, 2] += [tDx, tDy]
        tCur = cv2.warpAffine(tRef, tM, (848, 480), borderMode=cv2.BORDER_REPLICATE)
        tS = CA.similarity(tRef, tCur)
        tOut[tName] = None if tS is None else {k: (v if k != "H" else np.round(v, 6)) for k, v in tS.items()}
        tOut[f"advice_{tName}"] = CA.advice(tS)
    tOut["blank"] = CA.similarity(tRef, np.zeros_like(tRef))
    tOut["advice_none"] = CA.advice(None)
    tOut["advice_cases"] = [
        CA.advice(dict(dx=dx, dy=dy, rot=r, scale=s))
        for dx, dy, r, s in ((2.0, -2.0, 0.2, 0.99), (-1.0, 1.0, -0.2, 1.01), (1.5, 1.5, 0.15, 1.006), (0, 0, 0, 1))
    ]
    tTakes = tmp_path / "takes"
    for tTake in ("20261030_100000", "20261030_110000", "20261103_120000", "20261103_131000"):
        (tTakes / tTake).mkdir(parents=True)
        for tK in (0, 60):
            cv2.imwrite(str(tTakes / tTake / f"color_{tK:05d}.jpg"), cv2.resize(_scene(tRng), (1280, 720)))
    cv2.imwrite(str(tTakes / "20261030_110000" / "color_00060.jpg"), np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(CA, "TAKES", tTakes)
    tRefImg, tN = CA.makeRef("20261030")
    tOut["make_ref"] = [sha256(tRefImg.tobytes()), tN, CA.makeRef("20261103")[1]]
    try:
        CA.makeRef("20990101")
    except CA.CamAlignError as e:
        tOut["make_ref_missing"] = str(e)
    tState = CA.AlignState(CA.LiveGrabber(CA.SRC), "20261030")
    tState.mRefs["20261030"] = (tRefImg, tN)
    tState.mLive.mJpg = cv2.imencode(".jpg", cv2.resize(tRef, (1280, 720)))[1].tobytes()
    for tMode in ("blend", "edges", "ref", "cur"):
        tOut[f"view_{tMode}"] = sha256(tState.view(tMode).tobytes())
    tState.mLive.mJpg = None
    tOut["view_nolive"] = sha256(tState.view("blend").tobytes())
    tOut["consts"] = dict(
        REFS=CA.REFS, REF_BEFORE=CA.REF_BEFORE, VIEW_WH=CA.VIEW_WH, MASK_Y=CA.MASK_Y, page=sha256(CA.PAGE.encode())
    )
    checkGolden("tools/cam_align", tOut)


# ------------------------------------------------------------------------------------------------ X-VLA server pieces
def test_xvla_server_letterbox_and_rtc():
    from laundry_deploy import xvla_server as XS

    tRng = np.random.default_rng(25)
    tImg = (tRng.random((216, 384, 3)) * 255).astype(np.uint8)
    tOut = dict(
        letterbox=sha256(XS.letterbox(tImg).tobytes()),
        passthrough=XS.letterbox(np.ones((240, 320, 3), np.uint8)).sum(),
        consts=dict(TASK=XS.TASK, GRIP=XS.GRIP, CAMS=XS.CAMS, RENAME=XS.RENAME),
    )

    class Model:
        chunk_size = 4
        dim_action = 3

        def eval(self):
            pass

        def _get_target_dtype(self):
            return torch.float64

        def forward_vlm(self, input_ids, image_input, image_mask):
            return {"enc": image_input.sum()}

        def transformer(self, domain_id, action_with_noise, proprio, t, enc):
            return torch.tanh(action_with_noise * 0.5 + proprio.view(-1, 1, 1) * 0.1 + t.view(-1, 1, 1) + enc * 1e-3)

        action_space = types.SimpleNamespace(preprocess=lambda p, x: (p * 2.0, x + 0.1), postprocess=lambda a: a * 3.0)

    XS.installRtc(Model)
    XS.installRtc(Model)
    tM = Model()
    tArgs = (
        torch.zeros(1, 5),
        torch.ones(1, 2, dtype=torch.float32),
        None,
        torch.zeros(1),
        torch.tensor([0.3], dtype=torch.float32),
    )
    tX1 = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(1, 4, 3)
    tOut["plain"] = tM.generate_actions(*tArgs, 5, x1=tX1)
    tM.mRtc = (
        torch.full((1, 4, 3), 0.2, dtype=torch.float64),
        torch.tensor([1.0, 0.5, 0.1, 0.0], dtype=torch.float64).view(1, -1, 1),
    )
    tOut["rtc"] = tM.generate_actions(*tArgs, 3, x1=tX1)
    tOut["zero_steps"] = tM.generate_actions(*tArgs, 0, x1=tX1)
    tOut["patched"] = getattr(Model, "mRtcPatched", False)
    checkGolden("tools/xvla_server", tOut)


# ------------------------------------------------------------------------------------------------ return to start, N steps
def test_return_to_start(tmp_path, monkeypatch, capsys):
    from laundry_deploy.tools import return_to_start as RS
    import laundry_shared.visual_teleop.robot.bridge as vb
    import laundry_shared.visual_teleop.tools.robot_moves as rm

    monkeypatch.setattr(RS, "START_FILE", tmp_path / "start_pose.json")
    (tmp_path / "start_pose.json").write_text(json.dumps({"q": {s: (Q0[s] + 0.05).tolist() for s in SIDES}}))
    monkeypatch.setattr(RS, "makeArms", lambda **k: fakes.fakeArms(**k))
    monkeypatch.setattr(RS, "Collision", lambda arms: fakes.FakeCollision())
    monkeypatch.setattr(rm, "unpark", lambda br, s: br.mLog.append(("unpark", s)))
    monkeypatch.setattr(RS, "time", fakes.FakeClock(), raising=False)
    import time as _time

    tClock = fakes.FakeClock()
    monkeypatch.setattr(_time, "sleep", tClock.sleep)
    monkeypatch.setattr(_time, "monotonic", tClock.monotonic)
    tOut = {}
    for tName, tArgv, tQ in (
        ("dry", [], Q0),
        ("go", ["--go", "--deg-s", "20"], Q0),
        ("far", [], {s: Q0[s] + 1.0 for s in SIDES}),
        ("low", ["--min-z", "0.5"], Q0),
    ):
        tBr = fakes.FakeBridge(tQ, g={"left": 0.0, "right": 1.0}, status={"estop": tName == "go"})
        tInits = []
        monkeypatch.setattr(
            vb, "HttpBridge", lambda url, live=False, timeout_s=0.5, _b=tBr: (tInits.append([url, live, timeout_s]), _b)[1]
        )
        try:
            RS.returnToStart(RS.buildParser().parse_args(tArgv))
            tRes = "ok"
        except DeployError as e:
            tRes = str(e)
        tOut[tName] = dict(res=tRes, init=tInits, log=tBr.mLog, n=len(tBr.mSends), sends=tBr.mSends[:1] + tBr.mSends[-1:])
    tOut["stdout"] = capsys.readouterr().out
    checkGolden("tools/return_to_start", tOut)


def test_run_n_steps_ctl(monkeypatch, capsys):
    from laundry_deploy.tools import run_n_steps as RN

    tClock = fakes.FakeClock(10.0)
    monkeypatch.setattr(RN, "time", tClock)
    tC = RN.StepControl(2, 1.5)
    tC.mTLast = tClock.monotonic() + 2.0 - 1.5
    tOut = []
    for _ in range(14):
        tOut.append([tC.poll(), tC.mStepN, tC.takeStep(), tC.mStepsDone])
        tClock.advance(0.6)
    tC.publish({"t": 1})
    tOut.append([tC.mSnap, tC.mStepMode])
    tOut.append(capsys.readouterr().out)
    checkGolden("tools/run_n_steps_ctl", tOut)


# ------------------------------------------------------------------------------------------------ command lines
def test_cli_defaults(offlineEval):
    tOut = {}
    for tSub, tArgv in (
        ("cams", ["cams"]),
        ("wave", ["wave"]),
        ("start", ["start"]),
        ("run", ["run"]),
        (
            "run_all",
            ["run", "--policy", "dit", "--go", "--commit", "4", "--rtc", "guided", "--grip-mode", "relu", "--grip-at", "max"],
        ),
        ("wave_confirm", ["wave", "--confirm", "left=0x1", "right=0x2"]),
    ):
        tOut[f"deploy_{tSub}"] = fakes.captureParser("laundry_deploy.session.cli", tArgv)
    for tBad in (
        ["run", "--policy", "x"],
        ["run", "--rtc", "on"],
        ["run", "--grip-mode", "linear"],
        ["start", "--episode", "a"],
        [],
    ):
        try:
            fakes.captureParser("laundry_deploy.session.cli", tBad)
            tOut[" ".join(tBad) or "none"] = "parsed"
        except SystemExit as e:
            tOut[" ".join(tBad) or "none"] = f"exit {e.code}"
    tOut["ui"] = fakes.captureParser("laundry_deploy.ui.server", [])
    tOut["ui_set"] = fakes.captureParser("laundry_deploy.ui.server", ["--port", "9000", "--bridge-url", "http://x", "--dry-only"])
    tOut["offline_eval"] = fakes.captureParser(
        "laundry_deploy.tools.offline_eval", ["--ckpt", "c", "--data", "d", "--out", "o", "--device", "cpu"]
    )
    for tCmd in (["show"], ["point", "l", "400"], ["measure"], ["finish"], ["camxy"], ["anchorxy"]):
        tOut[f"height_calib_{tCmd[0]}"] = fakes.captureParser("laundry_deploy.tools.height_calib", tCmd)
    tOut["cam_viewer"] = fakes.captureParser("laundry_deploy.tools.cam_viewer", [])
    tOut["return_to_start"] = fakes.captureParser("laundry_deploy.tools.return_to_start", [])
    tOut["run_n_steps"] = fakes.captureParser("laundry_deploy.tools.run_n_steps", ["--ckpt", "c"])
    tOut["domain_gap"] = fakes.captureParser("laundry_deploy.tools.domain_gap", [])
    tOut["xvla_server"] = fakes.captureParser("laundry_deploy.xvla_server", [])
    tOut["xvla_server"].device = "<device>"
    checkGolden("tools/cli_defaults", tOut)
