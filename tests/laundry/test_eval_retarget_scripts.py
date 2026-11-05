"""Characterization of scripts/laundry/eval/*.py and scripts/laundry/retarget/*.py on synthetic takes / replay runs: the parity
scorer's stroke / rotation / free-motion / gripper metrics and printed tables, the side-by-side video, the replay-plan regression
metrics, the robotization of a take, and every command line's argparse defaults."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from laundry_common.paths import REPO_ROOT
from laundry_reference.tests.snapshot import checkSnapshot as _checkSnapshot
from laundry_reference.tests.snapshot import helpExitCode, parsedArgs

EXPECTED = Path(__file__).with_name("expected_scripts_characterization.json")
SCRIPTS = REPO_ROOT / "scripts" / "laundry"
PARITY = SCRIPTS / "eval" / "replay_parity.py"
REGRESS = SCRIPTS / "eval" / "regress_replay_plan.py"
SIDE_BY_SIDE = SCRIPTS / "eval" / "replay_side_by_side.py"
ALIGN = SCRIPTS / "retarget" / "umi_align.py"
ROBOTIZE = SCRIPTS / "retarget" / "umi_robotize_bi.py"
NS = 1_000_000_000
T0_NS = 10 * NS
UMI_HZ = 30.0
ROBOT_HZ = 20.0
SPEED = 0.5
ROBOT_T0_NS = 500 * NS
OBS_DELAY_NS = 60_000_000


def checkSnapshot(name, value, **kw):
    _checkSnapshot(name, value, expectedPath=EXPECTED, **kw)


def _load(path: Path, name: str):
    tSpec = importlib.util.spec_from_file_location(name, path)
    tModule = importlib.util.module_from_spec(tSpec)
    tSpec.loader.exec_module(tModule)
    return tModule


@pytest.fixture(scope="module")
def parity():
    return _load(PARITY, "replay_parity_under_test")


def _quiet(fn, *args, **kw):
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        tResult = fn(*args, **kw)
    return tResult, tOut.getvalue()


def _minJerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _segment(axis, sign, rep, startS, hand="오른손"):
    """hold_pre 0-1 s, stroke 1-3 s, hold_mid 3-4 s, back 4-6 s, hold_post 6-7 s after `startS`."""
    tA = T0_NS + int(startS * NS)
    return {"hand": hand, "axis": axis, "dir": "+" if sign > 0 else "-", "sign": sign, "rep": rep,
            "hold_pre": [tA, tA + NS], "hold_mid": [tA + 3 * NS, tA + 4 * NS], "back": [tA + 4 * NS, tA + 6 * NS],
            "hold_post": [tA + 6 * NS, tA + 7 * NS]}


def _umiTraj(segments, rng, durationS):
    tT = (T0_NS + np.arange(0, durationS, 1 / UMI_HZ) * NS).astype(np.int64)
    tP = np.tile(np.array([0.35, -0.2, 0.1]), (len(tT), 1))
    tRotvec = np.zeros((len(tT), 3))
    tAxes = {"X": 0, "Y": 1, "Z": 2, "YAW": 2, "PITCH": 1, "ROLL": 0}
    for tSeg in segments:
        tS = (tT - tSeg["hold_pre"][1]) / (2 * NS)
        tBack = (tT - tSeg["back"][0]) / (2 * NS)
        tProfile = _minJerk(tS) - _minJerk(tBack)
        if tSeg["axis"] in ("X", "Y", "Z"):
            tP[:, tAxes[tSeg["axis"]]] += tSeg["sign"] * 0.15 * tProfile
            tP[:, 2] += 0.01 * np.sin(np.pi * np.clip(tS, 0, 1))
        elif tSeg["axis"] in ("YAW", "PITCH", "ROLL"):
            tRotvec[:, tAxes[tSeg["axis"]]] += tSeg["sign"] * np.radians(30.0) * tProfile
        else:
            tP += np.outer(tProfile, [0.10, 0.05, 0.08])
            tRotvec[:, 2] += np.radians(20.0) * tProfile
    tP += rng.normal(0, 0.001, tP.shape)
    tR = (Rotation.from_rotvec(tRotvec) * Rotation.from_euler("y", 90, degrees=True)).as_matrix()
    return {"t_ns": tT, "p_B": tP, "R_B_tcp": tR, "ntags": (np.arange(len(tT)) % 13 != 0).astype(np.int64)}


def _replaySegment(traj, seg, rng, gripper=False):
    """Commanded = the UMI pose resampled at ROBOT_HZ; observed = commanded 98 % amplitude, OBS_DELAY_NS later, noisy."""
    tA = traj["t_ns"]
    tSrc = np.arange(seg["hold_pre"][0], seg["hold_post"][1], NS / ROBOT_HZ).astype(np.int64)
    tCmd = (ROBOT_T0_NS + (tSrc - tSrc[0]) / SPEED).astype(np.int64)
    tObs = tCmd + OBS_DELAY_NS
    tPCmd = np.stack([np.interp(tSrc, tA, traj["p_B"][:, k]) for k in range(3)], 1)
    tRv = Rotation.from_matrix(traj["R_B_tcp"]).as_rotvec()
    tRotCmd = np.stack([np.interp(tSrc, tA, tRv[:, k]) for k in range(3)], 1)
    tPObs = tPCmd[0] + 0.98 * (tPCmd - tPCmd[0]) + rng.normal(0, 0.0005, tPCmd.shape)
    tRotObs = tRotCmd + rng.normal(0, 0.002, tRotCmd.shape)
    tQ = np.radians(np.cumsum(rng.normal(0, 0.3, (len(tSrc), 6)), 0))
    tOut = {"src_t_ns": tSrc, "t_cmd_ns": tCmd, "t_obs_ns": tObs, "p_cmd": tPCmd, "p_obs": tPObs, "rot_cmd": tRotCmd,
            "rot_obs": tRotObs, "q_cmd": tQ}
    if gripper:
        tPhase = (tSrc - tSrc[0]) / (tSrc[-1] - tSrc[0])
        tG = ((tPhase > 0.3) & (tPhase < 0.7)).astype(float)
        tOut["g_cmd"] = tG
        tOut["g_obs"] = np.convolve(np.pad(tG, 3, mode="edge"), np.ones(7) / 7, "valid")
    return tOut


@pytest.fixture(scope="module")
def replayData(tmp_path_factory):
    tRoot = tmp_path_factory.mktemp("parity")
    tRng = np.random.default_rng(11)
    tTake = tRoot / "20261020_120000"
    tTake.mkdir()
    tSegs = [_segment("X", 1, 1, 0.0), _segment("X", 1, 2, 8.0), _segment("X", 1, 3, 16.0), _segment("Y", -1, 1, 24.0),
             _segment("YAW", 1, 1, 32.0), _segment("YAW", 1, 2, 40.0), _segment("Z", 1, 1, 48.0, hand="왼손")]
    tTraj = _umiTraj(tSegs, tRng, 56.0)
    np.savez(tTake / "umi_traj_right.npz", **tTraj)
    (tTake / "segments.json").write_text(json.dumps(tSegs, ensure_ascii=False))
    tRun = tRoot / "run_abc"
    tRun.mkdir()
    (tRun / "meta.json").write_text(json.dumps({"speed": SPEED, "segments": tSegs[:6], "traj": ""}, ensure_ascii=False))
    for tJ, tSeg in enumerate(tSegs[:6]):
        np.savez(tRun / f"seg_{tJ:02d}.npz", **_replaySegment(tTraj, tSeg, tRng))
    # free motion take + run with the gripper
    tFreeTake = tRoot / "20261020_130000"
    tFreeTake.mkdir()
    tFreeSegs = [dict(_segment("FREE", 1, 1, 0.0), axis="FREE"), dict(_segment("FREE", 1, 2, 8.0), axis="FREE")]
    tFreeTraj = _umiTraj(tFreeSegs, tRng, 16.0)
    np.savez(tFreeTake / "umi_traj_right.npz", **tFreeTraj)
    (tFreeTake / "segments.json").write_text(json.dumps(tFreeSegs, ensure_ascii=False))
    tJaw = tFreeTake / "wrist"
    tJaw.mkdir()
    tEv = []
    for tSeg in tFreeSegs:
        tA, tB = (tSeg["hold_pre"][0] + tSeg["hold_pre"][1]) // 2, (tSeg["hold_post"][0] + tSeg["hold_post"][1]) // 2
        tEv += [(tA + int(0.31 * (tB - tA)), "open"), (tA + int(0.69 * (tB - tA)), "close"), (tA + int(0.9 * (tB - tA)), "open")]
    np.savez(tJaw / "umi_jaw.npz", ev_t=np.array([e[0] for e in tEv], np.int64), ev_kind=np.array([e[1] for e in tEv]))
    tFreeRun = tRoot / "run_free"
    tFreeRun.mkdir()
    (tFreeRun / "meta.json").write_text(json.dumps({"speed": SPEED, "segments": tFreeSegs, "gripper": str(tJaw)}, ensure_ascii=False))
    for tJ, tSeg in enumerate(tFreeSegs):
        np.savez(tFreeRun / f"seg_{tJ:02d}.npz", **_replaySegment(tFreeTraj, tSeg, tRng, gripper=True))
    # data-axes mode: camera-frame positions only (no robot calibration)
    tCamTake = tRoot / "20261020_140000"
    tCamTake.mkdir()
    tCamSegs = [_segment(tAxis, 1, tRep, 8.0 * (3 * tI + tRep - 1)) for tI, tAxis in enumerate("XYZ") for tRep in (1, 2)]
    tCamTraj = _umiTraj(tCamSegs, tRng, 8.0 * 6 + 1)
    tTilt = Rotation.from_euler("xyz", [10, -5, 30], degrees=True).as_matrix()
    np.savez(tCamTake / "umi_traj_right.npz", t_ns=tCamTraj["t_ns"], p_C=tCamTraj["p_B"] @ tTilt.T)
    (tCamTake / "segments.json").write_text(json.dumps(tCamSegs, ensure_ascii=False))
    return {"root": tRoot, "take": tTake, "run": tRun, "traj": tTraj, "segs": tSegs, "free_take": tFreeTake,
            "free_run": tFreeRun, "free_traj": tFreeTraj, "free_segs": tFreeSegs, "jaw": tJaw, "cam_take": tCamTake}


def test_parity_stroke_helpers(parity):
    tRng = np.random.default_rng(2)
    tS = np.linspace(0, 1, 50)
    tA = np.outer(_minJerk(tS), [0.15, 0.0, 0.0]) + tRng.normal(0, 0.001, (50, 3))
    tB = tA * 1.01
    tC = tA * 0.95 + np.outer(np.sin(np.pi * tS), [0, 0, 0.01])
    tStroke = parity.stroke(tA, np.array([1.0, 0, 0]), tB, tC)
    checkSnapshot("parity.helpers", {
        "on_progress": parity.onProgress(tA, np.array([1.0, 0, 0]))[::10],
        "on_progress_flat": parity.onProgress(np.zeros((4, 3)), np.array([1.0, 0, 0])),
        "perp": parity.perp(np.array([[1.0, 2.0, 3.0]]), np.array([0, 1.0, 0])),
        "stroke": tStroke, "judge": parity.judge(tStroke),
        "judge_rot": parity.judge(dict(tStroke, geo_med_deg=6.0, fake_mm=12.0, sat_deg_s=10.0), rot=True),
        "windows": parity.windows(_segment("X", 1, 1, 0.0)),
        "sel": parity.selectWindow(np.arange(10), np.arange(30.0).reshape(10, 3), 2, 7, (0, 3)),
        "sel_short": parity.selectWindow(np.arange(10), np.arange(30.0).reshape(10, 3), 2, 4),
        "judge_free": parity.judgeFree({"pos_rmse": 25.0, "pos_p95": 10.0, "rot_med": 1.0, "rot_p95": 12.0, "tr_ratio": 0.8,
                                         "rot_ratio": 1.3, "amp_A_deg": 15.0, "jumps": 1}),
    })


def test_parity_synthetic_selftest(parity):
    tOk, tStdout = _quiet(parity.synthetic)
    checkSnapshot("parity.synthetic", {"ok": tOk, "stdout": tStdout})


def test_parity_rot_free_grip(parity, replayData, tmp_path):
    tTraj, tSegs = replayData["traj"], replayData["segs"]
    tRot = np.load(replayData["run"] / "seg_04.npz")
    tW0, tW1 = parity.windows(tSegs[4])
    tRotResult, tRotStdout = _quiet(parity.rotStroke, tTraj, tRot, tSegs[4], SPEED, tW0, tW1)
    tNoRot, tNoRotStdout = _quiet(parity.rotStroke, {"t_ns": tTraj["t_ns"]}, tRot, tSegs[4], SPEED, tW0, tW1)
    tFree = np.load(replayData["free_run"] / "seg_00.npz")
    tFreeResult = parity.freeStroke(replayData["free_traj"], tFree, replayData["free_segs"][0], SPEED)
    tGrip = parity.gripEvents(tFree, replayData["free_segs"][0], SPEED, replayData["jaw"])
    checkSnapshot("parity.rot_free_grip", {"rot": tRotResult, "rot_stdout": tRotStdout, "no_rot": tNoRot, "no_rot_stdout": tNoRotStdout,
                                           "free": tFreeResult, "grip": tGrip})


def test_parity_run_files_modes(parity, replayData):
    tData = replayData
    tModes = {
        "umi_only": Namespace(take=str(tData["take"]), run=None, hand="오른손", umi_only=True, synthetic=False),
        "umi_only_cam_axes": Namespace(take=str(tData["cam_take"]), run=None, hand="오른손", umi_only=True, synthetic=False),
        "abc": Namespace(take=str(tData["take"]), run=str(tData["run"]), hand="오른손", umi_only=False, synthetic=False),
        "free": Namespace(take=str(tData["free_take"]), run=str(tData["free_run"]), hand="오른손", umi_only=False, synthetic=False),
    }
    tResult = {}
    for tName, tArgs in tModes.items():
        tOk, tStdout = _quiet(parity.runFiles, tArgs)
        tResult[tName] = {"ok": tOk, "stdout": tStdout.replace(str(tData["root"]), "<tmp>")}
    tResult["free_plots"] = sorted(tPath.name for tPath in tData["free_run"].glob("free_*.png"))
    checkSnapshot("parity.run_files", tResult)


def test_parity_cli(monkeypatch):
    checkSnapshot("parity.cli", parsedArgs(monkeypatch, str(PARITY), ["take", "run"], isPath=True))
    assert helpExitCode(monkeypatch, str(PARITY), [], isPath=True) == 0


def test_regress_metrics_and_cli(tmp_path, monkeypatch):
    tRegress = _load(REGRESS, "regress_replay_plan_under_test")
    tRng = np.random.default_rng(4)
    tDump = tmp_path / "plan.npz"
    np.savez(tDump, hz=np.float64(20.0), right_Q=np.cumsum(tRng.normal(0, 0.01, (40, 6)), 0), right_pe=np.abs(tRng.normal(0, 0.001, 40)),
             right_re=np.abs(tRng.normal(0, 0.3, 40)), right_linkz=0.05 + tRng.normal(0, 0.01, 40), gaps=0.2 + tRng.normal(0, 0.02, 40))
    checkSnapshot("regress.metrics", {"metrics": tRegress.metrics(tDump), "cmd": tRegress.CMD, "tol": tRegress.TOL,
                                      "base": str(tRegress.BASE.relative_to(REPO_ROOT)), "base_exists": tRegress.BASE.is_file()})
    checkSnapshot("regress.cli", parsedArgs(monkeypatch, str(REGRESS), [], isPath=True))
    assert helpExitCode(monkeypatch, str(REGRESS), [], isPath=True) == 0


def test_side_by_side_video(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    tModule = _load(SIDE_BY_SIDE, "replay_side_by_side_under_test")
    tHuman, tReplay, tRobot = tmp_path / "take", tmp_path / "replay", tmp_path / "robot_rec"
    for tDir in (tHuman, tReplay / "right", tReplay / "left", tRobot):
        tDir.mkdir(parents=True)
    tTh = (T0_NS + np.arange(40) * NS / 15).astype(np.int64)
    np.save(tHuman / "t_ns.npy", tTh)
    tSrc = tTh[2:36:2]
    tCmd = (ROBOT_T0_NS + (tSrc - tSrc[0]) / SPEED).astype(np.int64)
    for tSide in ("right", "left"):
        np.savez(tReplay / tSide / "seg_00.npz", src_t_ns=tSrc, t_cmd_ns=tCmd, g_cmd=(np.arange(len(tSrc)) % 6 < 3).astype(float))
    tTv = (ROBOT_T0_NS - NS // 10 + np.arange(60) * NS / 15).astype(np.int64)
    np.save(tRobot / "t_ns.npy", tTv)
    for tK in range(40):
        cv2.imwrite(str(tHuman / f"color_{tK:05d}.jpg"), np.full((90, 160, 3), 4 * tK, np.uint8))
    for tK in range(60):
        if tK != 30:
            cv2.imwrite(str(tRobot / f"color_{tK:05d}.jpg"), np.full((90, 160, 3), 255 - 3 * tK, np.uint8))
    tOutFile = tmp_path / "sbs.mp4"
    tResult = {"nearest": [tModule.nearest(np.array([0.0, 10.0, 20.0]), tQ) for tQ in (-5.0, 4.0, 6.0, 15.0, 25.0)]}
    for tName, tExtra in (("human_time", []), ("robot_time", ["--robot-time", "--slow", "2"])):
        monkeypatch.setattr(sys, "argv", ["sbs", str(tHuman), str(tReplay), str(tRobot), "--out", str(tOutFile), "--fps", "10",
                                          "--w", "160", *tExtra])
        _, tStdout = _quiet(tModule.main)
        tCap = cv2.VideoCapture(str(tOutFile))
        tResult[tName] = {"stdout": tStdout.replace(str(tmp_path), "<tmp>"), "frames": int(tCap.get(cv2.CAP_PROP_FRAME_COUNT)),
                          "size": [int(tCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(tCap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}
        tCap.release()
    checkSnapshot("side_by_side", tResult)
    checkSnapshot("side_by_side.cli", parsedArgs(monkeypatch, str(SIDE_BY_SIDE), ["a", "b", "c"], isPath=True))
    assert helpExitCode(monkeypatch, str(SIDE_BY_SIDE), ["a", "b", "c"], isPath=True) == 0


def test_robotize_bi_take(tmp_path, monkeypatch):
    tModule = _load(ROBOTIZE, "umi_robotize_bi_under_test")
    tTake = tmp_path / "20261021_113243"
    tTake.mkdir()
    tRng = np.random.default_rng(9)
    tT = (T0_NS + np.arange(0, 20, 1 / UMI_HZ) * NS).astype(np.int64)
    tSegs = [{"rep": 1, "hold_pre": [int(tT[10]), int(tT[20])], "hold_post": [int(tT[200]), int(tT[260])], "back": None},
             {"rep": 2, "hold_pre": [int(tT[300]), int(tT[310])], "hold_post": None, "back": [int(tT[400]), int(tT[560])]},
             {"rep": 3, "hold_pre": [int(tT[590]), int(tT[592])], "hold_post": [int(tT[593]), int(tT[595])], "back": None}]
    (tTake / "segments.json").write_text(json.dumps(tSegs))
    tTheta = np.linspace(0, 6 * np.pi, len(tT))
    for tSide, tY0 in (("right", -0.05), ("left", 0.04)):
        tP = np.stack([0.25 + 0.1 * np.sin(tTheta), tY0 + 0.12 * np.sin(tTheta / 3), 0.01 + 0.03 * np.cos(tTheta)], 1)
        tP += tRng.normal(0, 0.001, (len(tT), 3))
        np.savez(tTake / f"umi_traj_{tSide}_fused.npz", t_ns=tT, p_B=tP, R_B_tcp=np.tile(np.eye(3), (len(tT), 1, 1)),
                 extra=np.array("keep"))
    monkeypatch.setattr(sys, "argv", ["robotize", str(tTake), "--src", "_fused", "--sep", "0.12"])
    _, tStdout = _quiet(tModule.main)
    tResult = {"stdout": tStdout.replace(str(tmp_path), "<tmp>"), "soft": [tModule.softFloor(np.array([-0.1, 0.0, 0.2]), 0.05, 0.01),
                                                                          tModule.softCeil(np.array([-0.1, 0.0, 0.2]), 0.05, 0.01)]}
    for tSide in ("right", "left"):
        with np.load(tTake / f"umi_traj_{tSide}_fused_robot.npz", allow_pickle=True) as tZ:
            tResult[tSide] = {tKey: {"dtype": str(tZ[tKey].dtype), "value": tZ[tKey] if tZ[tKey].ndim < 3 else tZ[tKey].sum()}
                              for tKey in sorted(tZ.files)}
            tResult[tSide]["robotize"]["value"] = json.loads(str(tZ["robotize"]).replace(str(tmp_path), "<tmp>"))
    checkSnapshot("robotize_bi", tResult)
    checkSnapshot("robotize_bi.cli", parsedArgs(monkeypatch, str(ROBOTIZE), ["take"], isPath=True))
    assert helpExitCode(monkeypatch, str(ROBOTIZE), ["take"], isPath=True) == 0


def test_align_cli(monkeypatch):
    checkSnapshot("align.cli", parsedArgs(monkeypatch, str(ALIGN), ["take"], isPath=True))
    assert helpExitCode(monkeypatch, str(ALIGN), ["take"], isPath=True) == 0


def test_align_holds(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    tModule = _load(ALIGN, "umi_align_under_test")
    tTake = tmp_path / "20261020_150000"
    tTake.mkdir()
    tStart = 7 * NS
    tProtocol = [{"text": "준비", "t0": 0.0, "dur": 2.0}]
    for tI in range(4):
        tProtocol.append({"text": f"UMI 를 그리퍼에 맞춰 대고 멈추세요 {tI + 1}/4", "t0": 2.0 + 6.0 * tI, "dur": 5.0})
    (tTake / "meta.json").write_text(json.dumps({"t_start_monotonic_ns": tStart, "protocol": tProtocol}, ensure_ascii=False))
    tT = (tStart + np.arange(0, 28, 1 / UMI_HZ) * NS).astype(np.int64)
    tRng = np.random.default_rng(21)
    tBase = Rotation.from_euler("xyz", [178, -3, 91], degrees=True)
    tR = (Rotation.from_rotvec(tRng.normal(0, np.radians(1.5), (len(tT), 3))) * tBase).as_matrix()
    tTags = (np.arange(len(tT)) % 4 != 0).astype(np.int64)
    tTags[(tT > tStart + 19 * NS) & (tT < tStart + 25 * NS)] = 0             # hold 4 has too few tag frames
    np.savez(tTake / "umi_traj_right.npz", t_ns=tT, ntags=tTags, R_B_tcp=tR)
    tOutFile = tmp_path / "align.json"
    monkeypatch.setattr(sys, "argv", ["align", str(tTake), "--write", "--out", str(tOutFile), "--holds", "1", "2", "4"])
    _, tStdout = _quiet(tModule.main)
    tWritten = json.loads(tOutFile.read_text())
    tWritten["take"] = tWritten["take"].replace(str(tmp_path), "<tmp>")
    monkeypatch.setattr(sys, "argv", ["align", str(tTake), "--holds", "4"])
    tStderr = io.StringIO()
    with contextlib.redirect_stderr(tStderr):
        tCode, _ = _quiet(tModule.main)
    assert tCode == 1
    checkSnapshot("align.main", {"stdout": tStdout.replace(str(tmp_path), "<tmp>"), "written": tWritten,
                                 "exit": tStderr.getvalue().strip()},
                  rel=1e-9, abs_=1e-9)
