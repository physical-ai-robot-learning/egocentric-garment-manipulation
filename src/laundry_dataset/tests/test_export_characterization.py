"""Characterization of the LeRobot export stage on the synthetic world: the shared converter pieces and the v1 builder
(tag_rig_to_lerobot: arm assignment, prep, per-demo grid + QA, split, image loading, jaw calibration gate, --qa-only run) and
the unsplit builder (tag_rig_unsplit: per-arm masks / weights per option set, QA video, --qa-only run). The LeRobot write
pass needs lerobot and is not run here."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest

from laundry_dataset import arm_assignment as AA
from laundry_dataset import calibration
from laundry_dataset import export_common as EC
from laundry_dataset import tag_rig_to_lerobot as T2L
from laundry_dataset import tag_rig_unsplit as U
from laundry_dataset import unsplit_build as UB
from laundry_dataset.takes import demoWindows
from laundry_dataset.unsplit_build import UnsplitOptions
from laundry_dataset.unsplit_render import renderMaskVideo

from .synthetic import ASSIGNMENT

TAKE = "20261022_124754"


def _demo(world, take, idx):
    return demoWindows(world.tag_rig / take)[idx]


# ---------------------------------------------------------------- tag_rig_to_lerobot
def test_constants_and_meta(golden):
    golden.check("t2l/meta", json.dumps(T2L.sourceMeta(), sort_keys=True, ensure_ascii=False))
    golden.check("t2l/consts", dict(TASK=EC.TASK, IMG_WH=EC.IMG_WH, CAMS=EC.CAMS, MAX_SRC_GAP_S=EC.MAX_SRC_GAP_S,
                                    MAX_JAW_GAP_S=EC.MAX_JAW_GAP_S, MIN_SECONDS=T2L.MIN_SECONDS, MIN_SEG_S=T2L.MIN_SEG_S,
                                    GRIP_RAMP_S=EC.GRIP_RAMP_S, TAG_GAP_MASK_S=T2L.TAG_GAP_MASK_S,
                                    MIN_GOOD_FRAC=T2L.MIN_GOOD_FRAC,
                                    SPLIT_FRAC=EC.SPLIT_FRAC, QUALITY_NAMES=T2L.QUALITY_NAMES,
                                    ROBOTIZE_ARGS=AA.ROBOTIZE_ARGS, IK_WORKERS=AA.IK_WORKERS))


def test_demos_and_ang_speed(golden, world):
    golden.check("t2l/demos", [dict(idx=tD.mIdx, t0_ns=tD.mT0Ns, t1_ns=tD.mT1Ns,
                                    ended_early=tD.mEndedEarly) for tD in demoWindows(world.tag_rig / TAKE)])
    golden.check("t2l/ang_speed", AA._angSpeed(np.load(world.tag_rig / TAKE / "umi_traj_right.npz")))


def test_flow_mag(golden, world, tmp_path):
    tCam = tmp_path / "cam"
    tCam.mkdir()
    tT = world.tag_rig / TAKE / "wrist"
    for tI in range(14):
        if tI != 6:                                                  # a dropped jpg is skipped
            shutil.copy(tT / f"wrist_{tI * 3:05d}.jpg", tCam / f"wrist_{tI:05d}.jpg")
    np.save(tCam / "t_ns.npy", np.arange(14) * 10_000_000)
    golden.check("t2l/flow", list(AA._flowMag(tCam, 0, 200_000_000, step=1)))
    golden.check("t2l/flow2", list(AA._flowMag(tCam, 20_000_000, 120_000_000)))


def _fakeFlow(world):
    tZ = {tS: np.load(world.tag_rig / TAKE / f"umi_traj_{tS}.npz") for tS in ("right", "left")}
    tA = {tS: AA._angSpeed(tZ[tS]) for tS in tZ}

    def flow(camDir, tLo, tHi, step=2):
        tT = np.load(camDir / "t_ns.npy")
        tSel = tT[(tT >= tLo) & (tT <= tHi)][::step]
        tRig = "left" if camDir.name == "wrist" else "right"         # cam 'wrist' moves with rig 'left'
        tTa, tAa = tA[tRig]
        return tSel[1:], np.interp(tSel[1:], tTa, tAa) + 0.01 * np.sin(tSel[1:] / 1e8)
    return flow


def test_assign_arms(golden, world, monkeypatch):
    monkeypatch.setattr(AA, "_flowMag", _fakeFlow(world))
    tOut = AA.assignArms(world.tag_rig / TAKE)
    assert tOut["rig_of_arm"] == ASSIGNMENT["rig_of_arm"] and tOut["cam_of_arm"] == ASSIGNMENT["cam_of_arm"]
    golden.check("t2l/assign", json.dumps(tOut, sort_keys=True))


def test_assign_arms_conflict_and_same_side(world, tmp_path, monkeypatch):
    tFlow = _fakeFlow(world)
    monkeypatch.setattr(AA, "_flowMag", lambda c, a, b,
                        step=2: tFlow(c.parent / ("wrist_left" if c.name == "wrist" else "wrist"), a, b, step))
    with pytest.raises(AA.ArmAssignmentError, match=r"\[20261022_124754\] ASSIGNMENT CONFLICT"):
        AA.assignArms(world.tag_rig / TAKE)
    tTake = tmp_path / "same"
    shutil.copytree(world.tag_rig / TAKE, tTake, ignore=shutil.ignore_patterns("*.jpg"))
    tZ = dict(np.load(tTake / "umi_traj_left.npz"))
    tZ["p_B"] = tZ["p_B"] * [1, -1, 1]
    np.savez(tTake / "umi_traj_left.npz", **tZ)
    with pytest.raises(ValueError, match="rigs on the same side of the base"):
        AA.assignArms(tTake)


@pytest.mark.parametrize("pose_src", ["", "_fused"])
def test_prep_take(golden, world, tmp_path, monkeypatch, pose_src):
    monkeypatch.setattr(AA, "_flowMag", _fakeFlow(world))
    tCalls = []

    def fakeRun(cmd, **kw):
        tCalls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, "ik out\n", "ik err\n")
    monkeypatch.setattr(subprocess, "run", fakeRun)
    tAsg = AA.prepTake(world.tag_rig / TAKE, tmp_path, pose_src, "ik_v1")
    tD = tmp_path / (TAKE + pose_src)
    assert tAsg["robotize"] == "ik_v1"
    tDisk = json.loads((tD / "assignment.json").read_text())
    assert tDisk.pop("take") == str(world.tag_rig / TAKE) and tDisk.pop("created")
    golden.check(f"t2l/prep_{pose_src}", json.dumps(tDisk, sort_keys=True))
    for tS in ("left", "right"):
        golden.check(f"t2l/prep_{pose_src}_{tS}", {tK: tV for tK, tV in np.load(tD / f"umi_traj_{tS}.npz").items()})
    assert (tD / "segments.json").exists() and (tD / "robotize_ik.log").read_text() == "ik out\nik err\n"
    (tCmd, tKw), = tCalls
    tTail = tCmd[tCmd.index(next(tC for tC in tCmd if tC.endswith("robotize_ik.py") or tC == "laundry_dataset.robotize_ik")) + 1:]
    assert tTail == [str(world.tag_rig / TAKE), str(tD), "--workers", "6"]
    assert str(tCmd[0]).endswith("xvla-mac/bin/python") and tKw["capture_output"] and tKw["text"]
    assert tKw["env"]["PYTHONPATH"].split(":")[0].endswith("/src")
    tCalls.clear()
    AA.prepTake(world.tag_rig / TAKE, tmp_path, pose_src)            # assignment exists -> only the IK stage runs again
    assert len(tCalls) == 1
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3, "", "boom"))
    with pytest.raises(RuntimeError, match="robotize_ik failed: boom"):
        AA.prepTake(world.tag_rig / TAKE, tmp_path, pose_src)


def _bd(b):
    return {tK: tV for tK, tV in b.items() if tK != "cam_dir"}


@pytest.mark.parametrize("take", ["20261022_124754", "20261023_114817"])
def test_build_demo_ik_v1(golden, world, monkeypatch, take):
    for tIdx in (0, 1):
        tB = T2L.buildDemo(world.tag_rig / take, world.derived / (take + world.takes[take]), ASSIGNMENT, _demo(world, take, tIdx),
                           "ik_v1")
        assert tB["cam_dir"]["left_wrist"] == world.tag_rig / take / "wrist_left"
        golden.check(f"t2l/build_{take}_{tIdx}", _bd(tB))


def test_build_demo_sep(golden, world, tmp_path, monkeypatch):
    tD = tmp_path / TAKE
    shutil.copytree(world.derived / (TAKE + "_fused"), tD)
    for tS in ("left", "right"):
        tZ = dict(np.load(tD / f"umi_traj_{tS}.npz", allow_pickle=True))
        tZ["p_B_human"] = tZ["p_B"] + 0.01
        tZ["robotize_quality"] = (np.arange(len(tZ["t_ns"])) % 3).astype(np.int64)
        tZ["seg_src"] = (np.arange(len(tZ["t_ns"])) // 50).astype(np.int64)
        np.savez(tD / f"umi_traj_{tS}_robot.npz", **tZ)
    monkeypatch.setattr(calibration, "GRIPPER_ALIGN", world.align)
    tB = T2L.buildDemo(world.tag_rig / TAKE, tD, ASSIGNMENT, _demo(world, TAKE, 0), "sep")
    golden.check("t2l/build_sep", _bd(tB))


def test_build_demo_errors(world, tmp_path, monkeypatch):
    tD = tmp_path / TAKE
    shutil.copytree(world.derived / (TAKE + "_fused"), tD)
    (tD / "ik_v1" / "demo01.npz").unlink()
    with pytest.raises(FileNotFoundError, match=r"ik_v1 robotization missing \(demo01.npz\)"):
        T2L.buildDemo(world.tag_rig / TAKE, tD, ASSIGNMENT, _demo(world, TAKE, 1))


def test_load_img_and_split(golden, world, tmp_path):
    tB = {"cam_dir": {"global": world.tag_rig / TAKE, "left_wrist": tmp_path, "right_wrist": world.tag_rig / TAKE / "wrist"}}
    golden.check("t2l/img_global", EC.loadImage(tB, "global", 3))
    shutil.copy(world.tag_rig / TAKE / "wrist_left" / "wrist_00012.jpg", tmp_path / "wrist_00012.jpg")
    golden.check("t2l/img_fallback", EC.loadImage(tB, "left_wrist", 10))
    with pytest.raises(IOError, match="unreadable"):
        EC.loadImage(tB, "left_wrist", 100)
    tKeys = [f"2026102{tT}_1{tT}0000/demo{tD:02d}" for tT in range(4) for tD in range(2 + 3 * tT)]
    for tN in (2, 3, 7, len(tKeys)):
        golden.check(f"t2l/split_{tN}", EC.splitAssign(tKeys[:tN], 0))
    golden.check("t2l/split_seed5", EC.splitAssign(tKeys, 5, (0.7, 0.2, 0.1)))


def test_check_jaw_calib(world, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "JAW_CALIB", world.jaw_calib)
    assert calibration.checkJawCalib() == dict(file=str(world.jaw_calib), threshold_px={"left": 191.0, "right": 150.0})
    tBad = tmp_path / "jaw.json"
    tBad.write_text(json.dumps({"left": {"threshold_px": 190}, "right": {"threshold_px": 150}}))
    monkeypatch.setattr(calibration, "JAW_CALIB", tBad)
    with pytest.raises(calibration.JawCalibrationError, match="jaw calibration jaw.json thresholds"):
        calibration.checkJawCalib()


def _capturedParser(monkeypatch, entry):
    tBox = {}

    class _Stop(Exception):
        pass

    def capture(self, *a, **kw):
        tBox["parser"] = self
        raise _Stop
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_Stop):
        entry()
    monkeypatch.undo()
    return tBox["parser"]


def _actions(parser):
    return json.dumps({tA.dest: (tA.option_strings, tA.choices, tA.nargs, tA.required, tA.type.__name__ if tA.type else None,
                                 tA.default if not str(tA.default).startswith("/") else "<path>") for tA in parser._actions},
                      sort_keys=True,
                      ensure_ascii=False)


def test_t2l_cli(golden, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tag_rig_to_lerobot.py"])
    tP = _capturedParser(monkeypatch, T2L.main)
    golden.check("t2l/cli", _actions(tP))
    with pytest.raises(SystemExit):
        tP.parse_args(["--help"])
    assert "usage: tag_rig_to_lerobot.py --out data/lerobot/tagrig_laundry_v1" in capsys.readouterr().out


@pytest.fixture
def fakeLerobot(monkeypatch):
    for tName in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
        monkeypatch.setitem(sys.modules, tName, types.ModuleType(tName))
    sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = None


@pytest.fixture
def derivedCopy(world, tmp_path):
    tD = tmp_path / "derived"
    shutil.copytree(world.derived, tD)
    return tD


def _fakeIkRun(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))


def test_t2l_qa_only(golden, world, derivedCopy, tmp_path, monkeypatch, capsys, fakeLerobot):
    _fakeIkRun(monkeypatch)
    monkeypatch.setattr(calibration, "JAW_CALIB", world.jaw_calib)
    tQa = tmp_path / "qa.csv"
    tArgv = ["--out", str(tmp_path / "out"), "--takes", *[str(world.tag_rig / tT) for tT in world.takes], "--derived",
             str(derivedCopy),
             "--pose-src", "_fused", "--qa-only", str(tQa), "--tag-gap-s", "0.6", "--min-good-frac", "0.2"]
    monkeypatch.setattr(sys, "argv", ["tag_rig_to_lerobot.py", *tArgv])
    T2L.main()
    golden.check("t2l/qa_csv", tQa.read_text())
    golden.check("t2l/qa_stdout", capsys.readouterr().out.replace(str(tmp_path), "<TMP>").replace(str(world.root), "<ROOT>"))


# ---------------------------------------------------------------- tag_rig_unsplit
def test_unsplit_helpers(golden, world, tmp_path, monkeypatch):
    tRuns = [UB.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)), UB.runs(np.zeros(3, bool)), UB.runs(np.ones(4, bool))]
    golden.check("un/runs", tRuns)
    for tName in ("umi_traj_right.npz", "umi_traj_right_fused.npz"):
        golden.check(f"un/measured_{tName}", UB.measuredFlag(np.load(world.tag_rig / TAKE / tName)))
    np.savez(tmp_path / "slam.npz", ntags=np.array([0, 1, 0]), slam_ok=np.array([1, 0, 0]))
    golden.check("un/measured_slam", UB.measuredFlag(np.load(tmp_path / "slam.npz")))
    tGl = np.array([0, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 9.9])
    golden.check("un/pos_weights", UB.posWeights(tGl, tGl < 1.0, np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], bool)))
    tOpts = UnsplitOptions(mConfDir=str(tmp_path), mConfHigh=0.8, mConfMid=0.5)
    golden.check("un/conf_weight", UB.confWeight(np.array([0.1, 0.5, 0.79, 0.8, 1.0]), tOpts))
    (tmp_path / TAKE).mkdir()
    np.savez(tmp_path / TAKE / "umi_conf_left.npz", t_ns=np.arange(3), score=np.array([0.1, 0.9, 0.5]))
    golden.check("un/conf_load", list(UB.loadPosConf(tmp_path, TAKE, "right", "left")))
    with pytest.raises(FileNotFoundError, match="no SLAM<->IMU confidence"):
        UB.loadPosConf(tmp_path, TAKE, "x", "y")
    tDef = UnsplitOptions()
    golden.check("un/consts", dict(EXCLUDE_TAKES=sorted(U.EXCLUDE_TAKES), FILL_MASK_S=UB.FILL_MASK_S, IK_FAIL_MM=tDef.mIkFailMm,
                                   GRASP_OK_MM=UB.GRASP_OK_MM, GRASP_WARN_MM=UB.GRASP_WARN_MM, GRASP_W=UB.GRASP_W,
                                   DROP_MEASURED=U.DROP_MEASURED, DROP_FILL_IN_GRASP_S=U.DROP_FILL_IN_GRASP_S,
                                   VARIANTS=tDef.mVariants,
                                   MASK_MODE=tDef.mMaskMode, LABEL=tDef.mLabel, SPEED=tDef.mSpeed,
                                   JAW_WIN_S=UB.JAW_WIN_S, JAW_FILE=UB.JAW_FILE))


UNSPLIT_OPTIONS = {"default": dict(), "rob_stitched": dict(mLabel="rob", mMaskMode="stitched"),
                   "fused_gated_slow": dict(mLabel="fused", mMaskMode="gated", mSpeed=0.3, mFillMaskS=0.3, mIkFailMm=30.0),
                   "fused_conf": dict(mLabel="fused", CONF=True)}


def _confFiles(tmp_path, takes):
    for tTake in takes:
        (tmp_path / "conf" / tTake).mkdir(parents=True, exist_ok=True)
        for tRig in ("left", "right"):
            tT = np.arange(1_000_000_000_000, 1_000_000_000_000 + 21_000_000_000, 50_000_000)
            np.savez(tmp_path / "conf" / tTake / f"{tRig}.npz", t_ns=tT, conf=0.5 + 0.5 * np.sin(np.arange(len(tT)) / 7.0))


def _options(tmp_path, opts):
    opts = dict(opts)
    if opts.pop("CONF", False):
        _confFiles(tmp_path, (TAKE,))
        opts.update(mConfDir=str(tmp_path / "conf"), mConfHigh=0.8, mConfMid=0.4, mWMid=0.3)
    return UnsplitOptions(**opts)


def _win(world, take, idx):
    tD = _demo(world, take, idx)
    tA, tB = {0: (0.5, 8.5), 1: (0.3, 8.0)}[idx]
    return tD.mT0Ns + int(tA * 1e9), tD.mT0Ns + int(tB * 1e9)


@pytest.mark.parametrize("opt", list(UNSPLIT_OPTIONS))
def test_unsplit_build(golden, world, tmp_path, monkeypatch, opt):
    tOpts = _options(tmp_path, UNSPLIT_OPTIONS[opt])
    monkeypatch.setenv("EP_WEIGHT", "2" if opt == "rob_stitched" else "1")
    for tIdx in (0, 1):
        tB = UB.build(world.tag_rig / TAKE, world.derived / (TAKE + "_fused"), ASSIGNMENT, _demo(world, TAKE, tIdx),
                      _win(world, TAKE, tIdx), camShiftNs=-40_000_000 if tIdx else 0, options=tOpts)
        golden.check(f"un/build_{opt}_{tIdx}", _bd(tB))


def test_unsplit_build_short_window(world):
    tD = _demo(world, TAKE, 0)
    with pytest.raises(ValueError, match="trimmed window inside the recording only"):
        UB.build(world.tag_rig / TAKE, world.derived / (TAKE + "_fused"), ASSIGNMENT, tD, (tD.mT0Ns, tD.mT0Ns + 3_000_000_000))


def test_unsplit_render(golden, world, tmp_path):
    tB = UB.build(world.tag_rig / TAKE, world.derived / (TAKE + "_fused"), ASSIGNMENT, _demo(world, TAKE, 0),
                  _win(world, TAKE, 0))
    tOut = tmp_path / "v.mp4"
    renderMaskVideo(tB, tOut, "demo00")
    tCap = cv2.VideoCapture(str(tOut))
    tMeans = []
    while True:
        tOk, tFrame = tCap.read()
        if not tOk:
            break
        tMeans.append(tFrame.reshape(-1, 3).mean(0))
    assert len(tMeans) == tB["n"] and tFrame is None
    golden.check("un/render_shape", np.array([len(tMeans), *cv2.imread(str(world.tag_rig / TAKE / "color_00000.jpg")).shape]))
    golden.check("un/render_means", np.round(np.array(tMeans), 0))


def test_unsplit_cli(golden, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tag_rig_unsplit.py"])
    tP = _capturedParser(monkeypatch, U.main)
    golden.check("un/cli", _actions(tP))
    with pytest.raises(SystemExit):
        tP.parse_args(["--help"])
    assert "usage: tag_rig_unsplit.py --out data/lerobot/tagrig_laundry_v2u" in capsys.readouterr().out


@pytest.mark.parametrize("variant", ["gated", "stitched_fused_cont", "posconf"])
def test_unsplit_qa_only(golden, world, derivedCopy, tmp_path, monkeypatch, capsys, variant):
    _fakeIkRun(monkeypatch)
    monkeypatch.setattr(calibration, "JAW_CALIB", world.jaw_calib)
    monkeypatch.setattr(U, "TRIM_CSV", world.trim)
    tQa = tmp_path / "qa.csv"
    tArgv = ["--out", str(tmp_path / "out"), "--takes", *[str(world.tag_rig / tT) for tT in world.takes], "--derived",
             str(derivedCopy),
             "--qa-only", str(tQa)]
    if variant == "stitched_fused_cont":
        tArgv += ["--mask-mode", "stitched", "--label", "fused", "--speed", "0.5", "--ik-fail-mm", "30", "--grip-label", "cont",
                  "--demos", f"{TAKE}/demo01", "20261023_114817/demo00"]
    if variant == "posconf":
        _confFiles(tmp_path, (TAKE,))
        for tTake in world.takes:
            (tmp_path / "conf" / tTake).mkdir(parents=True, exist_ok=True)
            for tRig in ("left", "right"):
                shutil.copy(tmp_path / "conf" / TAKE / f"{tRig}.npz", tmp_path / "conf" / tTake / f"umi_conf_{tRig}.npz")
        tArgv += ["--label", "rob", "--pos-conf-dir", str(tmp_path / "conf"), "--conf-high", "0.8", "--conf-mid", "0.4",
                  "--pose-src", "auto"]
    monkeypatch.setattr(sys, "argv", ["tag_rig_unsplit.py", *tArgv])
    U.main()
    golden.check(f"un/qa_csv_{variant}", tQa.read_text())
    tStdout = capsys.readouterr().out.replace(str(tmp_path), "<TMP>").replace(str(world.root), "<ROOT>")
    golden.check(f"un/qa_stdout_{variant}", tStdout)


def test_unsplit_refuses_excluded_and_conf_without_thresholds(world, tmp_path, monkeypatch):
    _fakeIkRun(monkeypatch)
    monkeypatch.setattr(calibration, "JAW_CALIB", world.jaw_calib)
    monkeypatch.setattr(U, "TRIM_CSV", world.trim)
    monkeypatch.setattr(sys, "argv", ["tag_rig_unsplit.py", "--out", "x", "--takes", str(tmp_path / "20261023_113219")])
    with pytest.raises(SystemExit, match=r"excluded takes passed: \['20261023_113219'\]"):
        U.main()
    monkeypatch.setattr(sys, "argv", ["tag_rig_unsplit.py", "--out", "x", "--takes", "y", "--pos-conf-dir", "c"])
    with pytest.raises(SystemExit, match="--pos-conf-dir needs --conf-high and --conf-mid"):
        U.main()
