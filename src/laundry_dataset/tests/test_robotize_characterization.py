"""Characterization of the IK robotization stage (robotize_ik: Arm, smoothing, bump, the per-demo label run, the CLI) on the
synthetic world, with the deterministic FakeIk solver in place of the Pink QP solver (no QP solver in the test environment)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pytest

from laundry_dataset import calibration
from laundry_dataset import robot_arm
from laundry_dataset import robotize_ik as R
from laundry_dataset.takes import demoWindows

from .conftest import fakeArmClass, installFakeSolvers, requireRebotUrdf, robotizeArgs
from .synthetic import ROBOTIZE_DEFAULTS


def _npz(path):
    tZ = np.load(path, allow_pickle=False)
    return {tK: tZ[tK] for tK in tZ.files}


def test_smooth_bump_pose_err_demos(golden, world):
    tRng = np.random.default_rng(0)
    tX = tRng.normal(0, 1, (30, 3))
    for tK in (1, 2, 5, 9):
        golden.check(f"rik/smooth_{tK}", R.smooth(tX, tK))
    golden.check("rik/bump", [R.bump(40, 10, 15, 4), R.bump(40, 0, 3, 5), R.bump(40, 36, 40, 10)])
    tF, tT = np.eye(4), np.eye(4)
    tT[:3, 3] = [0.3, -0.4, 0.0]
    assert robot_arm.poseErr(tF, tT) == pytest.approx(0.5, abs=0)
    golden.check("rik/demos", [dict(idx=tD.mIdx, t0_ns=tD.mT0Ns,
                                    t1_ns=tD.mT1Ns) for tD in demoWindows(world.tag_rig / "20261022_124754")])
    assert robot_arm.DT_S == 0.05


@pytest.fixture(scope="module")
def arms():
    requireRebotUrdf()
    return {tS: installFakeSolvers(R.Arm(tS, robotizeArgs(level_max=25.0))) for tS in ("left", "right")}


def test_arm_geometry(golden, arms):
    for tS, tArm in arms.items():
        golden.check(f"rik/arm_{tS}", {"lo": tArm.mLo, "hi": tArm.mHi, "qr": tArm.mQReady, "fids": tArm.mLinkFrameIds,
                                       "links": tArm.links(tArm.mQReady), "fk": tArm.mPk.fk(tArm.mQReady), "side": tArm.mSide})
        tT = tArm.mPk.fk(tArm.mQReady)
        tOut = []
        for tZ in (0.2, 0.03, 0.01, -0.02):
            for tG in (0.0, 0.5, 1.0, 2.0):
                tT2 = tT.copy()
                tT2[2, 3] = tZ
                tOut.append([tArm.cornerLift(tT2, tG), tArm.level(tT2, tG)])
        golden.check(f"rik/lift_level_{tS}", tOut)
        tFlat = np.eye(4)
        tFlat[:3, 3] = [0.3, 0.1, 0.0]
        golden.check(f"rik/level_flat_{tS}", tArm.level(tFlat, 0.5))


def _targets(arm):
    """60 frames: a small joint wiggle, a 0.25 m jump at frame 25 the bounded solver lags behind, an unreachable stretch 45-50"""
    tQ = arm.mQReady[None] + 0.1 * np.sin(np.arange(60) / 6.0)[:, None] * np.array([1, 0.5, -0.5, 0.3, 0.2, 0.4])
    tT = np.array([arm.mPk.fk(tQRow) for tQRow in tQ])
    tT[25:, 0, 3] += 0.25
    tT[45:51, 0, 3] += 3.0
    return tT


@pytest.mark.parametrize("mode", ["seq", "reseed", "indep", "indep_flip", "bridge", "bridge2"])
def test_arm_solve_modes(golden, arms, mode):
    tArm = arms["left"]
    tArm.mArgs = robotizeArgs(level_max=25.0, ik_mode=mode.replace("_flip", ""), reseed_n=3, seeds=3,
                              **({"indep_flip": True} if mode == "indep_flip" else {}))
    tQ, tPe, tFlip = tArm.solve(_targets(tArm), np.ones(60))
    golden.check(f"rik/solve_{mode}", {"Q": tQ, "PE": tPe, "flip": tFlip, "reseed": tArm.mLastReseed, "bridge": tArm.mLastBridge,
                                       "log": json.dumps(tArm.mLastLog), "phi": tArm.mLastPhi})


def test_arm_multistart_and_join(golden, arms):
    tArm = arms["right"]
    tArm.mArgs = robotizeArgs()
    tT = _targets(tArm)
    tRng = np.random.default_rng(3)
    golden.check("rik/multistart", list(tArm._multiStart(tArm.mQReady, tT[30], tRng, 3)))
    golden.check("rik/multistart_pref", list(tArm._multiStartNear(tArm.mQReady, tT[30], tRng, 3)))
    tA = {tFrame: tArm.mQReady + 0.01 * tFrame for tFrame in range(0, 10)}
    tB = {tFrame: tArm.mQReady + 0.02 * tFrame for tFrame in range(5, 20)}
    golden.check("rik/join", list(tArm._join(tA, tB, 0, 10, 19, np.radians(7.5))))
    assert tArm._join(tA, {}, 0, 10, 19, np.radians(7.5)) is None


def test_run_demo_outputs(golden, world):
    for tName, tSrc in world.takes.items():
        for tDemo in (0, 1):
            tZ = _npz(world.derived / (tName + tSrc) / "ik_v1" / f"demo{tDemo:02d}.npz")
            golden.check(f"rik/run_{tName}_{tDemo}", tZ)


def test_run_demo_variants(golden, world, tmp_path, monkeypatch):
    import shutil
    tName = "20261022_124754"
    tDst = tmp_path / tName
    shutil.copytree(world.derived / (tName + "_fused"), tDst)
    monkeypatch.setattr(calibration, "GRIPPER_ALIGN", world.align)
    monkeypatch.setattr(R, "Arm", fakeArmClass(R.Arm))
    tArgs = robotizeArgs(force=True, arm_gap=0.6, rounds=2, level_max=20.0, ik_mode="reseed", reseed_n=3, seeds=2)
    tOut, tMsg = R.runDemo((str(world.tag_rig / tName), str(tDst), 0, tArgs))
    golden.check("rik/variant_msg", tMsg)
    golden.check("rik/variant", _npz(tOut))
    assert R.runDemo((str(world.tag_rig / tName), str(tDst), 0, robotizeArgs()))[1] == "exists"
    tOut2, tMsg2 = R.runDemo((str(world.tag_rig / tName), str(tmp_path / "missing"), 1, tArgs))
    assert tOut2.endswith("ik_v1/demo01.npz") and tMsg2.startswith("FAILED FileNotFoundError")


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


def test_cli_defaults(golden, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["robotize_ik.py"])
    tP = _capturedParser(monkeypatch, R.main)
    tNs = vars(tP.parse_args(["TAKE", "DERIVED"]))
    assert tNs == dict(ROBOTIZE_DEFAULTS, take="TAKE", derived="DERIVED")
    tActions = {tA.dest: (tA.option_strings, tA.choices, tA.type.__name__ if tA.type else None) for tA in tP._actions}
    golden.check("rik/cli_actions", json.dumps(tActions, sort_keys=True))
    with pytest.raises(SystemExit):
        tP.parse_args(["--help"])
    assert "usage: robotize_ik.py <take_dir> <derived_take_dir>" in capsys.readouterr().out


def test_cli_main_runs_in_process(world, tmp_path, monkeypatch, capsys):
    import shutil
    tName = "20261023_114817"
    tDst = tmp_path / tName
    shutil.copytree(world.derived / tName, tDst)
    monkeypatch.setattr(calibration, "GRIPPER_ALIGN", world.align)
    monkeypatch.setattr(R, "Arm", fakeArmClass(R.Arm))
    monkeypatch.setattr(sys, "argv", ["robotize_ik.py", str(world.tag_rig / tName), str(tDst), "--demos", "1"])
    R.main()
    tOut = capsys.readouterr().out.splitlines()
    assert tOut[0] == "demo01.npz: exists"
    assert tOut[1].startswith(f"done {tName} 1 demos in ")
