"""Test fixtures of laundry_dataset: a golden-value recorder and a synthetic recording / training tree.

`golden`  compares every value a test hands it against tests/golden_dataset.npz, exactly (array_equal, NaN == NaN). With
          LAUNDRY_CHAR_RECORD=1 the values are recorded instead (the characterization run against the pre-refactor code).
`world`   the synthetic tree of tests/synthetic.py (takes, calibration, idle-trim table, derived copies + ik_v1 labels made by the
          IK stage with the deterministic FakeIk solver), built once per session.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

GOLDEN_FILE = Path(__file__).with_name("golden_dataset.npz")
RECORD = os.environ.get("LAUNDRY_CHAR_RECORD") == "1"


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for tKey in sorted(value, key=str):
            _flatten(f"{prefix}/{tKey}", value[tKey], out)
    elif isinstance(value, (list, tuple)) and not (value and all(isinstance(tX, (int, float, bool, np.number)) for tX in value)):
        out[f"{prefix}#len"] = np.asarray(len(value))
        for tIndex, tItem in enumerate(value):
            _flatten(f"{prefix}[{tIndex}]", tItem, out)
    elif torch.is_tensor(value):
        out[prefix] = value.detach().cpu().numpy()
    elif value is None:
        out[prefix] = np.asarray("<None>")
    elif isinstance(value, (str, Path)):
        out[prefix] = np.asarray(str(value))
    else:
        out[prefix] = np.asarray(value)


class Golden:
    def __init__(self):
        self.mStore = dict(np.load(GOLDEN_FILE, allow_pickle=False)) if GOLDEN_FILE.exists() else {}
        self.mNew = {}

    def check(self, name, value):
        tFlat = {}
        _flatten(name, value, tFlat)
        if RECORD:
            self.mNew.update(tFlat)
            return
        tExpectedKeys = sorted(tK for tK in self.mStore if tK == name or tK.startswith(name + "/") or tK.startswith(name + "[")
                               or tK.startswith(name + "#"))
        assert sorted(tFlat) == tExpectedKeys, f"{name}: keys differ: {sorted(set(tFlat) ^ set(tExpectedKeys))[:10]}"
        for tKey, tValue in tFlat.items():
            tExp = self.mStore[tKey]
            assert tValue.shape == tExp.shape, f"{tKey}: shape {tValue.shape} != {tExp.shape}"
            if tExp.dtype.kind in "fc" or tValue.dtype.kind in "fc":
                assert np.array_equal(tValue.astype(np.float64), tExp.astype(np.float64), equal_nan=True), \
                    f"{tKey}: max diff {np.nanmax(np.abs(tValue.astype(np.float64) - tExp.astype(np.float64)))}"
            else:
                assert np.array_equal(tValue, tExp), f"{tKey}: {tValue!r} != {tExp!r}"


_GOLDEN = Golden()


@pytest.fixture(scope="session")
def golden():
    yield _GOLDEN
    if RECORD and _GOLDEN.mNew:
        tAll = dict(_GOLDEN.mStore)
        tAll.update(_GOLDEN.mNew)
        np.savez_compressed(GOLDEN_FILE, **tAll)




# ---------------------------------------------------------------- synthetic world
def installFakeSolvers(arm):
    """replace the Pink QP solvers of a robotize_ik.Arm by the deterministic FakeIk (same step bounds as the real ones)"""
    from .synthetic import FakeIk
    arm.mFree = FakeIk(arm.mPk, arm.mArgs.ik_step, 2)
    arm.mF0 = FakeIk(arm.mPk, 180, 3)
    arm.mFine = FakeIk(arm.mPk, 2.0, 1)
    return arm


def fakeArmClass(armClass):
    class FakeArm(armClass):
        def __init__(self, side, a):
            super().__init__(side, a)
            installFakeSolvers(self)
    return FakeArm


def robotizeArgs(**over):
    from .synthetic import ROBOTIZE_DEFAULTS
    return SimpleNamespace(**{**ROBOTIZE_DEFAULTS, **over})


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    import json
    from laundry_dataset import calibration
    from laundry_dataset import robotize_ik as R
    from . import synthetic as syn
    tRoot = tmp_path_factory.mktemp("world")
    tEgo, tTrain = tRoot / "ego_collector", tRoot / "laundry_dit_training"
    tTagRig = tEgo / "datasets/tag_rig"
    tDerived = tTrain / "data/tag_rig_derived"
    tAlign, tJawCalib = syn.writeCalibration(tEgo / "configs/calibration")
    tTakes = {}
    for tSeed, tName in enumerate(syn.TAKES):
        tFused = tName != "20261023_114817"
        syn.writeTake(tTagRig / tName, tSeed, images=(tSeed == 0), fused=tFused)
        tSrc = "_fused" if tFused else ""
        syn.writeDerived(tTagRig / tName, tDerived / (tName + tSrc), tSrc)
        tTakes[tName] = tSrc
    tTrim = tTrain / "reports/idle_trim.csv"
    syn.writeIdleTrim(tTrim, syn.TAKES)
    tSaved = calibration.GRIPPER_ALIGN, R.Arm
    calibration.GRIPPER_ALIGN = tAlign
    R.Arm = fakeArmClass(tSaved[1])
    try:
        for tName, tSrc in tTakes.items():
            for tDemo in (0, 1):
                tOut, tMsg = R.runDemo((str(tTagRig / tName), str(tDerived / (tName + tSrc)), tDemo, robotizeArgs()))
                assert tMsg not in ("exists", "short") and not tMsg.startswith("FAILED"), tMsg
    finally:
        calibration.GRIPPER_ALIGN, R.Arm = tSaved
    return SimpleNamespace(root=tRoot, ego=tEgo, train=tTrain, tag_rig=tTagRig, derived=tDerived, align=tAlign,
                           jaw_calib=tJawCalib,
                           trim=tTrim, takes=tTakes)
