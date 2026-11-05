"""Characterization of laundry_jaw.from_motor: hysteresis states / events and the umi_jaw.npz it writes per hand."""
from __future__ import annotations

import contextlib
import io
import sys

import numpy as np

from laundry_jaw import from_motor
from laundry_jaw.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

MODULE = "laundry_jaw.from_motor"
SAMPLE_NS = 10_000_000
N_SAMPLES = 300
RIGHT_REPLUG_SHIFT = 2150.0


def _openCloseProfile(n: int) -> np.ndarray:
    tPhase = np.linspace(0.0, 4.0 * np.pi, n)
    return 0.5 + 0.5 * np.sin(tPhase) + 0.03 * np.cos(17.0 * tPhase)


def _writeTake(root):
    tTake = root / "20261030_120000"
    tTake.mkdir()
    tT = (np.arange(N_SAMPLES) * SAMPLE_NS + 123_456_789).astype(np.int64)
    tProfile = _openCloseProfile(N_SAMPLES)
    # Jaw rests closed most of the time: square the profile so the median sits near closed.
    tShape = tProfile ** 3
    tLeft = -850.0 + tShape * (200.0 + 850.0)
    tRight = 1830.0 + tShape * (1340.0 - 1830.0) + RIGHT_REPLUG_SHIFT
    np.savez(tTake / "gripper_motor_left.npz", host_ns=tT, pos=tLeft.astype(np.float32))
    np.savez(tTake / "gripper_motor_right.npz", host_ns=tT, pos=tRight.astype(np.float32))
    return tTake


def test_hysteresis_states_and_events():
    tT = (np.arange(N_SAMPLES) * SAMPLE_NS).astype(np.int64)
    tG = _openCloseProfile(N_SAMPLES)
    tState, tEvents = from_motor.hysteresis(tT, tG, 0.55, 0.35, 0.1)
    checkSnapshot("from_motor.hysteresis", {"state": tState, "events": tEvents, "dtype": str(tState.dtype)})


def test_main_writes_umi_jaw_per_hand(tmp_path, monkeypatch):
    tTake = _writeTake(tmp_path)
    monkeypatch.setattr(sys, "argv", ["from_motor", str(tTake), "--dwell", "0.05"])
    tOut = io.StringIO()
    with contextlib.redirect_stdout(tOut):
        from_motor.main()
    tResult = {"stdout": tOut.getvalue().replace(str(tmp_path), "<tmp>")}
    for tDir in ("wrist", "wrist_left"):
        with np.load(tTake / tDir / "umi_jaw.npz") as tZ:
            tResult[tDir] = {tKey: {"dtype": str(tZ[tKey].dtype), "value": tZ[tKey]} for tKey in sorted(tZ.files)}
    checkSnapshot("from_motor.main", tResult)


def test_cli_defaults(monkeypatch):
    checkSnapshot("from_motor.cli", parsedArgs(monkeypatch, MODULE, ["takeA", "takeB"]))
    assert helpExitCode(monkeypatch, MODULE, []) == 0


def test_flipped_jaw_direction_takes_open_from_the_long_tail(tmp_path):
    # 11-04: left jaw re-assembled -> reading FALLS when opening (calibration says it rises: closed -850, open 200)
    tPos = np.full(1000, -850.0)
    tPos[300:400] = -1450.0                    # opens downwards by 600 ticks
    tNs = (np.arange(1000) * 1e7).astype(np.int64)
    np.savez(tmp_path / "gripper_motor_left.npz", host_ns=tNs, pos=tPos)
    with contextlib.redirect_stdout(io.StringIO()) as tOut:
        from_motor.labelHand(tmp_path, "left", from_motor.DEFAULT_HI, from_motor.DEFAULT_LO, from_motor.DEFAULT_DWELL_S)
    assert "FLIPPED" in tOut.getvalue()
    with np.load(tmp_path / "wrist_left" / "umi_jaw.npz") as tJaw:
        assert tJaw["g_cont"][350] == 1.0 and tJaw["g_cont"][100] == 0.0
