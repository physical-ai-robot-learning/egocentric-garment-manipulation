"""Characterization of the laundry_reference command lines: argparse defaults of every module, `--help`, the console-script
functions of cli.py, and the small helpers the regression tests in tests/laundry do not call directly."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from laundry_common.paths import REPO_ROOT
from laundry_reference import cli
from laundry_reference import parity_metrics
from laundry_reference import trajio
from laundry_reference import robotize
from laundry_reference.tests.snapshot import checkSnapshot, helpExitCode, parsedArgs

DEMO = REPO_ROOT / "assets" / "laundry" / "examples" / "synthetic_bimanual_demo.csv"
REPLAY = REPO_ROOT / "assets" / "laundry" / "examples" / "synthetic_robot_replay.csv"
CLI_CASES = {
    "gap_fill": ("laundry_reference.gap_fill", ["a.csv", "b.csv"]),
    "relative_se3": ("laundry_reference.relative_se3", ["a.csv"]),
    "robotize": ("laundry_reference.robotize", ["a.csv", "b.csv"]),
    "parity_metrics": ("laundry_reference.parity_metrics", ["a.csv", "b.csv"]),
    "plot_example": ("laundry_reference.plot_example", ["a.csv", "b.png"]),
}


@pytest.mark.parametrize("name", sorted(CLI_CASES))
def test_cli_defaults(monkeypatch, name):
    tModule, tArgv = CLI_CASES[name]
    checkSnapshot(f"reference.cli.{name}", parsedArgs(monkeypatch, tModule, tArgv))
    assert helpExitCode(monkeypatch, tModule, tArgv) == 0


def test_console_script_functions(tmp_path, monkeypatch):
    tFilled, tRobot = tmp_path / "f.csv", tmp_path / "r.csv"
    tCalls = {
        "gap_fill": [str(DEMO), str(tFilled)],
        "robotize": [str(tFilled), str(tRobot)],
        "parity_metrics": [str(tRobot), str(REPLAY)],
        "relative_se3": [str(tFilled)],
        "plot_example": [str(tRobot), str(tmp_path / "p.png")],
    }
    tResult = {}
    for tName, tArgv in tCalls.items():
        monkeypatch.setattr(sys, "argv", [tName, *tArgv])
        tOut = io.StringIO()
        with contextlib.redirect_stdout(tOut):
            tReturned = getattr(cli, tName)()
        tResult[tName] = {"returned_falsy": not tReturned, "stdout": tOut.getvalue().replace(str(tmp_path), "<tmp>")}
    checkSnapshot("reference.console_scripts", tResult)


def test_small_helpers():
    tY = np.linspace(-0.1, 0.1, 9)
    tX = np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 3.0], [3.0, 5.0]])
    tP = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [np.nan, 0, 0]])
    tQ = np.array([[0, 0, 0, 1.0], [0, 0, np.sin(0.1), np.cos(0.1)], [0, 0, 0, 1.0]])
    tRel, tRot, tOk = parity_metrics.relative(tP, tQ)
    checkSnapshot("reference.helpers", {
        "softceil": robotize.softCeil(tY, 0.02, 0.01),
        "quality_nan": robotize.quality(np.array([np.nan, 19.9, 20.0, 49.9, 50.0]), 20.0, 50.0),
        "side_warp_left": list(robotize.sideWarp(-tY, 1.0, 0.05, 0.01, 0.4)),
        "side_warp_empty": list(robotize.sideWarp(np.full(3, np.nan), -1.0, 0.05, 0.01, 0.4)),
        "resample": parity_metrics.resample(np.arange(4.0), tX, np.array([-1.0, 0.5, 2.5, 3.0, 4.0])),
        "relative": [tRel, tRot.as_quat(), tOk],
        "fmt": [trajio._fmt(tV) for tV in (np.nan, 3.0, -2.0, 0.1234567, 1e-9)],
    })


def test_compare_hand_without_overlap():
    tRef = {"t": np.arange(5.0)}
    tTest = {"t": np.arange(5.0) + 100.0}
    for tK in trajio.POSE:
        tRef[f"R_{tK}"] = np.ones(5)
        tTest[f"R_{tK}"] = np.ones(5)
    assert parity_metrics.compareHand(tRef, tTest, "R", -np.inf, np.inf, 0.5, 0.1) == {"error": "no overlap"}


def test_paths_of_examples_exist():
    assert DEMO.is_file() and REPLAY.is_file() and Path(robotize.__file__).is_file()
