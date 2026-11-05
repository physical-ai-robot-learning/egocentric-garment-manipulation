"""Characterization snapshots: values recorded once from the pre-refactor code, compared on every later run.

`checkSnapshot(name, value)` compares `value` with the entry `name` of the package's `expected_characterization.json`.
With LAUNDRY_CHAR_RECORD=1 in the environment it (re)writes that entry instead; do that only on the code the snapshot is
meant to pin (the commit "tests: characterization of <pkg>"), never to make a refactor pass.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

RECORD_ENV = "LAUNDRY_CHAR_RECORD"
EXPECTED_PATH = Path(__file__).with_name("expected_characterization.json")
# Same code, same inputs: every number must come back to the last bits. Torch inference on several CPU threads may
# reorder sums, so those calls pass a looser tolerance explicitly.
# 1e-12 on the recording machine; another machine with the same package versions (CI) differs in the last bits of
# the IK / least-squares outputs, so snapshots are compared at 1e-6 relative.
EXACT_REL = 1e-6
EXACT_ABS = 1e-9
NAN_TOKEN = "nan"


def plain(value):
    """JSON-able copy: arrays -> lists, numpy scalars -> Python, NaN -> NAN_TOKEN, tuples -> lists, Paths -> str."""
    if isinstance(value, dict):
        return {str(tKey): plain(tVal) for tKey, tVal in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(tVal) for tVal in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return NAN_TOKEN
    return value


def _approx(expected, rel: float, abs_: float):
    if isinstance(expected, dict):
        return {tKey: _approx(tVal, rel, abs_) for tKey, tVal in expected.items()}
    if isinstance(expected, list):
        return [_approx(tVal, rel, abs_) for tVal in expected]
    if isinstance(expected, float):
        return pytest.approx(expected, rel=rel, abs=abs_)
    return expected


def checkSnapshot(name: str, value, rel: float = EXACT_REL, abs_: float = EXACT_ABS, expectedPath: Path = EXPECTED_PATH) -> None:
    tPlain = plain(value)
    tAll = json.loads(expectedPath.read_text()) if expectedPath.exists() else {}
    if os.environ.get(RECORD_ENV):
        tAll[name] = tPlain
        tLines = [f"{json.dumps(tKey)}: {json.dumps(tAll[tKey], ensure_ascii=False)}" for tKey in sorted(tAll)]
        expectedPath.write_text("{\n" + ",\n".join(tLines) + "\n}\n")
        return
    assert name in tAll, f"no recorded snapshot {name!r} in {expectedPath}"
    assert tPlain == _approx(tAll[name], rel, abs_)


def _runAsMain(target: str, isPath: bool) -> None:
    with warnings.catch_warnings():
        # runpy warns when the module was already imported by the test; running it again as __main__ is the point.
        warnings.simplefilter("ignore", RuntimeWarning)
        if isPath:
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)


class ParsedArgs(Exception):
    """Raised by the patched ArgumentParser.parse_args to hand the parsed Namespace back to the test."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__("parsed")
        self.mNamespace = namespace


def parsedArgs(monkeypatch, target: str, argv: list[str], isPath: bool = False) -> dict:
    """Run module (or script path) `target` as __main__ with `argv` and return vars() of the first parse_args result."""
    tOriginal = argparse.ArgumentParser.parse_args

    def _capture(self, args=None, namespace=None):
        raise ParsedArgs(tOriginal(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    monkeypatch.setattr(sys, "argv", [target, *argv])
    try:
        _runAsMain(target, isPath)
    except ParsedArgs as tParsed:
        return vars(tParsed.mNamespace)
    raise AssertionError(f"{target} never parsed its arguments")


def helpExitCode(monkeypatch, target: str, argv: list[str], isPath: bool = False) -> int:
    """Exit status of `target --help` (argparse exits 0 after printing the usage)."""
    monkeypatch.setattr(sys, "argv", [target, *argv, "--help"])
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _runAsMain(target, isPath)
        except SystemExit as tExit:
            return int(tExit.code or 0)
    raise AssertionError(f"{target} --help did not exit")


def runMain(monkeypatch, target: str, argv: list[str], isPath: bool = False) -> tuple[str, int]:
    """(stdout, exit status) of running `target` as __main__ with `argv`."""
    monkeypatch.setattr(sys, "argv", [target, *argv])
    tOut = io.StringIO()
    tCode = 0
    with contextlib.redirect_stdout(tOut):
        try:
            _runAsMain(target, isPath)
        except SystemExit as tExit:
            tCode = tExit.code if isinstance(tExit.code, int) else (0 if tExit.code is None else 1)
    return tOut.getvalue(), tCode
