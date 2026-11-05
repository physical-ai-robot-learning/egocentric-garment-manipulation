"""Golden records of the characterization tests: `checkGolden(name, value)` compares a value with golden/<name>.json.

LAUNDRY_DEPLOY_REGEN=1 rewrites the record instead (done once, on the code before the rewrite). Floats compare to a relative
1e-9; NaN equals NaN; numpy arrays, tuples and paths are normalised to JSON lists / strings first.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REGEN_ENV = "LAUNDRY_DEPLOY_REGEN"
REL_TOL = 1e-9
ABS_TOL = 1e-12
HOME = str(Path.home())


def normalise(value):
    if isinstance(value, dict):
        return {str(tKey): normalise(tValue) for tKey, tValue in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(tValue) for tValue in value]
    if isinstance(value, np.ndarray):
        return normalise(value.tolist())
    if type(value).__name__ == "Tensor":
        return normalise(value.detach().cpu().double().numpy())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, bytes):
        return "sha256:" + hashlib.sha256(value).hexdigest()
    if hasattr(value, "__fspath__"):
        return normalise(str(value))
    if isinstance(value, str) and value.startswith(HOME + "/"):
        return "~" + value[len(HOME):]                                # machine-independent: the user's home as ~
    if isinstance(value, argparse.Namespace):
        return normalise(vars(value))
    return value


def _same(got, want, where: str) -> None:
    tNum = (int, float)
    if isinstance(got, str) and isinstance(want, str) and got in ("nan", "inf", "-inf"):
        assert got == want, f"{where}: {got} != {want}"
        return
    if (
        (isinstance(got, float) or isinstance(want, float))
        and isinstance(got, tNum)
        and isinstance(want, tNum)
        and not isinstance(got, bool)
        and not isinstance(want, bool)
    ):
        assert math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{where}: {got} != {want}"
        return
    if isinstance(got, dict) and isinstance(want, dict):
        assert sorted(got) == sorted(want), f"{where}: keys differ {sorted(set(got) ^ set(want))}"
        for tKey in want:
            _same(got[tKey], want[tKey], f"{where}.{tKey}")
        return
    if isinstance(got, list) and isinstance(want, list):
        assert len(got) == len(want), f"{where}: length {len(got)} != {len(want)}"
        for tI, (tGot, tWant) in enumerate(zip(got, want)):
            _same(tGot, tWant, f"{where}[{tI}]")
        return
    assert got == want, f"{where}: {got!r} != {want!r}"


def checkGolden(name: str, value) -> None:
    tPath = GOLDEN_DIR / f"{name}.json"
    tGot = json.loads(json.dumps(normalise(value), ensure_ascii=False))
    if os.environ.get(REGEN_ENV) == "1":
        tPath.parent.mkdir(parents=True, exist_ok=True)
        tPath.write_text(json.dumps(tGot, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert tPath.exists(), f"no golden record {tPath} (run once with {REGEN_ENV}=1 on the reference code)"
    _same(tGot, json.loads(tPath.read_text(encoding="utf-8")), name)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
