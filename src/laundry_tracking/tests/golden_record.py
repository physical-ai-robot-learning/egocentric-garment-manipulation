"""Golden records for the characterization tests (test helper): compact, tolerance-aware summaries of arrays, tables and text.

summarize(value) turns a result into JSON-able data: float arrays -> shape, NaN count and three sums (plain, absolute, weighted by a
fixed cosine pattern); integer / bool arrays -> shape and a hash; strings and small texts verbatim; dicts / lists / tuples recursively.
checkGolden(name, summary) compares with tests/golden/<name>.json, or writes it when LT_RECORD_GOLDEN=1."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
RECORD_ENV = "LT_RECORD_GOLDEN"
REL_TOL = 1e-7          # same code, same machine: equal to the last bits; the tolerance only absorbs summation order in the sums
ABS_TOL = 1e-9
MAX_VERBATIM_CHARS = 60000


def _hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _floatSummary(tArr: np.ndarray) -> dict:
    tFlat = tArr.astype(np.float64).ravel()
    tFinite = np.where(np.isfinite(tFlat), tFlat, 0.0)
    tWeights = np.cos(np.arange(len(tFlat)) * 0.37)
    return dict(kind="f", shape=list(tArr.shape), nan=int(np.isnan(tFlat).sum()), inf=int(np.isinf(tFlat).sum()),
                sum=float(tFinite.sum()), abs=float(np.abs(tFinite).sum()), wsum=float((tFinite * tWeights).sum()))


def summarize(value):
    if isinstance(value, dict):
        return {str(tKey): summarize(tVal) for tKey, tVal in value.items()}
    if isinstance(value,
                  (list, tuple)) and not (value and all(isinstance(tV, (int, float, np.floating, np.integer)) for tV in value)):
        return [summarize(tV) for tV in value]
    if isinstance(value, (str, bytes)):
        return value if isinstance(value, str) else value.decode("utf-8", "replace")
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if hasattr(value, "as_quat"):                                        # scipy Rotation
        return summarize(np.asarray(value.as_quat()))
    tArr = np.asarray(value)
    if tArr.dtype.kind in "fc":
        if tArr.ndim == 0:
            return dict(kind="f", value=float(tArr))
        return _floatSummary(np.real(tArr))
    if tArr.dtype.kind in "iub":
        if tArr.ndim == 0:
            return dict(kind="i", value=int(tArr))
        return dict(kind="i", shape=list(tArr.shape), hash=_hash(np.ascontiguousarray(tArr.astype(np.int64)).tobytes()))
    if tArr.dtype.kind in "US":
        return dict(kind="s", shape=list(tArr.shape), value=[str(tV) for tV in tArr.ravel()])
    return dict(kind="o", value=repr(value))


def _close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))
    return a == b


def diff(expected, actual, path: str = "") -> list[str]:
    """human-readable differences between two summaries (empty = equal)"""
    if isinstance(expected, dict) and isinstance(actual, dict):
        tOut = []
        for tKey in sorted(set(expected) | set(actual)):
            if tKey not in expected or tKey not in actual:
                tOut.append(f"{path}/{tKey}: only in {'actual' if tKey in actual else 'expected'}")
                continue
            tOut += diff(expected[tKey], actual[tKey], f"{path}/{tKey}")
        return tOut
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        tOut = []
        for tI, (tE, tA) in enumerate(zip(expected, actual)):
            tOut += diff(tE, tA, f"{path}[{tI}]")
        return tOut
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
        return [] if _close(float(expected), float(actual)) else [f"{path}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{path}: {str(expected)[:300]!r} != {str(actual)[:300]!r}"]


def _jsonSafe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if isinstance(value, dict):
        return {tKey: _jsonSafe(tVal) for tKey, tVal in value.items()}
    if isinstance(value, list):
        return [_jsonSafe(tV) for tV in value]
    return value


def _jsonRestore(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {tKey: _jsonRestore(tVal) for tKey, tVal in value.items()}
    if isinstance(value, list):
        return [_jsonRestore(tV) for tV in value]
    return value


def dumps(summary) -> str:
    return json.dumps(_jsonSafe(summary), indent=1, ensure_ascii=False, sort_keys=True)


def loads(text: str):
    return _jsonRestore(json.loads(text))


def checkGolden(name: str, summary) -> None:
    tPath = GOLDEN_DIR / f"{name}.json"
    tSummary = loads(dumps(summary))                                    # same JSON round trip as the stored file
    if os.environ.get(RECORD_ENV) == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        tPath.write_text(dumps(tSummary))
        return
    tExpected = loads(tPath.read_text())
    tDiff = diff(tExpected, tSummary)
    assert not tDiff, f"{name}: {len(tDiff)} differences\n" + "\n".join(tDiff[:60])


def fileSummary(path: Path):
    """summary of one output file by type: npz arrays, csv / json / txt verbatim (when small), anything else by size"""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as tZ:
            return {tKey: summarize(tZ[tKey]) for tKey in tZ.files}
    if path.suffix == ".npy":
        return summarize(np.load(path, allow_pickle=True))
    if path.suffix in (".csv", ".json", ".txt", ".jsonl"):
        tText = path.read_text()
        return tText if len(tText) <= MAX_VERBATIM_CHARS else dict(kind="text", hash=_hash(tText.encode()), chars=len(tText))
    return dict(kind="file", nonempty=path.stat().st_size > 0)
