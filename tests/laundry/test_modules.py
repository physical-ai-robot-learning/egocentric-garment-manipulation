"""Per-module regression tests on the synthetic example, imported via the ``laundry_reference`` package.

``expected_synthetic.json`` was recorded from the pre-refactor code on ``main`` (the flat ``scripts/*.py`` modules,
numpy 2.0.2 / scipy 1.13.1) by running the same calls as below; the package must reproduce it. Nothing in it was
hand-written.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from laundry_reference import gap_fill, parity_metrics, plot_example, relative_se3, robotize, trajio

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "assets" / "laundry" / "examples" / "synthetic_bimanual_demo.csv"
REPLAY = ROOT / "assets" / "laundry" / "examples" / "synthetic_robot_replay.csv"
EXPECTED = json.loads((Path(__file__).with_name("expected_synthetic.json")).read_text())
REL = 1e-6


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _approx(x):
    if isinstance(x, dict):
        return {k: _approx(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_approx(v) for v in x]
    if isinstance(x, float):
        return pytest.approx(x, rel=REL, abs=1e-9)
    return x


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    f, r = d / "f.csv", d / "r.csv"
    out = {
        "gap_fill": _quiet(gap_fill.run, [str(DEMO), str(f)]),
        "gap_fill_cubic_max05": _quiet(gap_fill.run, [str(DEMO), str(d / "fc.csv"), "--cubic", "--max-gap", "0.5"]),
        "robotize": _quiet(robotize.run, [str(f), str(r)]),
        "parity_robotized_vs_replay": _quiet(parity_metrics.run, [str(r), str(REPLAY)]),
        "parity_filled_vs_replay": _quiet(parity_metrics.run, [str(f), str(REPLAY)]),
        "relative_se3_worst": _quiet(relative_se3.run, [str(f)]),
    }
    return d, f, r, out


# --- io --------------------------------------------------------------------------------------------------------

def test_io_load_save_roundtrip(tmp_path):
    d = trajio.load(DEMO)
    assert trajio.handsIn(d) == ["R", "L"]
    assert len(d["t"]) == 900
    assert trajio.position(d, "R").shape == (900, 3) and trajio.quat(d, "L").shape == (900, 4)
    out = tmp_path / "copy.csv"
    trajio.save(out, d)
    assert out.read_bytes() == DEMO.read_bytes()  # the committed CSV is exactly what save() writes


def test_io_constants():
    assert trajio.HANDS == ("R", "L")
    assert trajio.POSE == ("x", "y", "z", "qx", "qy", "qz", "qw")


# --- pose ------------------------------------------------------------------------------------------------------

def test_pose_gap_fill_report(pipeline):
    _, f, _, out = pipeline
    assert out["gap_fill"] == _approx(EXPECTED["gap_fill"])
    assert out["gap_fill_cubic_max05"] == _approx(EXPECTED["gap_fill_cubic_max05"])
    d = trajio.load(f)
    assert list(d) == EXPECTED["filled_columns"]
    assert {k: int(np.isnan(v).sum()) for k, v in d.items()} == EXPECTED["filled_nan_count"]
    assert {k: float(np.nansum(v)) for k, v in d.items()} == _approx(EXPECTED["filled_sum"])


def test_pose_gap_fill_runs():
    assert gap_fill.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 3), (4, 5), (7, 8)]


def test_pose_relative_se3_roundtrip_on_example(pipeline):
    _, f, _, out = pipeline
    assert out["relative_se3_worst"] < 1e-12
    d = trajio.load(f)
    p, q = trajio.position(d, "R"), trajio.quat(d, "R")
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    T = relative_se3.toMatrix(p[ok], q[ok])
    ch = relative_se3.chunks(T, d["t"][ok], 1.0)
    assert len(ch) == 30  # printed by the CLI on main: "R: 30 chunks of 1.0 s"
    for i, Tr in ch:
        assert np.allclose(Tr[0], np.eye(4))


# --- retarget --------------------------------------------------------------------------------------------------

def test_retarget_robotize_report(pipeline):
    _, _, r, out = pipeline
    assert out["robotize"] == _approx(EXPECTED["robotize"])
    d = trajio.load(r)
    assert {k: float(np.nansum(v)) for k, v in d.items()} == _approx(EXPECTED["robotized_sum"])


def test_retarget_defaults_unchanged():
    import inspect
    sig = inspect.signature(robotize.robotize)
    assert {k: v.default for k, v in sig.parameters.items() if k != "data"} == {
        "sep": 0.05, "width": 0.01, "zTable": 0.0, "clearance": 0.0, "sMin": 0.4, "q1Mm": 20.0, "q2Mm": 50.0}


# --- evaluation ------------------------------------------------------------------------------------------------

def test_evaluation_parity_metrics(pipeline):
    _, _, _, out = pipeline
    assert out["parity_robotized_vs_replay"] == _approx(EXPECTED["parity_robotized_vs_replay"])
    assert out["parity_filled_vs_replay"] == _approx(EXPECTED["parity_filled_vs_replay"])


def test_evaluation_continuous_quat():
    q = np.array([[0, 0, 0, 1.0], [0, 0, 0, -1.0], [np.nan] * 4, [0, 0, 0, -1.0]])
    c = parity_metrics.continuousQuat(q)
    assert np.allclose(c[[0, 1, 3], 3], 1.0) and np.all(np.isnan(c[2]))


# --- viz -------------------------------------------------------------------------------------------------------

def test_viz_plot_example(pipeline, tmp_path):
    _, _, r, _ = pipeline
    png = tmp_path / "p.png"
    _quiet(plot_example.main, [str(r), str(png), "--robot", str(REPLAY)])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 10_000


# --- examples generator ----------------------------------------------------------------------------------------

def test_make_synthetic_reproduces_committed_examples(tmp_path, monkeypatch):
    tSpec = importlib.util.spec_from_file_location("make_synthetic", ROOT / "assets" / "laundry" / "examples" / "make_synthetic.py")
    make_synthetic = importlib.util.module_from_spec(tSpec)
    tSpec.loader.exec_module(make_synthetic)
    monkeypatch.setattr(make_synthetic, "HERE", tmp_path)
    _quiet(make_synthetic.main)
    for name in ("synthetic_bimanual_demo.csv", "synthetic_robot_replay.csv"):
        new, old = trajio.load(tmp_path / name), trajio.load(ROOT / "assets" / "laundry" / "examples" / name)
        assert list(new) == list(old)
        for k in old:
            np.testing.assert_allclose(new[k], old[k], rtol=0, atol=1.5e-6, equal_nan=True)
    with np.load(tmp_path / "synthetic_bimanual_demo.npz") as a, \
            np.load(ROOT / "assets" / "laundry" / "examples" / "synthetic_bimanual_demo.npz") as b:
        assert sorted(a.files) == sorted(b.files)
        for k in b.files:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-12, equal_nan=True)


# --- module entry points ---------------------------------------------------------------------------------------

def test_python_dash_m_entry_points(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    cmds = [
        ["laundry_reference.gap_fill", str(DEMO), f"{tmp_path}/f.csv"],
        ["laundry_reference.robotize", f"{tmp_path}/f.csv", f"{tmp_path}/r.csv"],
        ["laundry_reference.parity_metrics", f"{tmp_path}/r.csv", str(REPLAY), "--json", f"{tmp_path}/m.json"],
        ["laundry_reference.relative_se3", f"{tmp_path}/f.csv"],
        ["laundry_reference.plot_example", f"{tmp_path}/r.csv", f"{tmp_path}/p.png", "--robot", str(REPLAY)],
    ]
    for c in cmds:
        subprocess.run([sys.executable, "-m", *c], cwd=tmp_path, env=env, check=True, capture_output=True)
    m = json.loads(Path(f"{tmp_path}/m.json").read_text())
    assert m == _approx(EXPECTED["parity_robotized_vs_replay"])
