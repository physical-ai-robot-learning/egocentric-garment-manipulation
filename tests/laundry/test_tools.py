"""Smoke and property tests for the utilities, on the synthetic example only.

Run:  PYTHONPATH=src python -m pytest -q tests/laundry      (or: PYTHONPATH=src python tests/laundry/test_tools.py)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from laundry_reference import gap_fill, parity_metrics, relative_se3, robotize, trajio

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "assets" / "laundry" / "examples" / "synthetic_bimanual_demo.csv"
REPLAY = ROOT / "assets" / "laundry" / "examples" / "synthetic_robot_replay.csv"


def test_gap_fill_flags_and_slerp():
    t = np.arange(0, 2, 0.1)
    p = np.stack([t, 0 * t, 0 * t], 1)
    q = Rotation.from_euler("z", (90 * t / 2)[:, None], degrees=True).as_quat()  # (N, 1): scipy>=1.18 rejects 1-D
    vis = np.ones_like(t)
    vis[5:9] = 0          # 0.5 s gap  -> filled
    vis[12:19] = 0        # 0.8 s gap  -> too long for max_gap 0.5
    p[vis == 0] = np.nan
    q[vis == 0] = np.nan
    p2, q2, flag = gap_fill.fillHand(t, p, q, vis, maxGapS=0.5)
    assert np.all(flag[5:9] == 1) and np.all(flag[12:19] == 2) and np.all(flag[:5] == 0)
    assert np.allclose(p2[5:9, 0], t[5:9])
    yaw = Rotation.from_quat(q2[5:9]).as_euler("xyz", degrees=True)[:, 2]
    assert np.allclose(yaw, 90 * t[5:9] / 2, atol=1e-6)
    assert np.all(np.isnan(p2[12:19]))


def test_robotize_properties():
    z = np.linspace(-0.05, 0.05, 101)
    zf = robotize.softFloor(z, 0.0, 0.01)
    assert np.all(zf > 0.0) and np.all(np.diff(zf) > 0)               # above floor, monotone
    assert abs(zf[-1] - z[-1]) < 1e-3                                  # far above: unchanged
    y = np.linspace(-0.3, 0.02, 200)                                   # right hand crossing midline
    y2, s = robotize.sideWarp(y, -1.0, sep=0.05, w=0.01, sMin=0.4)
    assert s < 1.0 and np.all(y2 < -0.025) and np.all(np.diff(y2) > 0)
    q = robotize.quality(np.array([5.0, 25.0, 80.0]), 20, 50)
    assert list(q) == [0, 1, 2]


def test_relative_se3_roundtrip():
    rng = np.random.default_rng(1)
    T = relative_se3.toMatrix(rng.normal(size=(50, 3)), Rotation.random(50, random_state=1).as_quat())
    T0 = T[7]
    assert np.allclose(relative_se3.fromRelative(relative_se3.toRelative(T, T0), T0), T)
    assert np.allclose(relative_se3.toRelative(T, T0)[7], np.eye(4))


def test_parity_recovers_lag_and_identity():
    d = trajio.load(DEMO)
    with tempfile.TemporaryDirectory() as tmp:
        filled = Path(tmp) / "f.csv"
        gap_fill.run([str(DEMO), str(filled)])
        res_self = parity_metrics.run([str(filled), str(filled)])
        assert res_self["R"]["lag_s"] == 0 and res_self["R"]["pos_rmse_mm"] < 1e-6
        robo = Path(tmp) / "r.csv"
        robotize.run([str(filled), str(robo)])
        res = parity_metrics.run([str(robo), str(REPLAY)])
        for h in ("R", "L"):
            assert abs(res[h]["lag_s"] - 0.25) <= 0.06       # generator delay
            assert res[h]["pos_rmse_mm"] < 10.0
    assert len(d["t"]) == 900


def test_cli_end_to_end():
    """The documented ``python -m laundry_reference.<module>`` commands."""
    with tempfile.TemporaryDirectory() as tmp:
        cmds = [
            ["-m", "laundry_reference.gap_fill", str(DEMO), f"{tmp}/f.csv"],
            ["-m", "laundry_reference.robotize", f"{tmp}/f.csv", f"{tmp}/r.csv"],
            ["-m", "laundry_reference.parity_metrics", f"{tmp}/r.csv", str(REPLAY), "--json", f"{tmp}/m.json"],
            ["-m", "laundry_reference.relative_se3", f"{tmp}/f.csv"],
            ["-m", "laundry_reference.plot_example", f"{tmp}/r.csv", f"{tmp}/p.png", "--robot", str(REPLAY)],
        ]
        for c in cmds:
            subprocess.run([sys.executable, *c], cwd=ROOT, check=True, capture_output=True)
        assert Path(f"{tmp}/p.png").stat().st_size > 10_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
