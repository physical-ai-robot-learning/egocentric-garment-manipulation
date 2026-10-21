"""Smoke and property tests for the utilities, on the synthetic example only.

Run:  python -m pytest -q tests      (or: python tests/test_tools.py)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gap_fill  # noqa: E402
import parity_metrics  # noqa: E402
import relative_se3  # noqa: E402
import robotize  # noqa: E402
import trajio  # noqa: E402

DEMO = ROOT / "examples" / "synthetic_bimanual_demo.csv"
REPLAY = ROOT / "examples" / "synthetic_robot_replay.csv"


def test_gap_fill_flags_and_slerp():
    t = np.arange(0, 2, 0.1)
    p = np.stack([t, 0 * t, 0 * t], 1)
    q = Rotation.from_euler("z", 90 * t / 2, degrees=True).as_quat()
    vis = np.ones_like(t)
    vis[5:9] = 0          # 0.5 s gap  -> filled
    vis[12:19] = 0        # 0.8 s gap  -> too long for max_gap 0.5
    p[vis == 0] = np.nan
    q[vis == 0] = np.nan
    p2, q2, flag = gap_fill.fill_hand(t, p, q, vis, max_gap=0.5)
    assert np.all(flag[5:9] == 1) and np.all(flag[12:19] == 2) and np.all(flag[:5] == 0)
    assert np.allclose(p2[5:9, 0], t[5:9])
    yaw = Rotation.from_quat(q2[5:9]).as_euler("xyz", degrees=True)[:, 2]
    assert np.allclose(yaw, 90 * t[5:9] / 2, atol=1e-6)
    assert np.all(np.isnan(p2[12:19]))


def test_robotize_properties():
    z = np.linspace(-0.05, 0.05, 101)
    zf = robotize.softfloor(z, 0.0, 0.01)
    assert np.all(zf > 0.0) and np.all(np.diff(zf) > 0)               # above floor, monotone
    assert abs(zf[-1] - z[-1]) < 1e-3                                  # far above: unchanged
    y = np.linspace(-0.3, 0.02, 200)                                   # right hand crossing midline
    y2, s = robotize.side_warp(y, -1.0, sep=0.05, w=0.01, s_min=0.4)
    assert s < 1.0 and np.all(y2 < -0.025) and np.all(np.diff(y2) > 0)
    q = robotize.quality(np.array([5.0, 25.0, 80.0]), 20, 50)
    assert list(q) == [0, 1, 2]


def test_relative_se3_roundtrip():
    rng = np.random.default_rng(1)
    T = relative_se3.to_matrix(rng.normal(size=(50, 3)), Rotation.random(50, random_state=1).as_quat())
    T0 = T[7]
    assert np.allclose(relative_se3.from_relative(relative_se3.to_relative(T, T0), T0), T)
    assert np.allclose(relative_se3.to_relative(T, T0)[7], np.eye(4))


def test_parity_recovers_lag_and_identity():
    d = trajio.load(DEMO)
    with tempfile.TemporaryDirectory() as tmp:
        filled = Path(tmp) / "f.csv"
        gap_fill.main([str(DEMO), str(filled)])
        res_self = parity_metrics.main([str(filled), str(filled)])
        assert res_self["R"]["lag_s"] == 0 and res_self["R"]["pos_rmse_mm"] < 1e-6
        robo = Path(tmp) / "r.csv"
        robotize.main([str(filled), str(robo)])
        res = parity_metrics.main([str(robo), str(REPLAY)])
        for h in ("R", "L"):
            assert abs(res[h]["lag_s"] - 0.25) <= 0.06       # generator delay
            assert res[h]["pos_rmse_mm"] < 10.0
    assert len(d["t"]) == 900


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cmds = [
            ["scripts/gap_fill.py", str(DEMO), f"{tmp}/f.csv"],
            ["scripts/robotize.py", f"{tmp}/f.csv", f"{tmp}/r.csv"],
            ["scripts/parity_metrics.py", f"{tmp}/r.csv", str(REPLAY), "--json", f"{tmp}/m.json"],
            ["scripts/relative_se3.py", f"{tmp}/f.csv"],
            ["scripts/plot_example.py", f"{tmp}/r.csv", f"{tmp}/p.png", "--robot", str(REPLAY)],
        ]
        for c in cmds:
            subprocess.run([sys.executable, *c], cwd=ROOT, check=True, capture_output=True)
        assert Path(f"{tmp}/p.png").stat().st_size > 10_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
