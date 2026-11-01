"""Fill short marker dropouts in a bimanual trajectory and flag what was filled.

For each hand, runs of frames with ``visible == 0`` are treated as gaps.

* A gap no longer than ``--max-gap`` seconds, bounded by measured frames on
  both sides, is filled: position by linear interpolation (or shape-preserving
  cubic with ``--cubic``), orientation by spherical linear interpolation (slerp)
  between the two bounding measured rotations.
* Longer gaps and gaps touching the start/end are left NaN.

A per-hand column ``<H>_filled`` is written: 0 = measured, 1 = interpolated,
2 = unfilled (too long or open-ended). Downstream code should treat class 1 as
a label of lower confidence, never as a measurement.

In a real pipeline the rotation inside a gap would come from a gyroscope
anchored to the measured frames, and a visual-SLAM filler could be accepted
where it agrees with the markers; this tool implements only the generic,
sensor-free baseline.

Usage:
    python scripts/gap_fill.py examples/synthetic_bimanual_demo.csv out/filled.csv --max-gap 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajio  # noqa: E402


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_hand(t: np.ndarray, p: np.ndarray, q: np.ndarray, visible: np.ndarray,
              max_gap: float, cubic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = p.copy()
    q = q.copy()
    valid = (visible > 0.5) & np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    flag = np.where(valid, 0, 2).astype(float)
    if valid.sum() < 2:
        return p, q, flag
    vi = np.flatnonzero(valid)
    interp = PchipInterpolator(t[vi], p[vi], axis=0) if cubic else None
    for a, b in runs(~valid):
        if a == 0 or b == len(t):
            continue  # open-ended: no anchor on one side
        t0, t1 = t[a - 1], t[b]
        if t1 - t0 > max_gap:
            continue
        tt = t[a:b]
        if cubic:
            p[a:b] = interp(tt)
        else:
            w = ((tt - t0) / (t1 - t0))[:, None]
            p[a:b] = (1 - w) * p[a - 1] + w * p[b]
        key = Rotation.from_quat(np.stack([q[a - 1], q[b]]))
        q[a:b] = Slerp([t0, t1], key)(tt).as_quat()
        flag[a:b] = 1
    return p, q, flag


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-gap", type=float, default=1.0, help="longest gap to fill [s] (default 1.0)")
    ap.add_argument("--cubic", action="store_true", help="PCHIP position interpolation instead of linear")
    a = ap.parse_args(argv)

    data = trajio.load(a.src)
    report = {}
    for h in trajio.hands_in(data):
        p, q, flag = fill_hand(data["t"], trajio.position(data, h), trajio.quat(data, h),
                               data[f"{h}_visible"], a.max_gap, a.cubic)
        trajio.set_pose(data, h, p, q)
        data[f"{h}_filled"] = flag
        n = len(flag)
        report[h] = {"measured": float(np.mean(flag == 0)), "interpolated": float(np.mean(flag == 1)),
                     "unfilled": float(np.mean(flag == 2)), "frames": n}
        print(f"{h}: measured {100*report[h]['measured']:.1f} %  interpolated "
              f"{100*report[h]['interpolated']:.1f} %  unfilled {100*report[h]['unfilled']:.1f} %")
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    trajio.save(a.dst, data)
    return report


if __name__ == "__main__":
    main()
