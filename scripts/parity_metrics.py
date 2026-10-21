"""Compare two time-stamped bimanual trajectories (e.g. human demo vs robot replay).

For each hand and an optional time window:

* both trajectories are resampled onto the reference time grid;
* both are made *relative* to their own first valid pose, so constant offsets
  between the two frames are not counted (see the caveat below);
* the test trajectory is shifted by the lag L* in [0, --max-lag] that minimises
  the mean squared position error, so a pure delay is not counted as error;
* reported: position RMSE / p95 / max [mm], rotation error median / p95 [deg]
  (geodesic angle between the relative rotations), amplitude ratio (RMS
  distance from the start, test over reference), the lag, and the fraction of
  frames that entered the comparison.

Caveat: relative comparison and lag search hide constant offsets and delays.
They answer "does the motion have the same shape?", not "is it in the right
place?". Absolute placement must be checked separately (e.g. against an
external camera or a tape measure).

Usage:
    python scripts/parity_metrics.py REF.csv TEST.csv [--t0 S --t1 S] [--max-lag 1.5] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajio  # noqa: E402


def resample(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.full((len(t_dst), x.shape[1]), np.nan)
    ok = np.all(np.isfinite(x), axis=1)
    if ok.sum() < 2:
        return out
    for j in range(x.shape[1]):
        out[:, j] = np.interp(t_dst, t_src[ok], x[ok, j], left=np.nan, right=np.nan)
    return out


def continuous_quat(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive valid samples lie in the same hemisphere."""
    q = q.copy()
    prev = None
    for i in range(len(q)):
        if not np.all(np.isfinite(q[i])):
            continue
        if prev is not None and np.dot(q[i], prev) < 0:
            q[i] = -q[i]
        prev = q[i]
    return q


def relative(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, Rotation | None, np.ndarray]:
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    if not ok.any():
        return p, None, ok
    i0 = np.flatnonzero(ok)[0]
    r0 = Rotation.from_quat(q[i0])
    qq = q.copy()
    qq[~ok] = [0, 0, 0, 1]
    r_rel = r0.inv() * Rotation.from_quat(qq / np.linalg.norm(qq, axis=1, keepdims=True))
    return p - p[i0], r_rel, ok


def compare_hand(ref: dict, test: dict, hand: str, t0: float, t1: float, max_lag: float,
                 lag_step: float) -> dict:
    t = ref["t"]
    win = (t >= t0) & (t <= t1)
    tr = t[win]
    pa, qa = trajio.position(ref, hand)[win], trajio.quat(ref, hand)[win]
    pa_rel, ra, oka = relative(pa, qa)

    best = None
    for lag in np.arange(0.0, max_lag + 1e-9, lag_step):
        pc = resample(test["t"], trajio.position(test, hand), tr + lag)
        ok = oka & np.all(np.isfinite(pc), axis=1)
        if ok.sum() < 10:
            continue
        i0 = np.flatnonzero(np.all(np.isfinite(pc), axis=1))[0]
        e = np.linalg.norm((pc - pc[i0])[ok] - pa_rel[ok], axis=1)
        mse = float(np.mean(e**2))
        if best is None or mse < best[0]:
            best = (mse, lag)
    if best is None:
        return {"error": "no overlap"}
    lag = best[1]

    pc = resample(test["t"], trajio.position(test, hand), tr + lag)
    qc = resample(test["t"], continuous_quat(trajio.quat(test, hand)), tr + lag)  # nlerp after normalisation
    pc_rel, rc, okc = relative(pc, qc)
    ok = oka & okc
    e = 1000.0 * np.linalg.norm(pc_rel[ok] - pa_rel[ok], axis=1)
    ang = np.degrees((ra[ok].inv() * rc[ok]).magnitude())
    amp_a = np.sqrt(np.mean(np.sum(pa_rel[ok] ** 2, axis=1)))
    amp_c = np.sqrt(np.mean(np.sum(pc_rel[ok] ** 2, axis=1)))
    return {
        "lag_s": round(float(lag), 3),
        "pos_rmse_mm": float(np.sqrt(np.mean(e**2))),
        "pos_p95_mm": float(np.percentile(e, 95)),
        "pos_max_mm": float(e.max()),
        "rot_median_deg": float(np.median(ang)),
        "rot_p95_deg": float(np.percentile(ang, 95)),
        "amplitude_ratio": float(amp_c / amp_a) if amp_a > 0 else float("nan"),
        "coverage": float(ok.mean()),
        "reference_range_mm": float(1000.0 * np.nanmax(np.linalg.norm(pa_rel, axis=1))),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref", help="reference trajectory CSV (e.g. human)")
    ap.add_argument("test", help="trajectory to evaluate (e.g. robot)")
    ap.add_argument("--t0", type=float, default=-np.inf)
    ap.add_argument("--t1", type=float, default=np.inf)
    ap.add_argument("--max-lag", type=float, default=1.5, help="lag search range [s]")
    ap.add_argument("--lag-step", type=float, default=0.02, help="lag search step [s]")
    ap.add_argument("--only-measured", action="store_true",
                    help="ignore reference frames that were not measured (visible == 0 or filled != 0)")
    ap.add_argument("--json", help="write the metrics to this JSON file")
    a = ap.parse_args(argv)

    ref, test = trajio.load(a.ref), trajio.load(a.test)
    res = {}
    for h in trajio.hands_in(ref):
        if f"{h}_x" not in test:
            continue
        if a.only_measured:
            bad = ref.get(f"{h}_visible", np.ones_like(ref["t"])) < 0.5
            if f"{h}_filled" in ref:
                bad |= ref[f"{h}_filled"] != 0
            for k in trajio.POSE:
                ref[f"{h}_{k}"] = np.where(bad, np.nan, ref[f"{h}_{k}"])
        res[h] = compare_hand(ref, test, h, a.t0, a.t1, a.max_lag, a.lag_step)
        r = res[h]
        if "error" in r:
            print(f"{h}: {r['error']}")
            continue
        print(f"{h}: lag {r['lag_s']:.2f} s  pos RMSE {r['pos_rmse_mm']:.1f} / p95 {r['pos_p95_mm']:.1f} mm  "
              f"rot median {r['rot_median_deg']:.1f} / p95 {r['rot_p95_deg']:.1f} deg  "
              f"amplitude {r['amplitude_ratio']:.3f}  coverage {100*r['coverage']:.0f} %")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
