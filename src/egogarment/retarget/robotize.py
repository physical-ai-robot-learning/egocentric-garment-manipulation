"""Make a human bimanual jaw-tip path more robot-feasible with smooth warps only.

The human path is never clipped. Two continuous warps are applied per hand:

1. **Side warp (lateral, y).** Each arm should stay on its own side of the
   body midline with a minimum hand separation ``d``. For the right hand
   (y < 0 side) let ``y_o`` / ``y_i`` be the 1st / 99th percentile of y (outer /
   inner extent). If ``y_i`` already respects ``-d/2`` nothing changes;
   otherwise the path is compressed affinely toward the outer side,

       s  = clip((-d/2 - y_o) / (y_i - y_o), s_min, 1)
       y' = softceil(y_o + s (y - y_o), -d/2, w)

   so the shape is kept and only the residual excursion is bent smoothly by a
   softplus ceiling. The left hand is mirrored.

2. **Table floor (z).** z' = softfloor(z, z_table + clearance, w), with
   softfloor(z, f, w) = f + w log(1 + exp((z - f) / w)). Note that points
   exactly at the floor are lifted by w ln 2.

For every frame the displacement ``|p' - p|`` (mm) and a quality class are
stored: 0 if displacement < q1, 1 if < q2, else 2. These labels are meant as
per-frame weights for later learning: large shifts during contact phases are
the harmful ones.

All numeric defaults here are illustrative, not tuned for any specific robot.

Usage:
    python scripts/robotize.py out/filled.csv out/robotized.csv --sep 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajio  # noqa: E402


def softfloor(z: np.ndarray, f: float, w: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):  # NaN (unfilled) frames stay NaN
        return f + w * np.logaddexp(0.0, (z - f) / w)


def softceil(y: np.ndarray, c: float, w: float) -> np.ndarray:
    return -softfloor(-y, -c, w)


def side_warp(y: np.ndarray, side: float, sep: float, w: float, s_min: float) -> tuple[np.ndarray, float]:
    """side = -1 for the right hand (y<0 half), +1 for the left hand."""
    ym = -side * y  # mirror so that the hand's own side is y < 0
    fin = np.isfinite(ym)
    if not fin.any():
        return y.copy(), 1.0
    y_o, y_i = np.percentile(ym[fin], [1, 99])
    lim = -sep / 2
    if y_i <= lim or y_i - y_o < 1e-9:
        s = 1.0
    else:
        s = float(np.clip((lim - y_o) / (y_i - y_o), s_min, 1.0))
    out = softceil(y_o + s * (ym - y_o), lim, w)
    return -side * out, s


def quality(disp_mm: np.ndarray, q1: float, q2: float) -> np.ndarray:
    q = np.full(disp_mm.shape, 2.0)
    q[disp_mm < q2] = 1.0
    q[disp_mm < q1] = 0.0
    q[~np.isfinite(disp_mm)] = np.nan
    return q


def robotize(data: dict, sep: float = 0.05, width: float = 0.01, z_table: float = 0.0,
             clearance: float = 0.0, s_min: float = 0.4, q1_mm: float = 20.0,
             q2_mm: float = 50.0) -> dict:
    out = {k: v.copy() for k, v in data.items()}
    for h in trajio.hands_in(data):
        side = -1.0 if h == "R" else 1.0
        p = trajio.position(data, h)
        p2 = p.copy()
        p2[:, 1], s = side_warp(p[:, 1], side, sep, width, s_min)
        p2[:, 2] = softfloor(p[:, 2], z_table + clearance, width)
        trajio.set_pose(out, h, p2)
        for i, k in enumerate("xyz"):
            out[f"{h}_human_{k}"] = p[:, i]
        disp = 1000.0 * np.linalg.norm(p2 - p, axis=1)
        out[f"{h}_disp_mm"] = disp
        out[f"{h}_quality"] = quality(disp, q1_mm, q2_mm)
        out.setdefault("_scale", {})[h] = s
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--sep", type=float, default=0.05, help="minimum hand separation across the midline [m]")
    ap.add_argument("--width", type=float, default=0.01, help="softplus width w [m]")
    ap.add_argument("--z-table", type=float, default=0.0, help="table height in the world frame [m]")
    ap.add_argument("--clearance", type=float, default=0.0, help="extra floor clearance [m]")
    ap.add_argument("--s-min", type=float, default=0.4, help="strongest allowed lateral compression")
    ap.add_argument("--q1", type=float, default=20.0, help="quality 0/1 boundary [mm]")
    ap.add_argument("--q2", type=float, default=50.0, help="quality 1/2 boundary [mm]")
    a = ap.parse_args(argv)

    data = trajio.load(a.src)
    out = robotize(data, a.sep, a.width, a.z_table, a.clearance, a.s_min, a.q1, a.q2)
    scales = out.pop("_scale")
    report = {}
    for h in trajio.hands_in(data):
        d = out[f"{h}_disp_mm"]
        q = out[f"{h}_quality"]
        fin = np.isfinite(d)
        report[h] = {"lateral_scale": scales[h], "disp_median_mm": float(np.median(d[fin])),
                     "disp_p95_mm": float(np.percentile(d[fin], 95)), "disp_max_mm": float(d[fin].max()),
                     "quality_share": [float(np.mean(q[fin] == c)) for c in (0, 1, 2)]}
        r = report[h]
        print(f"{h}: lateral scale {r['lateral_scale']:.2f}  displacement median {r['disp_median_mm']:.1f} / "
              f"p95 {r['disp_p95_mm']:.1f} / max {r['disp_max_mm']:.1f} mm  quality 0/1/2 = "
              + "/".join(f"{100*x:.0f}%" for x in r["quality_share"]))
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    trajio.save(a.dst, out)
    return report


if __name__ == "__main__":
    main()
