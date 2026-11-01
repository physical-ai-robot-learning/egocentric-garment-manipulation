"""Plot a (gap-filled, robotized) bimanual trajectory: top view, height over time, displacement.

Usage:
    python scripts/plot_example.py out/robotized.csv out/example.png [--robot examples/synthetic_robot_replay.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajio  # noqa: E402

COL = {"R": "#c0392b", "L": "#1f5fa8"}
NAME = {"R": "right", "L": "left"}


def shade(ax, t, mask, color, alpha, label=None):
    first = True
    for a, b in _runs(mask):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=alpha, lw=0,
                   label=label if first else None)
        first = False


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--robot", help="optional replay CSV to overlay")
    a = ap.parse_args(argv)
    d = trajio.load(a.src)
    r = trajio.load(a.robot) if a.robot else None
    t = d["t"]
    hands = trajio.hands_in(d)

    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(len(hands) + 1, 2, width_ratios=[1, 1.6])
    ax_top = fig.add_subplot(gs[:, 0])
    for h in hands:
        hx, hy = d.get(f"{h}_human_x", d[f"{h}_x"]), d.get(f"{h}_human_y", d[f"{h}_y"])
        ax_top.plot(hy, hx, color="0.7", lw=4, solid_capstyle="round",
                    label="human (measured + filled)" if h == "R" else None)
        ax_top.plot(d[f"{h}_y"], d[f"{h}_x"], color=COL[h], lw=1.4, label=f"{NAME[h]}: robotized")
        if r is not None:
            ax_top.plot(r[f"{h}_y"], r[f"{h}_x"], color=COL[h], lw=0.8, ls="--",
                        label=f"{NAME[h]}: replay")
    ax_top.axvline(0, color="k", lw=0.6, ls=":")
    ax_top.invert_xaxis()  # +y (left) drawn on the left, as seen from behind the robot
    ax_top.set_xlabel("y [m]  (+ = left)")
    ax_top.set_ylabel("x [m]  (forward)")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.set_title("Top view (world frame)")
    ax_top.legend(fontsize=7, loc="lower center")

    for k, h in enumerate(hands):
        ax = fig.add_subplot(gs[k, 1])
        hz = d.get(f"{h}_human_z", d[f"{h}_z"])
        ax.plot(t, 1000 * hz, color="0.6", lw=2.5, label="human")
        ax.plot(t, 1000 * d[f"{h}_z"], color=COL[h], lw=1.2, label="robotized")
        if r is not None:
            ax.plot(r["t"], 1000 * r[f"{h}_z"], color="k", lw=0.7, ls="--", label="replay")
        shade(ax, t, d[f"{h}_jaw"] < 0.5, "#f39c12", 0.18, "jaw closed")
        if f"{h}_filled" in d:
            shade(ax, t, d[f"{h}_filled"] == 1, "#8e44ad", 0.25, "interpolated")
            shade(ax, t, d[f"{h}_filled"] == 2, "#000000", 0.15, "unfilled gap")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"{NAME[h]} z [mm]")
        ax.set_title(f"{NAME[h].capitalize()} jaw-tip height", fontsize=9)
        ax.legend(fontsize=6, ncol=3, loc="upper right")

    ax = fig.add_subplot(gs[len(hands), 1])
    for h in hands:
        if f"{h}_disp_mm" in d:
            ax.plot(t, d[f"{h}_disp_mm"], color=COL[h], lw=1, label=NAME[h])
    for y in (20, 50):
        ax.axhline(y, color="k", lw=0.6, ls="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("displacement [mm]")
    ax.set_title("Robotization displacement per frame (dashed: quality class boundaries)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("SYNTHETIC example - not real data", color="#c0392b", fontsize=10)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.dst, dpi=130)
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
