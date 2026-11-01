# egogarment

Generic, sensor-free re-implementations of the trajectory stages of the pipeline (format, gap filling, relative SE(3),
robotization, parity metrics, plotting), runnable on the synthetic data in [`../../examples/`](../../examples).
They were written for this repository as small standalone versions; the code that ran on the real system is in
[`../visual_teleop/`](../visual_teleop/README.md), [`../rebot_prep/`](../rebot_prep/README.md) and [`../../scripts/`](../../README.md#3-components).

**Status:** current (reference implementations; numbers they produce describe the synthetic generator, not the real system)

Every subpackage below follows the same template. All tools read and write the trajectory CSV defined in
[`io/trajio.py`](io/trajio.py) (see [Codebase guide](../../README.md#6-codebase-guide)). Each command also exists as
`python scripts/<module>.py` (thin wrapper, no install needed) and, after `pip install -e .`, as an
`egogarment-…` console command.

---

## io

Reads and writes the wide, per-hand bimanual trajectory CSV that every other module uses.

**Status:** current

### Inputs → Outputs
CSV file ↔ `dict[str, np.ndarray]` keyed by column name (`t`, `R_x` … `L_visible`, plus any extra per-hand columns).

### Entry points
Library only: `trajio.load`, `trajio.save`, `trajio.position`, `trajio.quat`, `trajio.set_pose`, `trajio.hands_in`;
constants `HANDS = ("R", "L")`, `POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")`.

### Layout
* [`io/trajio.py`](io/trajio.py) — CSV format helpers (was `scripts/trajio.py`).

### Provenance
Written for this repository.

---

## pose

Turns measured per-frame jaw-tip poses into a gap-filled trajectory with per-frame provenance flags, and converts
absolute poses into the relative SE(3) learning contract.

**Status:** current (generic baseline; the real pipeline's pose source is described in the top-level README)

### Inputs → Outputs
* `gap_fill`: trajectory CSV with `visible` → CSV with short gaps filled (lerp or PCHIP position, slerp rotation) and
  a per-hand `<H>_filled` column (0 measured / 1 interpolated / 2 unfilled).
* `relative_se3`: trajectory CSV → chunks `T_rel = T_0^{-1} T(t)` and a round-trip check (printed).

### Entry points
```bash
python -m egogarment.pose.gap_fill SRC.csv DST.csv [--max-gap 1.0] [--cubic]     # egogarment-gap-fill
python -m egogarment.pose.relative_se3 SRC.csv [--chunk 1.0]                       # egogarment-relative-se3
```
Library: `gap_fill.runs`, `gap_fill.fill_hand`; `relative_se3.to_matrix`, `to_relative`, `from_relative`, `chunks`.

### Layout
* [`pose/gap_fill.py`](pose/gap_fill.py) — gap filling and flags (was `scripts/gap_fill.py`).
* [`pose/relative_se3.py`](pose/relative_se3.py) — relative SE(3) conversion (was `scripts/relative_se3.py`).

### Provenance
Written for this repository.

---

## retarget

Makes a human jaw-tip path more robot-feasible with continuous warps only (lateral side warp, soft table floor) and
records how far each frame was moved.

**Status:** current (illustrative defaults, not tuned for any specific robot)

### Inputs → Outputs
Trajectory CSV → CSV with warped `x, y, z`, the original path as `<H>_human_x/y/z`, per-frame `<H>_disp_mm` and
`<H>_quality` (0 / 1 / 2 at 20 / 50 mm by default).

### Entry points
```bash
python -m egogarment.retarget.robotize SRC.csv DST.csv [--sep 0.05] [--width 0.01] [--z-table 0] \
    [--clearance 0] [--s-min 0.4] [--q1 20] [--q2 50]                              # egogarment-robotize
```
Library: `robotize.robotize`, `softfloor`, `softceil`, `side_warp`, `quality`.

### Layout
* [`retarget/robotize.py`](retarget/robotize.py) — warps + displacement / quality (was `scripts/robotize.py`).

### Provenance
Written for this repository.

---

## evaluation

Compares two time-stamped bimanual trajectories (e.g. human vs. replay): relative to their first valid pose, after
a lag search.

**Status:** current

### Inputs → Outputs
Reference CSV + test CSV → per hand: lag, position RMSE / p95 / max [mm], rotation median / p95 [deg], amplitude
ratio, coverage, reference range (printed; optional JSON).

### Entry points
```bash
python -m egogarment.evaluation.parity_metrics REF.csv TEST.csv [--t0 S --t1 S] [--max-lag 1.5] \
    [--lag-step 0.02] [--only-measured] [--json out.json]                          # egogarment-parity-metrics
```
Library: `parity_metrics.compare_hand`, `resample`, `continuous_quat`, `relative`.

### Layout
* [`evaluation/parity_metrics.py`](evaluation/parity_metrics.py) — metrics (was `scripts/parity_metrics.py`).

### Provenance
Written for this repository. Metric definitions for the real replay are in
[`../../docs/results.md`](../../docs/results.md).

---

## viz

Plots a processed trajectory: top view, jaw-tip height over time with jaw / filled shading, robotization
displacement.

**Status:** current

### Inputs → Outputs
Robotized CSV (+ optional replay CSV) → PNG figure (e.g. [`../../docs/example_synthetic.png`](../../docs/example_synthetic.png)).

### Entry points
```bash
python -m egogarment.viz.plot_example SRC.csv DST.png [--robot REPLAY.csv]       # egogarment-plot-example
```

### Layout
* [`viz/plot_example.py`](viz/plot_example.py) — the figure (was `scripts/plot_example.py`).

### Provenance
Written for this repository.

---

## Other files

* [`cli.py`](cli.py) — console-command shims for `pyproject.toml` (call `main()` and discard its returned report).
* [`__init__.py`](__init__.py) — package docstring and version.
