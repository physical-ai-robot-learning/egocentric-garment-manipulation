# laundry_reference

Generic, sensor-free re-implementations of the trajectory stages of the pipeline (format, gap filling, relative SE(3),
robotization, parity metrics, plotting), runnable on the synthetic data in
[`../../assets/laundry/examples/`](../../assets/laundry/examples). They were written for this repository as small standalone
versions; the code that ran on the real system is in `laundry_shared.visual_teleop`, `laundry_shared.rebot_prep` and
[`../../scripts/laundry/`](../../scripts/laundry).

**Status:** current (reference implementations; numbers they produce describe the synthetic generator, not the real system)

One flat module per component (OA style; the former sub-package path is given under each). All tools read and write the
trajectory CSV defined in [`trajio.py`](trajio.py). Each command runs as `python -m laundry_reference.<module>` and, after
`pip install -e .`, as a `laundry-reference-…` console command ([`cli.py`](cli.py)). Every module with a command line has
`main(argv) -> int` (exit status) and `run(argv)`, which does the same and returns the report (dict / float) for programmatic use.

---

## trajio (was `io/trajio.py`)

Reads and writes the wide, per-hand bimanual trajectory CSV that every other module uses.
CSV file ↔ `dict[str, np.ndarray]` keyed by column name (`t`, `R_x` … `L_visible`, plus any extra per-hand columns).
Library only: `load`, `save`, `position`, `quat`, `setPose`, `handsIn`; constants `HANDS = ("R", "L")`,
`POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")`.

## gap_fill (was `pose/gap_fill.py`)

Trajectory CSV with `visible` → CSV with short gaps filled (lerp or PCHIP position, slerp rotation) and a per-hand
`<H>_filled` column (0 measured / 1 interpolated / 2 unfilled).
```bash
python -m laundry_reference.gap_fill SRC.csv DST.csv [--max-gap 1.0] [--cubic]           # laundry-reference-gap-fill
```
Library: `runs`, `fillHand`.

## relative_se3 (was `pose/relative_se3.py`)

Trajectory CSV → chunks `T_rel = T_0^{-1} T(t)` and a round-trip check (printed).
```bash
python -m laundry_reference.relative_se3 SRC.csv [--chunk 1.0]                             # laundry-reference-relative-se3
```
Library: `toMatrix`, `toRelative`, `fromRelative`, `chunks`.

## robotize (was `retarget/robotize.py`)

Makes a human jaw-tip path more robot-feasible with continuous warps only (lateral side warp, soft table floor) and records how
far each frame was moved: CSV with warped `x, y, z`, the original path as `<H>_human_x/y/z`, per-frame `<H>_disp_mm` and
`<H>_quality` (0 / 1 / 2 at 20 / 50 mm by default). Illustrative defaults, not tuned for any specific robot.
```bash
python -m laundry_reference.robotize SRC.csv DST.csv [--sep 0.05] [--width 0.01] [--z-table 0] \
    [--clearance 0] [--s-min 0.4] [--q1 20] [--q2 50]                                      # laundry-reference-robotize
```
Library: `robotize`, `softFloor`, `softCeil`, `sideWarp`, `quality`.

## parity_metrics (was `evaluation/parity_metrics.py`)

Reference CSV + test CSV → per hand: lag, position RMSE / p95 / max [mm], rotation median / p95 [deg], amplitude ratio,
coverage, reference range (printed; optional JSON), relative to their first valid pose, after a lag search.
```bash
python -m laundry_reference.parity_metrics REF.csv TEST.csv [--t0 S --t1 S] [--max-lag 1.5] \
    [--lag-step 0.02] [--only-measured] [--json out.json]                                  # laundry-reference-parity-metrics
```
Library: `compareHand`, `resample`, `continuousQuat`, `relative`.

## plot_example (was `viz/plot_example.py`)

Robotized CSV (+ optional replay CSV) → PNG figure: top view, jaw-tip height over time with jaw / filled shading,
robotization displacement.
```bash
python -m laundry_reference.plot_example SRC.csv DST.png [--robot REPLAY.csv]             # laundry-reference-plot-example
```
Library: `plotTrajectory`, `shade`.

## Tests

`src/laundry_reference/tests/` (command lines, console scripts, helpers) and `tests/laundry/test_modules.py`,
`tests/laundry/test_tools.py` (regression against `tests/laundry/expected_synthetic.json`).
