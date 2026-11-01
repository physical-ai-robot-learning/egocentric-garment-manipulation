# report

Technical Report v0.1 (10-27, current) and v0 (10-21), and the short technical note: PDF, LaTeX source and figures.

**Status:** current (v0.1); v0 kept for reference

## Inputs → Outputs
`report_v0.1_en.tex` + `figures/` → `report_v0.1_en.pdf`; `technical_note.tex` + `figures/` → `technical_note.pdf`
(the report PDFs were built before the 11-01 layout refactor and were not rebuilt).

## Entry points
`make_fig_robotize.py` (figure script). LaTeX: `report_v0.1_en.tex`, `report_v0_en.tex`, `technical_note.tex`; figure paths are relative to this folder.

## Layout
- `report_v0.1_en.{tex,pdf}`, `report_v0_en.{tex,pdf}`, `technical_note.{tex,pdf}`, `figures/` (shared; the technical
  report's figures are the `fig_*` files), `make_fig_robotize.py`
- Moved 11-01 (refactor v3) from `docs/technical_report/` to this top-level folder, next to the technical note.

## Provenance — repository paths cited in the report
The reports were written in the project's former working repository, `egocentric-laundry-folding` (merged into this
repository on 11-01), and name it on the title page. They name repository paths as they were before the 11-01 refactor. Current locations:

| path in the report | current path |
|---|---|
| `technical_report/` | `report/` (was `docs/technical_report/` from 11-01 until refactor v3) |
| `scripts/robot_cam_from_anchors.py`, `scripts/robot_cam_solve.py`, `scripts/tag_rig_{solve,pivot,pivot_joint,touch_tcp}.py`, `scripts/umi_tcp_traj.py`, `scripts/jaw_*.py` | `scripts/pose/` (same file names) |
| `scripts/tag_rig_{pose,fuse,corners,imu}.py` | `src/rebot_prep/parity/` |
| `scripts/umi_robotize_bi.py`, `scripts/umi_align.py` | `scripts/retarget/` |
| `scripts/replay_parity.py`, `scripts/replay_side_by_side.py`, `scripts/regress_replay_plan.py` | `scripts/evaluation/` |
| `processing/` | `src/garment_pipeline/` (moved whole to `scripts/processing/` 11-01, then made an importable package in refactor v3) |
| `robot_bridge/` | `scripts/deploy/robot_bridge/` |
| `visual_teleop/` | `src/visual_teleop/` |
| `configs/` | `configs/` (unchanged; `configs/laundry/` → `legacy/configs/laundry/`) |
| `legacy/...` | unchanged |
