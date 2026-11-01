# technical_report

Technical Report v0.1 (10-27, current) and v0 (10-21): PDF, LaTeX source and figures.

**Status:** current (v0.1); v0 kept for reference

## Inputs → Outputs
`report_v0.1_en.tex` + `figures/` → `report_v0.1_en.pdf` (the PDFs in this folder were built before the 11-01 layout
refactor and were not rebuilt).

## Entry points
`make_fig_robotize.py` (figure script). LaTeX: `report_v0.1_en.tex`, `report_v0_en.tex`; figure paths are relative to this folder.

## Layout
- `report_v0.1_en.{tex,pdf}`, `report_v0_en.{tex,pdf}`, `figures/`, `make_fig_robotize.py`

## Provenance — repository paths cited in the report
The reports were written in the project's former working repository, `egocentric-laundry-folding` (merged into this
repository on 11-01), and name it on the title page. They name repository paths as they were before the 11-01 refactor. Current locations:

| path in the report | current path |
|---|---|
| `technical_report/` | `docs/technical_report/` |
| `scripts/robot_cam_from_anchors.py`, `scripts/robot_cam_solve.py`, `scripts/tag_rig_{solve,pivot,pivot_joint,touch_tcp}.py`, `scripts/umi_tcp_traj.py`, `scripts/jaw_*.py` | `scripts/pose/` (same file names) |
| `scripts/tag_rig_{pose,fuse,corners,imu}.py` | `src/rebot_prep/parity/` |
| `scripts/umi_robotize_bi.py`, `scripts/umi_align.py` | `scripts/retarget/` |
| `scripts/replay_parity.py`, `scripts/replay_side_by_side.py`, `scripts/regress_replay_plan.py` | `scripts/evaluation/` |
| `processing/` | `scripts/processing/` (moved whole) |
| `robot_bridge/` | `scripts/deploy/robot_bridge/` |
| `visual_teleop/` | `src/visual_teleop/` |
| `configs/` | `configs/` (unchanged; `configs/laundry/` → `legacy/configs/laundry/`) |
| `legacy/...` | unchanged |
