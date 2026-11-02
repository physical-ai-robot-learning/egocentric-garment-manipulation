# report

The technical report of the project, status of 1 November 2026: one document in English and Korean, with LaTeX source
and figures. Earlier versions are in [`archive/`](archive/).

**Status:** current (`report_en`, `report_ko`); `archive/` kept for reference

## Inputs → Outputs
`report_en.tex` + `figures/` → `report_en.pdf`; `report_ko.tex` + `figures/` → `report_ko.pdf` (same content in Korean).
Build from this folder with `tectonic report_en.tex` and `tectonic report_ko.tex` (the Korean version uses xeCJK with the
macOS AppleMyungjo / Apple SD Gothic Neo fonts).

## Checking the numbers
Every number in `report_en.tex` is mapped to the repository file it was taken from (README, `docs/results.md`,
`docs/details.md`, `docs/status_2026-11-01.md`, `docs/status_2026-11-02.md` or `archive/report_v0.1_en.tex`) and checked by
[`../analysis/verify_claims.py`](../analysis/verify_claims.py) (run from the repository root; prints
`N claims checked, 0 failed`; also run in CI). The Korean version may print no number the English one lacks.

## Layout
- `report_en.{tex,pdf}`, `report_ko.{tex,pdf}` — the current report.
- `figures/` — figures of the current report (`fig_scene`, `fig_umi`, `fig_start_markers`, `fig_fusion_qa`,
  `fig_sidebyside`, `fig_close_height`, `fig_deploy_ui`, `fig_pinching_cg`, `fig_pinching_robot_1102` — a 2×2 grid of four fixed-camera frames from the 11:37 run of 2 November) and of the archived versions; `make_fig_robotize.py`
  is the script of the archived `fig_robotize.pdf`.
- `archive/` — Technical Report v0.1 (10-27, "Egocentric Laundry Folding"), v0 (10-21) and the short technical note,
  each as LaTeX source + PDF. They were built from this folder (figure paths `figures/...`), so to rebuild one, copy it
  next to `figures/` first.

## Provenance — repository paths cited in the archived reports
The archived reports were written in the project's former working repository, `egocentric-laundry-folding` (merged into
this repository on 11-01), and name it on the title page. They name repository paths as they were before the 11-01
refactor. Current locations:

| path in the archived reports | current path |
|---|---|
| `technical_report/` | `report/archive/` |
| `scripts/robot_cam_from_anchors.py`, `scripts/robot_cam_solve.py`, `scripts/tag_rig_{solve,pivot,pivot_joint,touch_tcp}.py`, `scripts/umi_tcp_traj.py`, `scripts/jaw_*.py` | `scripts/pose/` (same file names) |
| `scripts/tag_rig_{pose,fuse,corners,imu}.py` | `src/rebot_prep/parity/` |
| `scripts/umi_robotize_bi.py`, `scripts/umi_align.py` | `scripts/retarget/` |
| `scripts/replay_parity.py`, `scripts/replay_side_by_side.py`, `scripts/regress_replay_plan.py` | `scripts/evaluation/` |
| `processing/` | `src/garment_pipeline/` |
| `robot_bridge/` | `scripts/deploy/robot_bridge/` |
| `visual_teleop/` | `src/visual_teleop/` |
| `configs/` | `configs/` (unchanged; `configs/laundry/` → `legacy/configs/laundry/`) |
| `legacy/...` | unchanged |
