# report

The technical report of the project: English `report_en` status of 5 November 2026 (adds Section XII, flattening after
the pinching test: FLAT149, re-aimed camera, gap-based split, FLAT250, grasp labels); Korean `report_ko` status of
2 November 2026 (not updated). LaTeX source and figures. Earlier versions are in [`archive/`](archive/).

**Status:** current (`report_en`, `report_ko`); `archive/` kept for reference

## Inputs → Outputs
`report_en.tex` + `figures/` → `report_en.pdf`; `report_ko.tex` + `figures/` → `report_ko.pdf` (same content in Korean).
Build from this folder with `tectonic -X compile report_en.tex` and `tectonic -X compile report_ko.tex` (the Korean version uses xeCJK with the
macOS AppleMyungjo / Apple SD Gothic Neo fonts).

## Checking the numbers
Every number in `report_en.tex` is mapped to the repository file it was taken from (README, `docs/pipeline.md`,
`docs/history.md`, `docs/results.md`, `docs/details.md`, `docs/status_2026-11-01.md`, `docs/status_2026-11-02.md`,
`docs/status_2026-11-05.md` or `archive/report_v0.1_en.tex`) and checked by
[`../analysis/verify_claims.py`](../analysis/verify_claims.py) (run from the repository root; prints
`N claims checked, 0 failed`; also run in CI). The Korean version may print no number the English one lacks.

## Layout
- `report_en.{tex,pdf}`, `report_ko.{tex,pdf}` — the current report.
- `figures/` — figures of the current report (`fig_scene`, `fig_umi`, `fig_start_markers`, `fig_fusion_qa`,
  `fig_sidebyside`, `fig_close_height`, `fig_deploy_ui`, `fig_pinching_cg`, `fig_fusion_qa_1104` (11-04 QA frame of a GOLD demo under the re-aimed camera), `fig_pinching_robot_1102` — a 2×2 grid of four fixed-camera frames from the 11:37 run of 2 November) and of the archived versions; `make_fig_robotize.py`
  is the script of the archived `fig_robotize.pdf`.
- `archive/` — Technical Report v0.1 (10-27, "Egocentric Laundry Folding"), v0 (10-21) and the short technical note,
  each as LaTeX source + PDF. They were built from this folder (figure paths `figures/...`), so to rebuild one, copy it
  next to `figures/` first.

## Provenance — repository paths cited in the reports
The current reports (`report_en`, `report_ko`) name the paths of the layout of 5 November 2026 (`src/laundry_*`,
`scripts/laundry/`, `configs/laundry/`). The archived reports were written in the project's former working repository,
`egocentric-laundry-folding` (merged into this repository on 11-01), and name it on the title page; they name
repository paths as they were before the 11-01 refactor. Current locations:

| path in the archived reports | current path |
|---|---|
| `technical_report/` | `report/archive/` |
| `scripts/robot_cam_from_anchors.py`, `scripts/robot_cam_solve.py`, `scripts/tag_rig_{solve,pivot,pivot_joint,touch_tcp}.py`, `scripts/umi_tcp_traj.py` | `src/laundry_tracking/calib/` (same module names; run as `python -m laundry_tracking.calib.<name>`) |
| `scripts/jaw_{from_motor,aperture,kpt_label,kpt_train,motorlike}.py` | `src/laundry_jaw/{from_motor,aperture,kpt_label,kpt_train,motorlike}.py` |
| `scripts/tag_rig_{pose,fuse,corners,imu}.py` | `src/laundry_shared/rebot_prep/parity/` |
| `scripts/umi_robotize_bi.py`, `scripts/umi_align.py` | `scripts/laundry/retarget/` |
| `scripts/replay_parity.py`, `scripts/replay_side_by_side.py`, `scripts/regress_replay_plan.py` | `scripts/laundry/eval/` |
| `processing/` | `src/laundry_tracking/`, `src/laundry_dataset/`, `src/laundry_training/`, `src/laundry_deploy/` (module map in [docs/refactor_2026-11.md](../docs/refactor_2026-11.md)) |
| `robot_bridge/` | `src/laundry_deploy/robot_service/` |
| `visual_teleop/` | `src/laundry_shared/visual_teleop/` (the modules the pipeline imports) |
| `configs/` | `configs/laundry/` (the bare-hand collector configs: `archive/configs/laundry/`) |
| `legacy/...` | `archive/...` |

The intermediate layout of 11-01 → 11-05 (`src/garment_pipeline/`, `scripts/pose/`, `scripts/evaluation/`, …) is mapped
in [docs/refactor_2026-11.md](../docs/refactor_2026-11.md).
