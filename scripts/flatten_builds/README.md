# flatten_builds

Build tools of the flattening sets under the re-aimed camera (3–5 November): idle-trim audit, tracking-gap audit inside
the trimmed window, GOLD / SILVER / REJECT split, per-take LeRobot builds and QA renders.

**Status:** current (11-04); research scripts as they ran (paths of the recording Mac inside).

## Split rules (11-04)
Trim both ends while both arms are slower than 3 cm/s for 0.5 s (margin 0.3 s). **train** = GOLD whose longest
point-tracking gap inside the trimmed window is < 0.25 s (`GOLD_GAP_MAX`; 0.2 s at first); GOLD with a longer gap
alternates val / test (`GOLDGAP_TO_VALTEST=1`); SILVER without any gap → train (`SILVER_NOGAP_TRAIN=1`), other SILVER →
`silver` (auxiliary robustness evaluation); REJECT → `reject` (analysis only); `FORCE_TRAIN=<keys>` for demos the user
moved to train after review. The clean-GOLD holdout of the first version (`NHOLD`) is 0 in the final builds. Datasets: 0.3× time-stretch, label F, continuous
jaw, gated loss masks for gaps ≥ 0.3 s (`FILL_MASK_S=0.3`); training sets `MASK_CROSS_GAP=1`.

## Layout
- `idle_audit_1103.py`: idle start / end per demo; `ctgap_trimmed.py`: longest tracking gap inside the trimmed window
- `flat_gsr_build.py`: GOLD / SILVER / REJECT split + builds; `flat1103n_build.py`, `flat_build_upload.py`: per-take
  builds and upload to the training nodes
- `render_1103.py`: QA renders

Dataset builder used: `python -m garment_pipeline.labels_dataset.tag_rig_unsplit` (`--mask-mode gated`, `--grip-label cont`).
