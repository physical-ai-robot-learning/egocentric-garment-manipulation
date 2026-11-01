# Refactor candidates (not done)

Refactor v3 (2026-11-01) was behaviour-preserving: it moved the training-data pipeline into the package
`src/garment_pipeline/`, replaced `sys.path` edits with package imports and home / SSD literals with
[`egogarment.paths`](../src/egogarment/paths.py), and copied the missing helper modules into `src/`
([provenance.md](provenance.md)). Splitting large files or restructuring scripts would change code paths that can only be
checked on the recorded data and the robot, so it was not done. These are the candidates, with a proposed split.

## Large modules

| file | lines | proposed split |
|---|---|---|
| `src/garment_pipeline/eval_deploy/act_tagrig_deploy.py` | 1209 | `cameras` (wrist / global capture, UVC exposure), `bridge` (robot I/O, heartbeat, stop), `loop` (receding-horizon executor, RTC blend), `cli` (sub-commands `run`, `preview`, ...); the policy runners already live in `act_tagrig_core.py` |
| `src/visual_teleop/vision/stereo_vio_tracker.py` | 542 | belongs to the dropped stereo-VIO experiment; move to `legacy/` together with the `stereo_*_eval` scripts it needs (they are not in this repository) |
| `src/garment_pipeline/eval_deploy/act_tagrig_ui.py` | 528 | web app (Flask routes + page) vs. run control (settings, checkpoint discovery, deploy threads) |
| `src/garment_pipeline/labels_dataset/tag_rig_to_lerobot.py` | 526 | take preparation (arm assignment, robotize / IK sub-process) vs. LeRobot writer (frames, features, metadata); `tag_rig_unsplit.py` already reuses both halves |
| `src/garment_pipeline/training/tagrig_common.py` | 455 | datasets (`ActRelDataset`, `DitRelDataset`) vs. the trainer patches (`install_masks`, `install_weights`, `install_grip_bce`, ...) |
| `src/visual_teleop/tools/tag_rig_record.py` | 449 | device setup (Orbbec, wrist cameras, IMUs, jaw servos) vs. recording session / protocol prompts |
| `src/garment_pipeline/eval_deploy/act_tagrig_core.py` | 437 | policy runners (ACT / DiT / X-VLA client) vs. geometry helpers (frames, RTC weights, workspace clamp) |

## Scripts that run at import

These modules do their work at module level (no `if __name__ == "__main__":` guard), so importing them runs them;
`tests/test_imports.py` skips them. Wrapping the body in `main()` is a mechanical change but touches global-name
lookups inside nested functions, so it should be checked on a recorded take first.

`garment_pipeline.fgo_tools.tag_rot_extract_v002`, `garment_pipeline.fusion_stack.orbbec.v2b.cotrack.make_pack2`,
`...cotrack.run_cotracker`, `...cotrack.solve2` (functions first, then the solve loop), `...orbbec.v2b.tighten_trim`,
`garment_pipeline.labels_dataset.idle_trim`; `...orbbec.v2b.track_v31` has a guard but reads `start_seed_v1.json` and
creates `out/` at import.

## Working files next to the code

The fusion stage reads and writes `fusion_stack/orbbec/v2b/out/`, `fusion_stack/stitch/qa/` and `start_seed_*.json`
next to its modules, and the deploy tools keep `act_tagrig_cams.json`, `start_pose.json`, `height_calib*.json` and
`act_tagrig_ui_settings.json` next to theirs (on the original machine these folders were data folders). A
`FUSION_WORK` / `DEPLOY_STATE` root in `egogarment.paths` would move them out of the package; that changes where files
land relative to the repository, so it was left for a separate change.

`labels_dataset/tag_rig_to_lerobot.py` and `tag_rig_unsplit.py` default `--derived`, `--old-split` and `--render-dir`
to `<package>/data/...` and `<package>/reports/...` (`HERE.parent`), which meant the training workspace on the original
machine; pass them explicitly or move the defaults to `egogarment.paths.TRAIN_ROOT`.
