# Refactor candidates

Refactor v3 (2026-11-01) was behaviour-preserving: it moved the training-data pipeline into the package
`src/garment_pipeline/`, replaced `sys.path` edits with package imports and home / SSD literals with
[`egogarment.paths`](../src/egogarment/paths.py), and copied the missing helper modules into `src/`
([provenance.md](provenance.md)). Splitting large files or restructuring scripts would change code paths that can only be
checked on the recorded data and the robot, so it was not done. These are the candidates, with a proposed split.

## Done: the deploy UI split (2026-11-01, branch `ui-split`)

The three deploy modules were split into packages next to them; the old files stay as entry points and import names (every
old name is re-exported), so `python -m garment_pipeline.eval_deploy.act_tagrig_{deploy,ui} ...`, `run_n_steps.py`,
`act_tagrig_domain_gap.py` and `import ... act_tagrig_core as core` are unchanged. Working files (`act_tagrig_cams.json`,
`start_pose.json`, `act_tagrig_ui_settings.json`) stay in `eval_deploy/`: the new modules resolve `HERE` to that folder.
Checks: every moved statement is AST-identical to the original (except `HERE`, which now goes one folder up to the same path,
and the UI `main`, whose routes moved verbatim into `create_app`); `--help` of every entry is byte-identical; and
[`tests/test_ui_routes.py`](../tests/test_ui_routes.py) compares the routes, the page, settings persistence, CLI defaults and
helpers with a golden record taken before the split.

| was | lines | now |
|---|---|---|
| `eval_deploy/act_tagrig_deploy.py` | 1209 → 16 | `tagrig_deploy/`: `paths` (working files, `say`), `cameras` (UVC mapping, ring buffer, wrist / Orbbec capture, `cams`, `wave`), `bridge` (grip guard, poller, 30 Hz streamer, READY), `keys`, `preview`, `rig`, `session` (`run_session`, 365 lines, kept whole: one loop whose locals carry the state between the stages), `start_pose`, `report`, `cli` (parser + description = `--help`) |
| `eval_deploy/act_tagrig_ui.py` | 528 → 20 | `tagrig_ui/`: `settings` (persisted settings, checkpoint folders), `control` (`UICtl`, `App`, `move_ready`, `dep_parser`), `server` (`create_app` + `main`), `page.py` + `page.html` (byte-identical) |
| `eval_deploy/act_tagrig_core.py` | 437 → 28 | `tagrig_core/`: `policy` (inputs, decode, RTC weights, ACT / DiT / X-VLA / label runners), `geometry` (arms, FK, capsule collision, camera model, plot); `act_tagrig_core.py` re-exports both |

Not split: `App` (235 lines) stays one class (its methods share the run state); the small tools (`height_calib.py`,
`cam_align.py`, `xvla_server.py`, `return_to_start.py`, `run_n_steps.py`, `cam_viewer.py`) are already one job each.
The flat `import height_calib as HC` inside `run_session` / `start_cmd` was kept as it was (it resolves only when
`eval_deploy/` is on `sys.path`, e.g. `python src/.../act_tagrig_deploy.py run --height-calib`).

## Not done

### Large modules

| file | lines | proposed split |
|---|---|---|
| `src/visual_teleop/vision/stereo_vio_tracker.py` | 542 | belongs to the dropped stereo-VIO experiment; move to `legacy/` together with the `stereo_*_eval` scripts it needs (they are not in this repository) |
| `src/garment_pipeline/labels_dataset/tag_rig_to_lerobot.py` | 526 | take preparation (arm assignment, robotize / IK sub-process) vs. LeRobot writer (frames, features, metadata); `tag_rig_unsplit.py` already reuses both halves |
| `src/garment_pipeline/training/tagrig_common.py` | 455 | datasets (`ActRelDataset`, `DitRelDataset`) vs. the trainer patches (`install_masks`, `install_weights`, `install_grip_bce`, ...) |
| `src/visual_teleop/tools/tag_rig_record.py` | 449 | device setup (Orbbec, wrist cameras, IMUs, jaw servos) vs. recording session / protocol prompts |

### Scripts that run at import

These modules do their work at module level (no `if __name__ == "__main__":` guard), so importing them runs them;
`tests/test_imports.py` skips them. Wrapping the body in `main()` is a mechanical change but touches global-name
lookups inside nested functions, so it should be checked on a recorded take first.

`garment_pipeline.fgo_tools.tag_rot_extract_v002`, `garment_pipeline.fusion_stack.orbbec.v2b.cotrack.make_pack2`,
`...cotrack.run_cotracker`, `...cotrack.solve2` (functions first, then the solve loop), `...orbbec.v2b.tighten_trim`,
`garment_pipeline.labels_dataset.idle_trim`; `...orbbec.v2b.track_v31` has a guard but reads `start_seed_v1.json` and
creates `out/` at import.

### Working files next to the code

The fusion stage reads and writes `fusion_stack/orbbec/v2b/out/`, `fusion_stack/stitch/qa/` and `start_seed_*.json`
next to its modules, and the deploy tools keep `act_tagrig_cams.json`, `start_pose.json`, `height_calib*.json` and
`act_tagrig_ui_settings.json` next to theirs (on the original machine these folders were data folders). A
`FUSION_WORK` / `DEPLOY_STATE` root in `egogarment.paths` would move them out of the package; that changes where files
land relative to the repository, so it was left for a separate change.

`labels_dataset/tag_rig_to_lerobot.py` and `tag_rig_unsplit.py` default `--derived`, `--old-split` and `--render-dir`
to `<package>/data/...` and `<package>/reports/...` (`HERE.parent`), which meant the training workspace on the original
machine; pass them explicitly or move the defaults to `egogarment.paths.TRAIN_ROOT`.
