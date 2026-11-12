# evaluation

Offline comparison of human demo, robot command and executed robot motion, plus a regression test for the replay planner.

**Status:** current for the replay track; the policy offline eval is `python -m laundry_deploy.tools.offline_eval`

## Inputs → Outputs
A take `datasets/tag_rig/<take>` and a replay run `datasets/replay/<run>` → per-segment RMSE / pass tables
(`replay_parity.py`), a side-by-side video (`replay_side_by_side.py`), and a pass / fail against
`configs/laundry/regression/replay_plan_113243_demo1_right.json` (`regress_replay_plan.py`).

## Entry points
Run from the repository root with `src/` importable (`PYTHONPATH=src`, or after `pip install -e .`); the scripts edit no
`sys.path` themselves.
```bash
python scripts/laundry/eval/replay_parity.py --synthetic                                   # self-test, no data needed
python scripts/laundry/eval/replay_parity.py datasets/tag_rig/<take> datasets/replay/<run>
python scripts/laundry/eval/replay_side_by_side.py datasets/tag_rig/<take> datasets/replay/bi_<ts> datasets/tag_rig/<robot_rec>
python scripts/laundry/eval/regress_replay_plan.py [--update]                              # runs laundry_shared.visual_teleop.tools.umi_replay_bi
```
Characterization tests: `tests/laundry/test_eval_retarget_scripts.py`.

## Layout
- `replay_parity.py`, `replay_side_by_side.py`, `regress_replay_plan.py`
- `epoch_eval_gateway.sh` (11-05): follows a BH-gateway training run (`RUN`, `TAG`, `DSP` = local val/test set prefix,
  `EPOCH_UPD`, optional `PGRASP`), pulls each new epoch checkpoint from `mine/ckpt` (one gateway rsync at a time) and runs
  the offline evaluator on val and test (stride 16) → `reports/<TAG>/epoch_evals.jsonl`; keeps best-val + newest locally.
  Runs inside the production tree `~/laundry_dit_training` (its `evaluation/tagrig_offline_eval.py`, the twin of
  `laundry_deploy.tools.offline_eval`).

## Provenance
Moved from flat `scripts/` in the 11-01 layout refactor, then `scripts/evaluation/` -> `scripts/laundry/eval/`; OA code style 11-04.
