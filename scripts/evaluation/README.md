# evaluation

Offline comparison of human demo, robot command and executed robot motion, plus a regression test for the replay planner.

**Status:** current for the replay track; the policy offline eval is `scripts/processing/eval_deploy/tagrig_offline_eval.py`

## Inputs → Outputs
A take `datasets/tag_rig/<take>` and a replay run `datasets/replay/<run>` → per-segment RMSE / pass tables
(`replay_parity.py`), a side-by-side video (`replay_side_by_side.py`), and a pass / fail against
`configs/regression/replay_plan_113243_demo1_right.json` (`regress_replay_plan.py`).

## Entry points
```bash
python scripts/evaluation/replay_parity.py --synthetic                                   # self-test, no data needed
python scripts/evaluation/replay_parity.py datasets/tag_rig/<take> datasets/replay/<run>
python scripts/evaluation/replay_side_by_side.py datasets/tag_rig/<take> datasets/replay/bi_<ts> datasets/tag_rig/<robot_rec>
python scripts/evaluation/regress_replay_plan.py [--update]                              # runs visual_teleop.tools.umi_replay_bi with src/ on PYTHONPATH
```

## Layout
- `replay_parity.py`, `replay_side_by_side.py`, `regress_replay_plan.py`

## Provenance
Moved from flat `scripts/` in the 11-01 layout refactor.
