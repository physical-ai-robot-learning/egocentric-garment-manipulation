"""Planner regression test (2026-10-21; was scripts/evaluation/regress_replay_plan.py).

Pipeline stage: replay-track planning (umi_replay_bi), regression guard. The right-arm demo-1 plan of take 20261021_113243
(robotized, --only right) is the reference that ran on the real robot (datasets/replay/bi_20261021_143137: TCP median 6.9 mm after
lag, gripper 8/8). 10-21 15:40 re-based after the TCP fix (configs/laundry/robot/tcp_override_v001.json) on the GRASP window
19.5-24 s of demo 1, which picked the T-shirt up on the real robot (bi_20261021_152107; then --tip-clear 0.005 = finger edge just
touching, accepted). Re-plans it offline with the frozen options (run from the repository root, take path relative to it) and fails
if IK error / joint speed / link clearance / arm gap got worse than the baseline (+ tolerance).
Input: the planner dump; output: printed comparison, exit status 1 on a regression; baseline
configs/laundry/regression/replay_plan_113243_demo1_right.json.
    python scripts/laundry/eval/regress_replay_plan.py              # 비교
    python scripts/laundry/eval/regress_replay_plan.py --update     # 지금 결과를 새 기준으로 (의도한 변경일 때만)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from laundry_common.paths import CONFIGS, REPO_ROOT, SRC_ROOT

BASE = CONFIGS / "regression/replay_plan_113243_demo1_right.json"
# Frozen planner command (also stored in the baseline json as `cmd`).
CMD = ["-m", "laundry_shared.visual_teleop.tools.umi_replay_bi", "datasets/tag_rig/20261021_113243", "--rep", "1", "--window", "19.5,24",
       "--traj", "_robot", "--ori-cost", "0.1", "--speed", "0.5", "--q-smooth", "9", "--tcp-margin", "0.010", "--link-margin", "0.025",
       "--tip-clear", "0.005", "--only", "right", "--roll-flip", "off", "--offline"]
# allowed worsening per metric (negative: a clearance may shrink by that much)
TOL = dict(ik_err_mm=1.0, ik_rot_deg=1.0, max_deg_s=5.0, link_min_mm=-3.0, arm_gap_mm=-10.0)
# metrics where smaller is worse (clearances)
CLEARANCE_SUFFIXES = ("min_mm", "gap_mm")
REAL_REPLAY = "datasets/replay/bi_20261021_152107 (+ tip-clear 0.005 rerun)"
SIDE = "right"
MM_PER_M = 1000
FAILURE_TAIL_CHARS = 1500
JSON_INDENT = 1


class PlanError(RuntimeError):
    """The offline planner wrote no dump."""


def metrics(dump: Path) -> dict:
    """Worst IK position / rotation error, joint speed, link height and arm gap of the dumped right-arm plan."""
    tPlan = np.load(dump)
    tHz = float(tPlan["hz"])
    tQ = tPlan[f"{SIDE}_Q"]
    return dict(ik_err_mm=float(tPlan[f"{SIDE}_pe"].max() * MM_PER_M), ik_rot_deg=float(tPlan[f"{SIDE}_re"].max()),
                max_deg_s=float(np.degrees(np.abs(np.diff(tQ, axis=0))).max() * tHz),
                link_min_mm=float(tPlan[f"{SIDE}_linkz"].min() * MM_PER_M), arm_gap_mm=float(tPlan["gaps"].min() * MM_PER_M))


def planMetrics() -> dict:
    """Run the frozen planner command offline and return its metrics."""
    with tempfile.TemporaryDirectory() as tDir:
        tDump = Path(tDir) / "plan.npz"
        tEnv = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_ROOT), os.environ.get("PYTHONPATH")]))}
        tRun = subprocess.run([sys.executable, *CMD, "--dump", str(tDump)], cwd=REPO_ROOT, env=tEnv, capture_output=True, text=True)
        if not tDump.exists():
            raise PlanError("계획 실패:\n" + (tRun.stdout + tRun.stderr)[-FAILURE_TAIL_CHARS:])
        return metrics(tDump)


def compare(current: dict, baseline: dict) -> list[str]:
    """Print each metric against the baseline; return the names that got worse than TOL allows."""
    tBad = []
    for tKey, tValue in current.items():
        tWorse = (tValue - baseline[tKey]) if not tKey.endswith(CLEARANCE_SUFFIXES) else (baseline[tKey] - tValue)
        tLimit = abs(TOL[tKey])
        tFlag = tWorse > tLimit
        tBad += [tKey] if tFlag else []
        print(f"  {tKey:12s} 기준 {baseline[tKey]:8.1f}  지금 {tValue:8.1f}  {'<- 나빠짐' if tFlag else 'OK'}")
    return tBad


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("--update", action="store_true")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        tMetrics = planMetrics()
    except PlanError as tError:
        print(tError, file=sys.stderr)
        return 1
    if tArgs.update or not BASE.exists():
        BASE.parent.mkdir(parents=True, exist_ok=True)
        BASE.write_text(json.dumps(dict(metrics=tMetrics, cmd=CMD, tol=TOL, real_replay=REAL_REPLAY), indent=JSON_INDENT))
        print(f"기준 저장 {BASE}: {tMetrics}")
        return 0
    tBad = compare(tMetrics, json.loads(BASE.read_text())["metrics"])
    print("회귀 시험 통과" if not tBad else f"회귀 시험 실패: {tBad}")
    return 1 if tBad else 0


if __name__ == "__main__":
    sys.exit(main())
