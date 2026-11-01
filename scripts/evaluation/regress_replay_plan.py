"""Planner regression test (2026-10-21): the right-arm demo-1 plan of take 20261021_113243 (robotized, --only right) is the reference
that ran on the real robot (datasets/replay/bi_20261021_143137: TCP median 6.9 mm after lag, gripper 8/8). 10-21 15:40 re-based after the TCP fix
(configs/robot/tcp_override_v001.json) on the GRASP window 19.5-24 s of demo 1, which picked the T-shirt up on the real robot
(bi_20261021_152107; then --tip-clear 0.005 = finger edge just touching, accepted). Re-plans it offline with the
frozen options and fails if IK error / joint speed / link clearance / arm gap got worse than the baseline (+ tolerance).
    ~/xvla-mac/bin/python scripts/evaluation/regress_replay_plan.py              # 비교
    ~/xvla-mac/bin/python scripts/evaluation/regress_replay_plan.py --update     # 지금 결과를 새 기준으로 (의도한 변경일 때만)"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/regression/replay_plan_113243_demo1_right.json"
CMD = ["-m", "visual_teleop.tools.umi_replay_bi", "datasets/tag_rig/20261021_113243", "--rep", "1", "--window", "19.5,24", "--traj", "_robot", "--ori-cost", "0.1", "--speed", "0.5",
       "--q-smooth", "9", "--tcp-margin", "0.010", "--link-margin", "0.025", "--tip-clear", "0.005", "--only", "right", "--roll-flip", "off", "--offline"]
TOL = dict(ik_err_mm=1.0, ik_rot_deg=1.0, max_deg_s=5.0, link_min_mm=-3.0, arm_gap_mm=-10.0)     # allowed worsening


def metrics(dump: Path) -> dict:
    z = np.load(dump); hz = float(z["hz"]); s = "right"; Q = z[f"{s}_Q"]
    return dict(ik_err_mm=float(z[f"{s}_pe"].max() * 1000), ik_rot_deg=float(z[f"{s}_re"].max()), max_deg_s=float(np.degrees(np.abs(np.diff(Q, axis=0))).max() * hz),
                link_min_mm=float(z[f"{s}_linkz"].min() * 1000), arm_gap_mm=float(z["gaps"].min() * 1000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--update", action="store_true"); a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "plan.npz"
        r = subprocess.run([sys.executable, *CMD, "--dump", str(d)], cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}, capture_output=True, text=True)
        if not d.exists(): sys.exit("계획 실패:\n" + (r.stdout + r.stderr)[-1500:])
        m = metrics(d)
    if a.update or not BASE.exists():
        BASE.parent.mkdir(parents=True, exist_ok=True); BASE.write_text(json.dumps(dict(metrics=m, cmd=CMD, tol=TOL, real_replay="datasets/replay/bi_20261021_152107 (+ tip-clear 0.005 rerun)"), indent=1))
        print(f"기준 저장 {BASE}: {m}"); return
    b = json.loads(BASE.read_text())["metrics"]; bad = []
    for k, v in m.items():
        worse = (v - b[k]) if not k.endswith(("min_mm", "gap_mm")) else (b[k] - v)
        lim = abs(TOL[k]); flag = worse > lim; bad += [k] if flag else []
        print(f"  {k:12s} 기준 {b[k]:8.1f}  지금 {v:8.1f}  {'<- 나빠짐' if flag else 'OK'}")
    print("회귀 시험 통과" if not bad else f"회귀 시험 실패: {bad}"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
