"""UMI <-> robot gripper ORIENTATION alignment (2026-10-20, absolute replay). The robot holds READY (fixed R_B_E from the FK of
READY_Q_DEG); the user holds the UMI parallel to the gripper (same pointing direction, same jaw-opening direction) in each
'멈추세요' of a tag_rig_record --proto align take. Per hold (last 3 s, TAG frames only): R_E_U = R_B_E^T R_B_U. Mean over holds,
spread = how well the hand matched (expect a few deg). Writes configs/calibration/umi_gripper_align_v001.json: R_U_E (UMI TCP frame ->
robot TCP frame) so that the robot target orientation for a UMI pose is R_B_E* = R_B_U R_U_E.
    ~/xvla-mac/bin/python scripts/retarget/umi_align.py datasets/tag_rig/<align take> [--write]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--hand", default="right"); ap.add_argument("--write", action="store_true"); ap.add_argument("--holds", type=int, nargs="*", help="쓸 회차 (1부터), 기본 전부")
    ap.add_argument("--out"); a = ap.parse_args()
    a.out = a.out or str(ROOT / ("configs/calibration/umi_gripper_align_v001.json" if a.hand == "right" else "configs/calibration/umi_gripper_align_left_v001.json"))
    from visual_teleop import config as vcfg
    from visual_teleop.robot.ik_pink import PinkIkSolver, PinkParams
    from visual_teleop.tools.robot_moves import READY_Q_DEG
    from ego_teleop.robot.rebot_client import SIDE_SLICE
    c = vcfg.load(); j = np.zeros(14); j[SIDE_SLICE[a.hand]] = READY_Q_DEG[a.hand]; pk = PinkIkSolver(a.hand, PinkParams(**c.ik["pink"])); pk.sync(j)
    R_BE = R.from_matrix(pk.fk(np.radians(READY_Q_DEG[a.hand]))[:3, :3])
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); z = np.load(E / f"umi_traj_{a.hand}.npz"); t = z["t_ns"]; tag = z["ntags"] > 0
    t0 = int(meta["t_start_monotonic_ns"]); out = []
    hn = 0
    for e in meta["protocol"]:
        if "맞춰 대고" not in e["text"]: continue
        hn += 1
        if a.holds and hn not in a.holds: continue
        m = tag & (t >= t0 + (e["t0"] + e["dur"] - 3.0) * 1e9) & (t <= t0 + (e["t0"] + e["dur"]) * 1e9)
        if m.sum() < 5: print(f"  {e['text'][-3:]}: 태그 프레임 부족 ({int(m.sum())})"); continue
        RBU = R.from_matrix(z["R_B_tcp"][m]).mean(); out.append((R_BE.inv() * RBU)); print(f"  {e['text'][-3:]}: 프레임 {int(m.sum())}, R_E_U rotvec {np.round(np.degrees(out[-1].as_rotvec()), 1)}°")
    if len(out) < 2: sys.exit("정지 구간이 2개 이상 필요")
    M = R.concatenate(out).mean(); spread = [float(np.degrees((M.inv() * x).magnitude())) for x in out]
    print(f"평균 R_E_U rotvec {np.round(np.degrees(M.as_rotvec()), 1)}°, 회차별 차이 {np.round(spread, 1)}° (중앙 {np.median(spread):.1f}°)")
    if a.write:
        Path(a.out).write_text(json.dumps(dict(R_U_E=M.inv().as_matrix().tolist(), R_E_U=M.as_matrix().tolist(), spread_deg=spread, take=str(E), hand=a.hand,
                                               note="robot target orientation = R_B_U * R_U_E (R_B_U = umi_traj R_B_tcp)"), indent=1))
        print("기록", a.out)


if __name__ == "__main__":
    main()
