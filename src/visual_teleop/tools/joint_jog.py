"""Small single-joint jog to check a joint's direction and magnitude after a re-zero (2026-10-21, left j6 re-zeroed 178.5 -> 0).
Moves ONE joint of ONE arm through --seq offsets (deg, relative to where it is now) at --deg-s, holding every other joint and both jaws at
their observed values; after each step it waits --settle s and prints commanded vs measured. Sent through HttpBridge.send (ramp_to) --
never a hand POST. Without --go it only prints the plan.
    ~/xvla-mac/bin/python -m visual_teleop.tools.joint_jog --side left --joint 6 --seq 5,0,-5,0            # 계획만
    ~/xvla-mac/bin/python -m visual_teleop.tools.joint_jog --side left --joint 6 --seq 5,0,-5,0 --go       # 실제 이동"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from .robot_moves import ramp_to, unpark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["left", "right"]); ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--seq", default="5,0,-5,0", help="지금 각도 기준 상대 목표들 (도)"); ap.add_argument("--deg-s", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=1.5); ap.add_argument("--max-abs", type=float, default=30.0); ap.add_argument("--go", action="store_true"); a = ap.parse_args()
    seq = [float(v) for v in a.seq.split(",")]
    if max(abs(v) for v in seq) > a.max_abs: sys.exit(f"--seq 가 ±{a.max_abs}° 를 넘음 (--max-abs 로 명시적으로 올리세요)")
    c = vcfg.load(); br = HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"]); j = a.joint - 1
    ob = br.observe()
    if ob.estop: sys.exit("E-STOP/미연결/읽기 지연 -- 정지")
    q0 = ob.q[a.side].copy(); print(f"{a.side} 지금 관절(도): " + " ".join(f"j{k + 1}={v:6.1f}" for k, v in enumerate(np.degrees(q0))))
    print(f"계획: j{a.joint} 를 {np.degrees(q0[j]):.1f}° 기준으로 " + " -> ".join(f"{v:+.0f}" for v in seq) + f" (°), {a.deg_s:.0f}°/s, 나머지 관절·집게 고정")
    if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
    unpark(br, a.side)
    for v in seq:
        qt = q0.copy(); qt[j] += np.radians(v); ramp_to(br, a.side, qt, a.deg_s); time.sleep(a.settle)
        qm = br.observe().q[a.side]; others = np.degrees(np.abs(np.delete(qm - q0, j))).max()
        print(f"  목표 {v:+5.1f}° -> 측정 {np.degrees(qm[j] - q0[j]):+6.2f}° (j{a.joint} 절대 {np.degrees(qm[j]):6.1f}°), 다른 관절 최대 변화 {others:.2f}°", flush=True)
    print("끝")


if __name__ == "__main__":
    main()
