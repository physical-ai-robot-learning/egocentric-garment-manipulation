"""Jaw-only test (2026-10-21, left jaw opened too far): hold both arms where they are and step one jaw through --seq g values
(0 = closed, 1 = fully open as the bridge maps it), pausing --hold s at each. HttpBridge.send only (never a hand POST).
    ~/xvla-mac/bin/python -m visual_teleop.tools.jaw_jog --side left --seq 0,0.5,1,0 --go"""
from __future__ import annotations
import argparse, sys, time
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from .robot_moves import unpark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["left", "right"]); ap.add_argument("--seq", default="0,0.5,1,0"); ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--go", action="store_true"); a = ap.parse_args(); seq = [float(v) for v in a.seq.split(",")]
    c = vcfg.load(); br = HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"]); ob = br.observe()
    if ob.estop: sys.exit("E-STOP/미연결/읽기 지연 -- 정지")
    print(f"{a.side} 집게: 지금 g={ob.g(a.side):.2f}, 순서 {seq}, 팔은 지금 자세 고정, 브리지 열림 상한 {br.status().get('grip_open_frac')}")
    if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
    unpark(br, a.side); q = {s: ob.q[s].copy() for s in ("left", "right")}; g = {s: ob.g(s) for s in ("left", "right")}
    for v in seq:
        g[a.side] = v; t0 = time.monotonic()
        while time.monotonic() - t0 < a.hold:
            r = br.send(q, g)
            if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error}")
            time.sleep(1 / 30)
        print(f"  명령 g={v:.2f} -> 측정 g={br.observe().g(a.side):.2f}", flush=True)
    print("끝")


if __name__ == "__main__":
    main()
