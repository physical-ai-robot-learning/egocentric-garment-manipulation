#!/usr/bin/env python3
"""[2026-10-28] Joint-space return to deploy/start_pose.json from a nearby pose (after a few policy steps), so the UI's GO check
(READY or start pose within 5 deg) passes again.  Refuses if any joint is more than --max-deg away (not "nearby"), if the TCP dips
below --min-z anywhere on the straight joint path, or if the capsule arm-arm check fails.  /clear_estop first (the bridge then holds
the measured pose), smoothstep ramp at <= --deg-s through HttpBridge.send (the only sanctioned 14-vector path), E-STOP on refusal.
    .venv-mac/bin/python deploy/return_to_start.py [--go]"""
import argparse, json, pathlib, sys, time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: E402
from egogarment import paths as egopaths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--deg-s", type=float, default=10.0)
    ap.add_argument("--max-deg", type=float, default=25.0); ap.add_argument("--min-z", type=float, default=0.04); ap.add_argument("--url", default="http://127.0.0.1:8021")
    a = ap.parse_args()
    from visual_teleop.robot.bridge import HttpBridge
    from visual_teleop.tools.robot_moves import unpark
    sp = json.loads((egopaths.TAGRIG_STATE_DIR / "start_pose.json").read_text()); qs = {s: np.asarray(sp["q"][s], float) for s in core.SIDES}
    arms, ra = core.make_arms(); col = core.Collision(arms)
    br = HttpBridge(a.url, live=a.go, timeout_s=0.5); o = br.observe(); q0 = {s: o.q[s].copy() for s in core.SIDES}
    d = {s: qs[s] - q0[s] for s in core.SIDES}; mx = max(float(np.degrees(np.abs(d[s])).max()) for s in core.SIDES)
    print(f"시작 자세까지 최대 관절 차이 {mx:.1f}도 (한도 {a.max_deg})")
    if mx > a.max_deg: raise SystemExit("시작 자세에서 너무 멀다 -- READY 경유로 이동하세요")
    for u in np.linspace(0, 1, 41):
        q = {s: q0[s] + u * d[s] for s in core.SIDES}
        z = min(core.fk(arms, s, q[s])[2, 3] for s in core.SIDES)
        if z < a.min_z: raise SystemExit(f"경로 중 TCP 높이 {z:.3f} m < {a.min_z} -- 이동 안 함")
        if not col.check(q, table_z=ra.table_z)["ok_arm"]: raise SystemExit("경로 중 두 팔 충돌 위험 -- 이동 안 함")
    n = max(2, int(np.ceil(mx / a.deg_s * 30.0))); print(f"경로 검사 통과: {n / 30:.1f}초, TCP 최저 >= {a.min_z} m")
    if not a.go: print("계획만 출력 (--go 로 이동)"); return
    if br.status().get("estop"): br.s.post(br.base_url + "/clear_estop", timeout=2.0); time.sleep(0.3)
    for s in core.SIDES: unpark(br, s)
    o = br.observe(); q0 = {s: o.q[s].copy() for s in core.SIDES}; d = {s: qs[s] - q0[s] for s in core.SIDES}; g = {s: o.g(s) for s in core.SIDES}
    t0 = time.monotonic()
    for k in range(1, n + 1):
        u = k / n; u = u * u * (3 - 2 * u)
        r = br.send({s: q0[s] + u * d[s] for s in core.SIDES}, g)
        if not r.accepted: br.estop(); raise SystemExit(f"브리지 거부: {r.error} -- E-STOP")
        if k % 15 == 0 and br.observe().estop: br.estop(); raise SystemExit("이동 중 E-STOP/읽기 지연")
        time.sleep(max(0.0, t0 + k / 30.0 - time.monotonic()))
    time.sleep(0.8); o = br.observe()
    print("도착: 시작 자세와 최대 차이 " + ", ".join(f"{s} {np.degrees(np.abs(o.q[s] - qs[s])).max():.1f}도" for s in core.SIDES)); br.close()


if __name__ == "__main__":
    main()
