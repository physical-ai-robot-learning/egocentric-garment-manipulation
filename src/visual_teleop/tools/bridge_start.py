"""Start / restart the MIT bridge with the frozen deployment settings (2026-10-20, configs/robot/rebot_mit_deploy_v001.json).
The arm USB location is taken from usb_arm_ports (exactly one arm port -> that one; else pass --loc). A running bridge is only
replaced when its arms are PARKED (rest reached, torque off) -- otherwise this sends /rest first and waits, because killing a bridge
with torque on drops the arm. Kills by PID only (never a pattern; 10-01 incident).
    python3 -m visual_teleop.tools.bridge_start            # 상태와 할 일만 출력
    python3 -m visual_teleop.tools.bridge_start --go       # (필요하면 REST ->) 재시작 -> /connect -> 설정 확인"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def http(url, data=None, timeout=5):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/robot/rebot_mit_deploy_v001.json")); ap.add_argument("--loc"); ap.add_argument("--go", action="store_true")
    ap.add_argument("--sides", help="right | left | left,right (기본: 설정의 MIT_SIDES). 10-21: 양손"); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); b = cfg["bridge"]; env = dict(b["env"]); url = f"http://localhost:{b['port']}"
    if a.sides: env["MIT_SIDES"] = a.sides
    sides = [x for x in ("left", "right") if x in env["MIT_SIDES"].split(",")]
    out = subprocess.run([os.path.expanduser(b["python"]), "usb_arm_ports.py"], cwd=os.path.expanduser("~/robot-cockpit"), capture_output=True, text=True).stdout
    present = [l.split()[0].split("=")[1] for l in out.splitlines() if l.startswith("loc=")]
    for side in sides:
        loc = (a.loc if len(sides) == 1 and a.loc else None) or b.get("locs", {}).get(side)
        if not loc and len(sides) == 1 and len(present) == 1: loc = present[0]
        if not loc: sys.exit(f"{side} 팔 USB 위치가 설정에 없음 -- --loc 또는 설정 locs ({out.strip()})")
        if loc not in present: sys.exit(f"{side} 팔 위치 {loc} 가 지금 연결된 포트에 없음: {present} -- USB 를 확인하거나 설정 locs 를 고치세요")
        env[b["loc_env"][side]] = loc
    side = ",".join(sides); loc = ", ".join(f"{x} {env[b['loc_env'][x]]}" for x in sides)
    pid = subprocess.run(["pgrep", "-f", "robot_service_mit.py"], capture_output=True, text=True).stdout.split()
    st = None
    try: st = http(url + "/status")
    except Exception: pass
    print(f"설정 {Path(a.config).name}: {side} 팔 위치 {loc}, " + ", ".join(f"{k}={v}" for k, v in env.items() if k.startswith("MIT_K") or k == "MIT_GRIP_DEG_S"))
    if st: print(f"지금 브리지: PID {pid}, kp {st.get('kp')}, ki {st.get('ki')}, ki_joints {st.get('ki_joints')}, 집게 {st.get('grip_deg_s')}°/s, parked {st.get('parked')}")
    else: print("지금 브리지: 응답 없음 (꺼져 있음)")
    if not a.go: print("--go 로 실행 (켜져 있으면 REST 후 재시작)"); return
    if st and pid:
        if any(x not in (st.get("parked") or []) for x in (st.get("kp") or {})):
            print("REST 로 보냄 (토크가 켜진 채로 끄면 팔이 떨어짐)"); http(url + "/rest", {})
            for _ in range(60):
                time.sleep(1)
                st2 = http(url + "/status")
                if all(x in (st2.get("parked") or []) for x in (st2.get("kp") or {})): break
            else: sys.exit("REST 도착(토크 꺼짐)을 확인 못함 -- 재시작 안 함")
        for p in pid: subprocess.run(["kill", p])
        time.sleep(4)
        if subprocess.run(["pgrep", "-f", "robot_service_mit.py"], capture_output=True, text=True).stdout.strip(): sys.exit("이전 브리지가 아직 살아 있음")
    # 10-21: jaw zero check (read-only, ports are free now). The left jaw zero jumped -268 / -100 deg across power cycles -> "closed"
    # drove it open. Closed jaws should read ~0 deg; warn otherwise (fix: robot-cockpit/zero_left_gripper_only.py with the jaws closed).
    ports = {l.split("=")[1].split()[0]: l.split()[-1] for l in out.splitlines() if l.startswith("loc=")}
    try:
        chk = subprocess.run([os.path.expanduser(b["python"]), "check_grippers.py", *[ports[env[b["loc_env"][x]]] for x in sides if env[b["loc_env"][x]] in ports]],
                             cwd=os.path.expanduser("~/robot-cockpit"), capture_output=True, text=True, timeout=60).stdout
        for l in chk.splitlines():
            if l.startswith("GRIP"):
                _, port_, val = l.split(maxsplit=2); side_ = [x for x in sides if ports.get(env[b["loc_env"][x]]) == port_]
                bad = val.startswith("ERR") or abs(float(val)) > 5
                print(f"집게 {side_[0] if side_ else port_}: {val}{'°' if not val.startswith('ERR') else ''}" + ("  <- 닫혀 있는데 이 값이면 영점이 틀어짐! (zero_left_gripper_only.py)" if bad else "  OK"))
    except Exception as e:
        print("집게 영점 확인 실패:", str(e)[:120])
    full = {**os.environ, **env}; log = open(os.path.expanduser(b["log"]), "a")
    subprocess.Popen([os.path.expanduser(b["python"]), os.path.expanduser(b["script"])], cwd=os.path.expanduser("~/robot-cockpit"), env=full, stdout=log, stderr=log, start_new_session=True)
    for _ in range(40):
        time.sleep(1)
        try: http(url + "/status"); break
        except Exception: pass
    print("connect:", http(url + "/connect", {}, timeout=60).get("ok")); time.sleep(2); st = http(url + "/status")
    want_kp = [float(x) for x in env["MIT_KP"].split(",")]; ok = all(st.get("kp", {}).get(x) == want_kp for x in sides) and abs(st.get("grip_deg_s", 0) - float(env["MIT_GRIP_DEG_S"])) < 1e-6
    print(f"확인: connected {st.get('connected')}, estop {st.get('estop')}, kp {st.get('kp')}, ki {st.get('ki')}, ki_joints {st.get('ki_joints')}, 집게 {st.get('grip_deg_s')}°/s -> {'설정 일치' if ok else '설정 불일치!'}")


if __name__ == "__main__":
    main()
