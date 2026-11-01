#!/usr/bin/env python3
"""[2026-11-01] user "off를 임의로 주기보다 로봇에 맞추어서" + "왼쪽 에이프릴 태그로 면을 맞추자": robot <-> training-frame height calibration.
The training labels live in the B frame defined by the table tags 400/401 (configs/calibration/start_anchors_v003_70mm.json: centres at
z = 0.001 = table plane, measured with a PROVISIONAL camera "re-measure with the robot when it is installed").  Here the robot measures
them: an arm is parked (torque off), the operator puts the closed jaw tip on a tag centre (or anywhere on the table), presses Enter, and
the TCP = FK(measured q) with the same arms / FK as the deploy is recorded.  Per arm: dz = TCP z - anchor z (tags: also dx, dy), and a
table plane z = a x + b y + c from all touches of that arm.  The deploy's "calib" height mode adds -dz(x, y) of the arm, evaluated at the
arm's TCP, instead of a hand-typed z offset.  Not a limit -- a frame correction.
  measure:  python deploy/height_calib.py measure            (Korean prompts; result -> deploy/height_calib.json, older kept as .prev)
  show:     python deploy/height_calib.py show"""
from __future__ import annotations
import argparse, json, pathlib, shutil, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import act_tagrig_core as core  # noqa: E402

OUT = HERE / "height_calib.json"
ANCHORS = pathlib.Path.home() / "ego_collector/configs/calibration/start_anchors_v003_70mm.json"
TABLE_Z = 0.001        # anchors file: tags lie on the table -> z = 0.001 m in B


def fit(points):
    """per arm: tag offsets (median of touches on that tag) + table plane (least squares; < 3 points or collinear -> flat at the mean)"""
    out = {}
    for s in core.SIDES:
        P = [p for p in points if p["arm"] == s]
        if not P: continue
        tags = {}
        for tid in sorted({p["target"] for p in P if p["target"] != "table"}):
            d = np.array([np.subtract(p["p_robot"], p["p_expected"]) for p in P if p["target"] == tid])
            tags[tid] = dict(n=len(d), d_mm=np.round(np.median(d, 0) * 1e3, 1).tolist())
        X = np.array([p["p_robot"] for p in P]); dz = X[:, 2] - TABLE_Z
        if len(X) >= 3 and np.linalg.matrix_rank(np.c_[X[:, :2] - X[:, :2].mean(0)]) == 2:
            A = np.c_[X[:, :2], np.ones(len(X))]; coef, *_ = np.linalg.lstsq(A, dz, rcond=None)
            res = dz - A @ coef; tilt = float(np.degrees(np.arctan(np.hypot(*coef[:2]))))
        else:
            coef = np.array([0.0, 0.0, float(np.mean(dz))]); res = dz - coef[2]; tilt = None
        out[s] = dict(n=len(P), tags=tags, plane_dz=dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2])),
                      dz_mean_mm=round(float(np.mean(dz)) * 1e3, 1), resid_mm=round(float(np.abs(res).max()) * 1e3, 1),
                      tilt_deg=None if tilt is None else round(tilt, 2))
    return out


def dz_at(cal, side, xy):
    """height error of the training frame at TCP xy (m): robot table = data table + dz -> the deploy adds +dz to the commanded z"""
    f = (cal or {}).get("fit", {}).get(side)
    if not f: return 0.0
    p = f["plane_dz"]; return p["a"] * float(xy[0]) + p["b"] * float(xy[1]) + p["c"]


def to_B(cal, side, P):
    """[11-01] robot-FK points (N, 3) -> training B frame: + per-arm xy shift (camxy) and - table-plane dz(x, y) (UI overlay)"""
    P = np.array(P, np.float64, ndmin=2); f = (cal or {}).get("fit", {}).get(side)
    if not f: return P
    out = P.copy(); out[:, :2] += np.asarray(f.get("xy_cam_minus_robot", [0.0, 0.0]))   # overlay: the CURRENT camera (it moved since 10-26)
    out[:, 2] -= np.array([dz_at(cal, side, x[:2]) for x in P]); return out


def camxy():
    """[11-01] user "UI 에서 TCP 가 벗어나": the Orbbec (robot_cam_v005 = the B frame of the labels) sees the tag centres; their pixel ray
    on the table (z = 0.001) is the true B position, the pivot touches (pending p400 / p401) give the robot-FK position of the same
    point -> per-arm xy shift B - robot, stored into height_calib.json (plane untouched)"""
    import cv2, urllib.request
    C = pathlib.Path.home() / "ego_collector/configs/calibration"
    K = json.loads((C / "robot_cam_v001.json").read_text())["intrinsics"]; Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    T_B_C = np.array(json.loads((C / "robot_cam_v005.json").read_text())["T_B_C"])
    b = urllib.request.urlopen("http://127.0.0.1:8770/stream/global", timeout=3).read(400000)
    i = b.find(b"\xff\xd8", len(b) // 2); j = b.find(b"\xff\xd9", i); im = cv2.imdecode(np.frombuffer(b[i:j + 2], np.uint8), 0)
    c, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)).detectMarkers(im)
    pts = json.loads(PENDING.read_text()) if PENDING.exists() else []; cal = load()
    for cc, tid in zip(c if ids is not None else [], ids.ravel() if ids is not None else []):
        P = [p for p in pts if p["target"] == f"p{tid}"]
        if not P: continue
        d = np.linalg.inv(Km) @ np.r_[cc[0].mean(0), 1]; o = T_B_C[:3, 3]; r = T_B_C[:3, :3] @ d; pB = o + (TABLE_Z - o[2]) / r[2] * r
        pr = np.mean([p["p_robot"] for p in P], 0); s = P[0]["arm"]; sh = (pB - pr)[:2]
        cal["fit"][s]["xy_cam_minus_robot"] = sh.tolist(); cal["fit"][s]["xy_cam_from"] = f"tag {tid} camera ray vs {len(P)} pivots"
        print(f"{s}: tag {tid} B {np.round(pB * 1e3, 1)} robot {np.round(pr * 1e3, 1)} -> xy shift {np.round(sh * 1e3, 1)} mm, z check {(pB - pr)[2] * 1e3:+.1f} vs plane {-dz_at(cal, s, pr[:2]) * 1e3:+.1f} mm")
    OUT.write_text(json.dumps(cal, indent=1))


def anchorxy():
    """[11-01] user "xy 도 맞추어줘": camera-free xy shift for the robot COMMANDS = anchors-file tag centre (B, 10-26; the re-printed tags
    were stuck over the old ones) - robot FK of the touches on that tag (pivots p400/p401 pending + tag touches 400/401) -> xy_B_minus_robot"""
    anc = json.loads(ANCHORS.read_text())["tags"]; cal = load()
    pts = (json.loads(PENDING.read_text()) if PENDING.exists() else []) + cal["points"]
    for s, tid in (("left", "400"), ("right", "401")):
        P = np.array([p["p_robot"] for p in pts if p["arm"] == s and p["target"] in (tid, "p" + tid)])
        if not len(P): continue
        sh = np.asarray(anc[tid]["p_B"])[:2] - P[:, :2].mean(0)
        cal["fit"][s]["xy_B_minus_robot"] = sh.tolist(); cal["fit"][s]["xy_B_from"] = f"anchors tag {tid} vs {len(P)} touches (spread {np.round(P[:, :2].std(0) * 1e3, 1).tolist()} mm)"
        print(f"{s}: tag {tid} anchors {np.round(np.asarray(anc[tid]['p_B'])[:2] * 1e3, 1)} robot {np.round(P[:, :2].mean(0) * 1e3, 1)} (n {len(P)}) -> xy shift {np.round(sh * 1e3, 1)} mm")
    OUT.write_text(json.dumps(cal, indent=1))


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else None


def measure(a):
    from visual_teleop import config as vcfg
    from visual_teleop.robot.bridge import HttpBridge
    anc = json.loads(ANCHORS.read_text())["tags"]
    arms, _ = core.make_arms(); br = HttpBridge(vcfg.load().robot["base_url"], live=False, timeout_s=0.5)
    st = br.status(); parked = st.get("parked") or []
    print(f"브리지: {st.get('status')}  park(토크 꺼짐): {parked}", flush=True)
    if set(parked) != set(core.SIDES): print("주의: 두 팔 다 park(토크 꺼짐) 상태가 아님 -- UI 에서 '휴식(park)' 먼저", flush=True)
    print("측정 방법: 집게를 닫은 채로 집게 끝(두 손가락 끝 가운데)을 대상에 살짝 대고 Enter.\n"
          f"  대상: 400 / 401 = 태그 중심 (70 mm 태그), t = 테이블 아무 곳.  팔: l / r.  예) 'l 400', 'r t', 'u' = 마지막 취소, 'q' = 저장 후 끝", flush=True)
    pts = []
    while True:
        try: cmd = input("> ").strip().lower().split()
        except EOFError: cmd = ["q"]
        if not cmd: continue
        if cmd[0] == "q": break
        if cmd[0] == "u":
            if pts: print("취소:", pts.pop()["target"], flush=True)
            continue
        if cmd[0] not in ("l", "r") or len(cmd) < 2 or cmd[1] not in ("400", "401", "t"):
            print("형식: 'l 400' / 'r 401' / 'l t'", flush=True); continue
        s = "left" if cmd[0] == "l" else "right"; tgt = "table" if cmd[1] == "t" else cmd[1]
        qs = []
        for _ in range(10):                      # 0.5 s of reads: a hand-held arm wobbles a little
            qs.append(np.asarray(br.observe().q[s], np.float64)); time.sleep(0.05)
        q = np.median(qs, 0); T = core.fk(arms, s, q); p = T[:3, 3]
        exp = list(anc[tgt]["p_B"]) if tgt != "table" else [float(p[0]), float(p[1]), TABLE_Z]
        pts.append(dict(arm=s, target=tgt, q=q.tolist(), p_robot=p.tolist(), p_expected=exp, q_spread_deg=round(float(np.degrees(np.ptp(qs, 0)).max()), 2),
                        t=time.strftime("%H:%M:%S")))
        d = (p - np.asarray(exp)) * 1e3
        print(f"  {s} {tgt}: TCP ({p[0]*1e3:.0f}, {p[1]*1e3:.0f}, {p[2]*1e3:.1f}) mm | 데이터 기준과 차이 "
              + (f"dx {d[0]:+.1f} dy {d[1]:+.1f} " if tgt != "table" else "") + f"dz {d[2]:+.1f} mm"
              + (f"  (흔들림 {pts[-1]['q_spread_deg']}°)" if pts[-1]["q_spread_deg"] > 0.5 else ""), flush=True)
    if not pts: print("측정 없음 -- 저장 안 함"); return
    if OUT.exists(): shutil.copy(OUT, OUT.with_suffix(".json.prev"))
    cal = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), anchors=str(ANCHORS), table_z=TABLE_Z, points=pts, fit=fit(pts))
    OUT.write_text(json.dumps(cal, indent=1)); show(cal)


PENDING = HERE / "height_calib_pending.json"


def point(arm, tgt):
    """one touch, non-interactive (the operator touches, Claude records): appends to height_calib_pending.json"""
    from visual_teleop import config as vcfg
    from visual_teleop.robot.bridge import HttpBridge
    anc = json.loads(ANCHORS.read_text())["tags"]; arms, _ = core.make_arms()
    br = HttpBridge(vcfg.load().robot["base_url"], live=False, timeout_s=0.5); s = {"l": "left", "r": "right"}[arm]
    qs = []
    for _ in range(10): qs.append(np.asarray(br.observe().q[s], np.float64)); time.sleep(0.05)
    q = np.median(qs, 0); p = core.fk(arms, s, q)[:3, 3]; tg = "table" if tgt == "t" else tgt
    # 11-01 pivot: "p400" = jaw tip held on the tag-400 centre while the wrist is turned (TCP / joint-offset fit, kin_calib.py)
    exp = list(anc[tg.lstrip("p")]["p_B"]) if tg != "table" else [float(p[0]), float(p[1]), TABLE_Z]
    pts = json.loads(PENDING.read_text()) if PENDING.exists() else []
    pts.append(dict(arm=s, target=tg, q=q.tolist(), p_robot=p.tolist(), p_expected=exp, q_spread_deg=round(float(np.degrees(np.ptp(qs, 0)).max()), 2), t=time.strftime("%H:%M:%S")))
    PENDING.write_text(json.dumps(pts, indent=1)); d = (p - np.asarray(exp)) * 1e3
    print(f"#{len(pts)} {s} {tg}: TCP ({p[0]*1e3:.0f}, {p[1]*1e3:.0f}, {p[2]*1e3:.1f}) mm | " + (f"dx {d[0]:+.1f} dy {d[1]:+.1f} " if tg != "table" else "")
          + f"dz {d[2]:+.1f} mm | 흔들림 {pts[-1]['q_spread_deg']}°")


def finish():
    pts = json.loads(PENDING.read_text()) if PENDING.exists() else []
    if not pts: print("기록된 점 없음"); return
    if OUT.exists(): shutil.copy(OUT, OUT.with_suffix(".json.prev"))
    cal = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), anchors=str(ANCHORS), table_z=TABLE_Z, points=pts, fit=fit(pts))
    OUT.write_text(json.dumps(cal, indent=1)); PENDING.rename(PENDING.with_suffix(f".done_{time.strftime('%H%M%S')}.json")); show(cal)


def show(cal=None):
    cal = cal or load()
    if not cal: print("height_calib.json 없음 -- 먼저 measure"); return
    print(f"높이 캘리브레이션 {cal['created']}  ({len(cal['points'])} 점)")
    for s, f in cal["fit"].items():
        tg = "  ".join(f"태그 {k}: d {v['d_mm']} mm (n {v['n']})" for k, v in f["tags"].items())
        print(f"  {s}: 테이블 높이 차 평균 {f['dz_mean_mm']:+.1f} mm, 기울기 {f['tilt_deg']}°, 잔차 최대 {f['resid_mm']} mm, n {f['n']}  {tg}")
    print("  (+ = 실제 테이블이 학습 좌표보다 높음 -> 배포 시 그만큼 올림)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["measure", "show", "point", "finish", "camxy", "anchorxy"]); ap.add_argument("args", nargs="*"); a = ap.parse_args()
    if a.cmd == "measure": measure(a)
    elif a.cmd == "point": point(*a.args)        # point l 400 | point r 401 | point l t
    elif a.cmd == "finish": finish()
    elif a.cmd == "camxy": camxy()
    elif a.cmd == "anchorxy": anchorxy()
    else: show()
