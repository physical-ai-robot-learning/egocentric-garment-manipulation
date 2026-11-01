"""BIMANUAL UMI replay (2026-10-20, user: "양손 동시에 플레이"): both UMI trajectories of one take replayed on both arms at once, on the
same source clock, in ABSOLUTE robot-base positions with the human wrist orientation (umi_gripper_align[_left]_v001.json), bridge joint
limits inside the IK (--limit-ik behaviour of umi_replay), table floor, READY-height approach / retreat waypoints, and the jaw of each
UMI (wrist-camera classifier, <take>/wrist[_left]/umi_jaw.npz) sent --grip-lead-ms early. Extra check for two arms: the minimum distance
between any link origin of the left arm and any of the right arm (every 5th step and on the approach / retreat ramps) >= --arm-gap.
Plans everything before moving; dry-run unless --go. Logs <run>/right/seg_XX.npz and <run>/left/seg_XX.npz (+ meta.json each) in the
umi_replay format, so scripts/evaluation/replay_parity.py can score each arm:  replay_parity.py <take> <run>/right  /  --hand 왼손 <take> <run>/left
    ~/xvla-mac/bin/python -m visual_teleop.tools.umi_replay_bi datasets/tag_rig/<take> --rep 1 [--traj _fused] [--speed 0.5] [--offline]
    ... --go       (사람이 옆에 있을 때만; 두 팔 모두 READY 에서 시작)"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams
from .robot_moves import READY_Q_DEG, say, unpark
from .umi_jaw import grip_leads, grip_signal, sample_grip
from .umi_replay import load, stroke_path

ROOT = Path(__file__).resolve().parents[3]
HAND = {"right": "오른손", "left": "왼손"}


def ramp_both(br, qa: dict, qb: dict, g: dict, deg_s=10.0, hz=30.0):
    d = {s: qb[s] - qa[s] for s in qa}; n = max(2, int(np.ceil(max(np.degrees(np.abs(v)).max() for v in d.values()) / deg_s * hz))); t0 = time.monotonic()
    for k in range(1, n + 1):
        u = k / n; u = u * u * (3 - 2 * u)
        r = br.send({s: qa[s] + u * d[s] for s in qa}, g)
        if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
        if k % int(hz) == 0 and br.observe().estop: br.estop(); sys.exit("이동 중 E-STOP/읽기 지연 -- 정지")
        time.sleep(max(0.0, t0 + k / hz - time.monotonic()))


def stream_path(br, path: list, g: dict, deg_s=10.0, hz=30.0):
    """10-21: follow a dense joint path (list of {side: q}) at <= deg_s (max over joints), resampled by joint arc length -- the
    Cartesian approach/retreat legs (joint-space lines between far poses took links under the table)."""
    sides = list(path[0]); d = [0.0] + [max(np.degrees(np.abs(path[i + 1][s] - path[i][s])).max() for s in sides) for i in range(len(path) - 1)]
    cum = np.cumsum(d); n = max(2, int(np.ceil(cum[-1] / deg_s * hz))); t0 = time.monotonic()
    for k in range(1, n + 1):
        u = k / n; u = u * u * (3 - 2 * u); x = u * cum[-1]; i = int(np.clip(np.searchsorted(cum, x) - 1, 0, len(path) - 2)); f = (x - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
        r = br.send({s: path[i][s] + f * (path[i + 1][s] - path[i][s]) for s in sides}, g)
        if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
        if k % int(hz) == 0 and br.observe().estop: br.estop(); sys.exit("이동 중 E-STOP/읽기 지연 -- 정지")
        time.sleep(max(0.0, t0 + k / hz - time.monotonic()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--rep", type=int, required=True); ap.add_argument("--traj", default="_fused", choices=["", "_slam", "_fused", "_robot", "_fused_robot"])
    ap.add_argument("--speed", type=float, default=0.5); ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--smooth", type=int, default=15); ap.add_argument("--q-smooth", type=int, default=9)
    ap.add_argument("--max-deg-s", type=float, default=130.0); ap.add_argument("--max-rot-dev", type=float, default=20.0)
    ap.add_argument("--tip-clear", type=float, default=0.005, help="집게 손가락 아래 모서리의 테이블 위 최소 높이 (m); 끄려면 생략 불가 -> 매우 음수"); ap.add_argument("--jaw-half", type=float, default=0.020); ap.add_argument("--close-at-min", type=int, default=0, help="집게 닫힘을 앞뒤 구간에서 손이 가장 낮은 순간으로 (1=켬)"); ap.add_argument("--tip-clear-left-extra", type=float, default=0.005, help="왼팔만 손가락 모서리 여유를 더 (m) -- 10-21 데모1 전체 재생에서 왼팔이 테이블을 많이 긁음"); ap.add_argument("--stale-wait", type=float, default=30.0, help="읽기 지연(USB 재연결) 때 재생을 멈추고 기다리는 최대 시간 (s)"); ap.add_argument("--max-ik-mm", type=float, default=10.0, help="IK 위치 오차 합격선 (mm) -- 정확도 기준이지 안전 기준이 아님"); ap.add_argument("--window", help="데모 시작 기준 이 구간(초, UMI 시간)만 재생: 예 19.5,24"); ap.add_argument("--only", choices=["right", "left"], help="이 팔만 재생, 다른 팔은 READY 고정 (거리 검사는 그대로)"); ap.add_argument("--roll-flip", default="auto", choices=["auto", "off", "on"], help="평행 집게 도구축 180° 대칭: 데모마다 IK 가 더 잘 되는 쪽"); ap.add_argument("--roll-thr", type=float, default=115.0, help="j6 가 이 각도(도)를 넘으면 도구축 회전을 양보 (180 = 끔)"); ap.add_argument("--roll-relax", type=float, default=1.0); ap.add_argument("--lift", type=float, default=0.08, help="접근/복귀 경유점: 시작·끝점 위로 이만큼 (m)"); ap.add_argument("--ik-step", type=float, default=5.0, help="프레임당 관절 이동 상한 (도). 10-21: 180 이던 시절 j6 가 ±130 한계에 닿으면 IK 가 다른 자세로 튐 (2500°/s)"); ap.add_argument("--dump", help="계획(관절·목표·링크 높이)을 npz 로 저장 (진단용)"); ap.add_argument("--level-max", type=float, default=0.0, help="손가락 모서리가 테이블에 걸리는 프레임만 도구축을 수직쪽으로 최대 이만큼(도) 세워서 TCP 를 덜 올림 (0 = 끔) -- 10-21"); ap.add_argument("--ori-cost", type=float, default=0.1, help="IK 자세 가중치 (기본 설정값 0.3; 낮추면 위치 우선 -- 10-21 robotization)")
    ap.add_argument("--table-z", type=float, default=0.003); ap.add_argument("--tcp-margin", type=float, default=0.010); ap.add_argument("--link-margin", type=float, default=0.025)   # 10-21: 30/50 mm kept the jaws off the cloth
    ap.add_argument("--arm-gap", type=float, default=0.12); ap.add_argument("--grip-lead-ms", type=float); ap.add_argument("--grip-lead-open-ms", type=float); ap.add_argument("--grip-lead-close-ms", type=float); ap.add_argument("--no-gripper", action="store_true"); ap.add_argument("--grip-mode", default="state", choices=["state", "cont"], help="cont = 연속 집게값")
    ap.add_argument("--go", action="store_true"); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    if a.offline and a.go: sys.exit("--offline 과 --go 는 같이 못 씀")
    c = vcfg.load(); E = Path(a.take); sides = ("right", "left")
    dc = ROOT / "configs/robot/rebot_mit_deploy_v001.json"
    a.grip_lead = list(grip_leads(a, dc))
    br = None if a.offline else HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        qr = {s: np.radians(READY_Q_DEG[s]) for s in sides}
        if a.offline:
            from ego_teleop.robot.rebot_client import SIDE_SLICE
            jda = np.zeros(14)
            for s in sides: jda[SIDE_SLICE[s]] = READY_Q_DEG[s]
            q_now = qr
        else:
            ob = br.observe()
            if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다 (설치 전 점검은 --offline)")
            jda, q_now = ob.joints_deg_all, ob.q
        for s in sides:
            if a.only and s != a.only: continue      # 10-21: the idle arm is held where it is (checked against that pose below)
            if np.degrees(np.abs(q_now[s] - qr[s])).max() > 5: sys.exit(f"{s} 팔이 READY 자세가 아님 -- 먼저 ready_pose --side {s} --go")
        import pinocchio as pin
        plan = {}
        for s in sides:
            pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(jda); free = PinkIkSolver(s, PinkParams(max_joint_delta_deg=a.ik_step, iters=200, **({"orientation_cost": a.ori_cost} if a.ori_cost is not None else {}))); free.sync(jda)
            L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad); T_r = pk.fk(qr[s])
            for k_, iq in enumerate(free.ridx):
                jo_ = free.joff[s][k_]      # 10-21: bridge limits are in bridge convention; the model q carries the joint offset
                free.model.lowerPositionLimit[iq] = max(free.model.lowerPositionLimit[iq], lo[k_] + jo_ + np.radians(0.5)); free.model.upperPositionLimit[iq] = min(free.model.upperPositionLimit[iq], hi[k_] + jo_ - np.radians(0.5))
            al = ROOT / ("configs/calibration/umi_gripper_align_v001.json" if s == "right" else "configs/calibration/umi_gripper_align_left_v001.json")
            if not al.exists(): sys.exit(f"{s}: UMI-그리퍼 자세 정렬 없음 ({al.name}) -- tag_rig_record --proto align + umi_align.py --hand {s} --write")
            R_UE = np.array(json.loads(al.read_text())["R_U_E"])
            t, p, segs, Rm = load(E, s, a.traj); sg = [g for g in segs if g["rep"] == a.rep]
            if not sg: sys.exit(f"{a.rep}회 구간 없음")
            sg = sg[0]; src, P, RV, note = stroke_path(t, p, sg, a.hz, a.speed, a.smooth, Rm)
            if src is None: sys.exit(f"{s}: {note}")
            if a.window:      # 10-21: replay only [w0, w1] s (UMI source time from the demo start) -- e.g. one grasp + lift
                w0, w1 = (float(v) for v in a.window.split(",")); ts_ = (src - src[0]) / 1e9; mw = (ts_ >= w0) & (ts_ <= w1)
                src, P = src[mw], P[mw]; RV = RV[mw] if RV is not None else None
            off = stroke_path.p0 - T_r[:3, 3]; floor = a.table_z + a.tcp_margin; zt = T_r[2, 3] + P[:, 2] + off[2]; n_clip = int((zt < floor).sum())
            P = P.copy(); P[:, 2] = np.maximum(zt, floor) - T_r[2, 3] - off[2]; R0 = stroke_path.R0
            Rz180 = np.diag([-1.0, -1.0, 1.0])     # 10-21: parallel jaws are symmetric about the tool axis (= j6 axis, TCP on it)
            def Rz(ph): c_, s_ = np.cos(ph), np.sin(ph); return np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
            j6_hi = np.radians(a.roll_thr)
            jd_ = E / ("wrist" if s == "right" else "wrist_left") / "umi_jaw.npz"     # jaw opening per frame (for the finger-corner floor)
            Gpre = sample_grip(np.load(jd_)["t_ns"], grip_signal(np.load(jd_), a.grip_mode), src, 0, 0) if jd_.exists() else np.ones(len(src))
            def corner_lift(T, g_):
                # 10-21: the right arm scraped the table at TCP 19 mm with the tool tilted 46 deg down: the finger lower edge is +-20 mm
                # (TCP x) and the open jaws +-36 mm (TCP y) from the TCP centre. Raise the target so the lowest finger corner keeps --tip-clear.
                low = T[2, 3] - a.jaw_half * abs(T[2, 0]) - (0.008 + 0.028 * float(np.clip(g_, 0, 1))) * abs(T[2, 1])
                return max(0.0, a.table_z + a.tip_clear + (a.tip_clear_left_extra if s == "left" else 0.0) - low)
            def level(T, g_):
                # 10-21 (171057 demo2: jaws shut 20-50 mm above the cloth because corner_lift raised the tilted tool): only where the
                # lowest finger corner would hit the floor, tilt the tool axis toward straight down by the SMALLEST angle (<= --level-max)
                # that clears it, keeping the TCP position (= human height). Elsewhere the human orientation is untouched.
                if a.level_max <= 0 or a.tip_clear is None or corner_lift(T, g_) <= 0: return T, 0.0
                ax = T[:3, 2]; c = np.cross(ax, [0.0, 0.0, -1.0]); n_ = np.linalg.norm(c)
                if n_ < 1e-6: return T, 0.0
                amax = min(np.arccos(np.clip(-ax[2], -1, 1)), np.radians(a.level_max)); u = c / n_
                def rot(al): T2 = T.copy(); T2[:3, :3] = Rot.from_rotvec(u * al).as_matrix() @ T[:3, :3]; return T2
                if corner_lift(rot(amax), g_) > 0: return rot(amax), float(np.degrees(amax))
                lo, hi = 0.0, amax
                for _ in range(20):
                    md = (lo + hi) / 2; lo, hi = (md, hi) if corner_lift(rot(md), g_) > 0 else (lo, md)
                return rot(hi), float(np.degrees(hi))
            def solve_path(flip, f0_=None):
                # 10-21 wrist-roll robotization: j6 turns about the tool z axis (TCP on it), so R_target Rz(phi) moves j6 by ~+phi. When the
                # solved j6 passes --roll-thr the next targets give up that much tool-axis roll (phi), and phi relaxes back toward 0 at
                # <= --roll-relax deg/frame while j6 has headroom. Position is never traded.
                q = qr[s]; Q, err, worst, TT, PE, RE, PH, LV = [], 0.0, 0.0, [], [], [], [], []; phi = 0.0
                for i, x in enumerate(P + off):
                    T = T_r.copy(); T[:3, 3] += x; T[:3, :3] = Rot.from_rotvec(RV[i]).as_matrix() @ R0.as_matrix() @ R_UE @ (Rz180 if flip else np.eye(3))
                    if a.roll_thr < 180: T[:3, :3] = T[:3, :3] @ Rz(phi)
                    T, lv_ = level(T, Gpre[i]); LV.append(lv_)
                    if a.tip_clear is not None: T[2, 3] += corner_lift(T, Gpre[i])
                    T_h = T.copy()
                    if i == 0:      # 10-21: the first target is ~0.3 m from READY -> solve it unbounded, bound the steps after it
                        f0 = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=400, **({"orientation_cost": a.ori_cost} if a.ori_cost is not None else {}))); f0.sync(jda)
                        f0.model.lowerPositionLimit[:] = free.model.lowerPositionLimit; f0.model.upperPositionLimit[:] = free.model.upperPositionLimit
                        q, pe, re = f0.ik(q, T)
                    else: q, pe, re = free.ik(q, T)
                    if a.roll_thr < 180:
                        over = abs(q[5]) - j6_hi
                        if over > 0: phi -= np.sign(q[5]) * over
                        elif phi != 0: phi -= np.sign(phi) * min(abs(phi), np.radians(a.roll_relax), max(0.0, -over))
                    pe_h = float(np.linalg.norm(T_h[:3, 3] - T[:3, 3]))     # same position; rotation error vs the HUMAN target below
                    re_h = float(np.degrees(Rot.from_matrix(T_h[:3, :3].T @ free.fk(q)[:3, :3]).magnitude())) if hasattr(free, "fk") else re
                    Q.append(q); err = max(err, pe); worst = max(worst, re); TT.append(T_h); PE.append(pe); RE.append(re_h); PH.append(phi)
                solve_path.phi = np.degrees(np.array(PH)); solve_path.lv = np.array(LV)
                return np.array(Q), err, worst, TT, np.array(PE), np.array(RE)
            flips = {"off": [False], "on": [True], "auto": [False, True]}[a.roll_flip]; best = None
            for fl in flips:
                r_ = solve_path(fl); r_ = (*r_, solve_path.phi); score = (float(np.mean(r_[4] > 0.01)), r_[1])
                if len(flips) > 1 or a.roll_thr < 180:
                    ph_ = np.abs(r_[6]); print(f"  {HAND[s]}: 집게 180° 뒤집기 {'켬' if fl else '끔'} -> IK 10 mm 넘는 프레임 {score[0] * 100:.0f}%, 최대 {r_[1] * 1000:.0f} mm | 손목 회전 양보 최대 {ph_.max():.0f}°, 양보한 프레임 {np.mean(ph_ > 1) * 100:.0f}%")
                if a.level_max > 0: lv_ = solve_path.lv; print(f"  {HAND[s]}: 손가락 수평 맞춤 {np.mean(lv_ > 0.5) * 100:.0f}% 프레임, 세운 각 중앙 {np.median(lv_[lv_ > 0.5]) if (lv_ > 0.5).any() else 0:.0f}° 최대 {lv_.max():.0f}°")
                if best is None or score < best[0]: best = (score, fl, r_)
            _, flip_used, (Q, err, worst, TT, PE, RE, PHI) = best; Q_raw = Q.copy()
            if a.q_smooth > 1:
                k_ = np.ones(a.q_smooth) / a.q_smooth; Q = np.stack([np.convolve(np.pad(Q[:, j], a.q_smooth // 2, mode="edge"), k_, "valid")[:len(Q)] for j in range(6)], 1)
            fu = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=400)); fu.sync(jda)     # 10-21: approach waypoints need an UNBOUNDED solve
            fu.model.lowerPositionLimit[:] = free.model.lowerPositionLimit; fu.model.upperPositionLimit[:] = free.model.upperPositionLimit
            def above(qq, pk=pk, fu=fu, T_r=T_r):
                # 10-21: with the 5 deg/frame bound this "lift" barely moved, so READY -> start was one joint-space line that took links
                # 15 cm under the table. Lift to max(READY height, start + --lift) with an unbounded IK seeded at the start pose.
                Tq = pk.fk(qq); Tq[2, 3] = max(Tq[2, 3] + a.lift, T_r[2, 3]); return fu.ik(qq, Tq)[0]
            up0, up1 = above(Q[0]), above(Q[-1])
            fids = [i for i, f in enumerate(pk.model.frames) if f.name in (f"{s}_link3", f"{s}_link4", f"{s}_link5", f"{s}_link6", f"{s}_gripper_link", f"{s}_gripper_left", f"{s}_gripper_right", f"{s}_tcp")]
            def links(q6, pk=pk, fids=fids):     # 10-21 bug fix: without the default args every arm's closure used the LAST arm's pk/fids (late binding)
                pin.framesForwardKinematics(pk.model, pk.data, pk._q(q6)); return np.array([pk.data.oMf[i].translation for i in fids])
            G = None
            jd = E / ("wrist" if s == "right" else "wrist_left") / "umi_jaw.npz"
            if not a.no_gripper and jd.exists():
                lc = 0.0 if a.close_at_min else a.grip_lead[1]
                jz = np.load(jd); G = sample_grip(jz["t_ns"], grip_signal(jz, a.grip_mode), src, int(a.grip_lead[0] * 1e6 * a.speed), int(lc * 1e6 * a.speed))
                if a.close_at_min:
                    # 10-21 (171057 demo2: jaws closed 15-25 mm above the cloth): move every close to the LOWEST target point within
                    # [-0.5 s, +0.7 s] (source time) of it, so the arm is down before the jaws shut. Close lead = 0 here.
                    zt_ = np.array([T_[2, 3] for T_ in TT]); fps_src = len(src) / max((src[-1] - src[0]) / 1e9, 1e-6)
                    shifted = 0
                    for i_ in [k_ + 1 for k_ in np.where((G[:-1] > 0.5) & (G[1:] <= 0.5))[0]]:
                        lo_, hi_ = max(0, i_ - int(0.5 * fps_src)), min(len(G), i_ + int(0.7 * fps_src) + 1); j_ = lo_ + int(np.argmin(zt_[lo_:hi_]))
                        if j_ > i_: G[i_:j_] = G[i_ - 1]; shifted += 1
                        elif j_ < i_: G[j_:i_] = G[i_]; shifted += 1
                    print(f"  {HAND[s]}: 닫힘 {shifted}개를 가장 낮은 순간으로 옮김 (닫힘 선행 0)")
            plan[s] = dict(free=free, flip=flip_used, Q_raw=Q_raw, T_tgt=np.array(TT), pe=np.array(PE), re=np.array(RE), pk=pk, lo=lo, hi=hi, T_r=T_r, sg=sg, src=src, P=P, RV=RV, Q=Q, up=(up0, up1), off=off, links=links, G=G, err=err, worst=worst, n_clip=n_clip, p0=stroke_path.p0)
        if a.only:      # 10-21: single-arm reference replay with the bimanual planner (the other arm holds READY, link gap still checked)
            for s in sides:
                if s == a.only: continue
                hold = np.asarray(q_now[s], float); qr[s] = hold      # 10-21: hold the idle arm at its CURRENT pose (rest, READY, ...)
                x = plan[s]; m_ = len(x["Q"]); x.update(Q=np.tile(hold, (m_, 1)), Q_raw=np.tile(hold, (m_, 1)), up=(hold, hold), G=None, err=0.0, worst=0.0, n_clip=0,
                                                       pe=np.zeros(m_), re=np.zeros(m_), T_tgt=np.tile(x["pk"].fk(hold), (m_, 1, 1)))
            idle = [x for x in sides if x != a.only][0]
            print(f"  {HAND[idle]}: 지금 자세 고정 (--only {a.only}): 관절 {np.round(np.degrees(qr[idle]), 1).tolist()}")
        n = min(len(plan[s]["Q"]) for s in sides)
        for s in sides:
            for k in ("src", "P", "RV", "Q", "Q_raw", "T_tgt", "pe", "re"): plan[s][k] = plan[s][k][:n]
            if plan[s]["G"] is not None: plan[s]["G"] = plan[s]["G"][:n]
        # approach / retreat (both arms together): READY -> up0 -> Q0 ... Q-1 -> up1 -> READY
        def cart_leg(s, qa, qb, steps=60):
            """TCP straight line qa -> qb (orientation slerp), IK seeded step to step (5 deg bound), last point snapped to qb"""
            x = plan[s]; Ta, Tb = x["pk"].fk(qa), x["pk"].fk(qb); sl = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]])); q = qa.copy(); out = []
            for u in np.linspace(0, 1, steps):
                T = np.eye(4); T[:3, :3] = sl(u).as_matrix(); T[:3, 3] = (1 - u) * Ta[:3, 3] + u * Tb[:3, 3]; q = x["free"].ik(q, T)[0]; out.append(q)
            out.append(qb); return out
        def leg(qa_, qb_, cart):
            if not cart: return [{s: qa_[s] + u * (qb_[s] - qa_[s]) for s in sides} for u in np.linspace(0, 1, 20)]
            L_ = {s: cart_leg(s, qa_[s], qb_[s]) for s in sides}; m_ = max(len(v) for v in L_.values())
            return [{s: L_[s][min(i, len(L_[s]) - 1)] for s in sides} for i in range(m_)]
        Rd, U0, Q0, Q1, U1 = ({s: qr[s] for s in sides}, {s: plan[s]["up"][0] for s in sides}, {s: plan[s]["Q"][0] for s in sides},
                              {s: plan[s]["Q"][-1] for s in sides}, {s: plan[s]["up"][1] for s in sides})
        legs_q = [leg(Rd, U0, True), leg(U0, Q0, False), leg(Q1, U1, True), leg(U1, Rd, True)]     # 10-21: Cartesian legs except the short descent
        way = [({s: qr[s] for s in sides}, {s: plan[s]["up"][0] for s in sides}), ({s: plan[s]["up"][0] for s in sides}, {s: plan[s]["Q"][0] for s in sides}),
               ({s: plan[s]["Q"][-1] for s in sides}, {s: plan[s]["up"][1] for s in sides}), ({s: plan[s]["up"][1] for s in sides}, {s: qr[s] for s in sides})]
        ramp = [q for L_ in legs_q for q in L_]
        for s in sides:      # 10-21: lowest link per approach/retreat leg
            legs = ["READY->시작 위", "시작 위->시작", "끝->끝 위", "끝 위->READY"]
            lows = [min(float(plan[s]["links"](q[s])[:, 2].min()) for q in L_) for L_ in legs_q]
            print(f"  {HAND[s]} 접근/복귀 링크 최저: " + ", ".join(f"{l_} {v * 1000:.0f} mm" for l_, v in zip(legs, lows)))
        stream = [{s: plan[s]["Q"][k] for s in sides} for k in range(0, n, 5)]
        gaps = [float(np.min(np.linalg.norm(plan["left"]["links"](q["left"])[:, None] - plan["right"]["links"](q["right"])[None], axis=2))) for q in stream + ramp]
        gap = min(gaps); ig = int(np.argmin(gaps))       # 10-21: say WHERE the arms come closest (stream frame or approach/retreat ramp)
        where = f"재생 {ig * 5 / a.hz:.1f}s" if ig < len(stream) else f"접근/복귀 경로 {ig - len(stream)}/{len(ramp)}"
        qg = (stream + ramp)[ig]; Dg = np.linalg.norm(plan["left"]["links"](qg["left"])[:, None] - plan["right"]["links"](qg["right"])[None], axis=2); il, ir = np.unravel_index(np.argmin(Dg), Dg.shape)
        print(f"  두 팔 최근접: {where}, 왼 링크 {il} - 오른 링크 {ir}, 그때 TCP 왼 {np.round(plan['left']['pk'].fk(qg['left'])[:3, 3], 3)} 오른 {np.round(plan['right']['pk'].fk(qg['right'])[:3, 3], 3)}, 재생 중 120 mm 미만 {np.mean(np.array(gaps[:len(stream)]) < a.arm_gap) * 100:.0f}%")
        okall = True
        for s in sides:
            x = plan[s]; Q = x["Q"]; vel = np.degrees(np.abs(np.diff(Q, axis=0))).max() * a.hz
            low = min(float(min(l[:, 2].min() for l in [x["links"](q) for q in list(Q[::5])])), float(min(x["links"](q[s])[:, 2].min() for q in ramp)))
            lim = bool(np.any(Q < x["lo"]) or np.any(Q > x["hi"])); zmin = float(min(x["pk"].fk(q)[2, 3] for q in Q[::5]))
            ok = x["err"] < a.max_ik_mm / 1000 and x["worst"] < a.max_rot_dev and not lim and low >= a.table_z + a.link_margin - 0.02 and zmin >= a.table_z + a.tcp_margin - 0.005 and vel <= a.max_deg_s
            if a.only and s != a.only: ok = True      # 10-21: the idle arm does not move -- only the arm-arm gap below applies to it
            okall &= ok; gi = "" if x["G"] is None else f", 집게({a.grip_mode}) 바뀜 {int(np.sum(np.diff(x['G'] > 0.5) != 0))}회 (열림 {a.grip_lead[0]:.0f} / 닫힘 {a.grip_lead[1]:.0f} ms 먼저)"
            print(f"  {HAND[s]}: 시작 {np.round(x['p0'] * 1000).astype(int)} mm, {n / a.hz:.1f}s, IK 최대오차 {x['err'] * 1000:.1f} mm/{x['worst']:.1f}°, 관절속도 최대 {vel:.0f}°/s, "
                  f"TCP 최저 {zmin * 1000:.0f} mm (바닥 아래라 올린 점 {x['n_clip']}), 링크 최저 {low * 1000:.0f} mm{'  관절한계 밖' if lim else ''}{gi}  {'OK' if ok else '<- 불가'}")
        print(f"  두 팔 링크 최소 거리 {gap * 1000:.0f} mm (기준 {a.arm_gap * 1000:.0f})  {'OK' if gap >= a.arm_gap else '<- 불가 (충돌 위험)'}")
        okall &= gap >= a.arm_gap
        if a.dump:
            np.savez(a.dump, **{f"{s}_{k}": plan[s][k] for s in sides for k in ("Q", "Q_raw", "T_tgt", "pe", "re", "src")},
                     **{f"{s}_linkz": np.array([plan[s]["links"](q)[:, 2] for q in plan[s]["Q"]]) for s in sides}, gaps=np.array(gaps), hz=a.hz)
            print(f"  계획 저장 {a.dump}")
        print(f"{'실행 가능' if okall else '불가'}: {a.rep}회, {n / a.hz:.1f}초 ({a.speed}배속)")
        if not a.go or not okall: print("계획만 출력함" if okall else "이동 안 함"); return
        out = ROOT / "datasets/replay" / ("bi_" + time.strftime("%Y%m%d_%H%M%S"))
        for s in sides:
            (out / s).mkdir(parents=True, exist_ok=True)
            (out / s / "meta.json").write_text(json.dumps(dict(take=str(E), hand=HAND[s], side=s, speed=a.speed, hz=a.hz, smooth=a.smooth, T_ready=plan[s]["T_r"].tolist(), absolute=True, abs_orient=True,
                                                               table_z=a.table_z, orient=True, traj=a.traj, bimanual=True, gripper=str(E / ("wrist" if s == "right" else "wrist_left")) if plan[s]["G"] is not None else None,
                                                               grip_lead_ms=a.grip_lead, grip_mode=a.grip_mode, segments=[plan[s]["sg"]]), ensure_ascii=False, indent=1))
        say("양손 재생을 시작합니다" if not a.only else ("오른팔" if a.only == "right" else "왼팔") + " 재생을 시작합니다")
        for s in sides: unpark(br, s)
        ob = br.observe(); g0 = {s: (float(plan[s]["G"][0]) if plan[s]["G"] is not None else ob.g(s)) for s in sides}
        stream_path(br, [{s: br.observe().q[s] for s in sides}] + legs_q[0][1:], g0)
        stream_path(br, legs_q[1], g0); time.sleep(1.5); say(f"{a.rep}회")
        tc, to, qo, go = [], [], {s: [] for s in sides}, {s: [] for s in sides}; t0 = time.monotonic()
        for k in range(n):
            gk = {s: (float(plan[s]["G"][k]) if plan[s]["G"] is not None else g0[s]) for s in sides}
            r = br.send({s: plan[s]["Q"][k] for s in sides}, gk); tc.append(time.monotonic_ns())
            if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
            o = br.observe()
            if o.estop:
                # 10-21: a USB re-enumeration (bridge auto-reconnect) makes reads stale for a few seconds. An operator E-STOP still
                # stops at once; a stale read pauses the replay clock up to --stale-wait s and resumes when reads are fresh again.
                if br.status().get("estop"): br.estop(); sys.exit("E-STOP (운전자) -- 정지")
                tw = time.monotonic(); print(f"  [{k / a.hz:.1f}s] 읽기 지연 (USB 재연결 대기, 최대 {a.stale_wait:.0f}s)", flush=True)
                while time.monotonic() - tw < a.stale_wait:
                    time.sleep(0.2); o = br.observe()
                    if not o.estop: break
                if o.estop: br.estop(); sys.exit(f"{a.stale_wait:.0f}s 동안 복구 안 됨 -- E-STOP")
                t0 += time.monotonic() - tw; print(f"  통신 복구 ({time.monotonic() - tw:.1f}s 정지) -- 이어서 재생", flush=True)
            to.append(o.t_ns)
            for s in sides: qo[s].append(o.q[s].copy()); go[s].append(o.g(s))
            time.sleep(max(0.0, t0 + (k + 1) / a.hz - time.monotonic()))
        for _ in range(int(a.hz)):
            o = br.observe(); to.append(o.t_ns)
            for s in sides: qo[s].append(o.q[s].copy()); go[s].append(o.g(s))
            time.sleep(1 / a.hz)
        for s in sides:
            x = plan[s]; pk = x["pk"]; T_r = x["T_r"]; Qo = np.array(qo[s])
            np.savez(out / s / "seg_00.npz", src_t_ns=x["src"], t_cmd_ns=np.array(tc), q_cmd=x["Q"], q_send=x["Q"], start_offset=x["off"], lag_ms=np.zeros(6),
                     p_cmd=np.array([pk.fk(q)[:3, 3] for q in x["Q"]]) - T_r[:3, 3] - x["off"], t_obs_ns=np.array(to), q_obs=Qo,
                     p_obs=np.array([pk.fk(q)[:3, 3] for q in Qo]) - T_r[:3, 3] - x["off"],
                     rot_cmd=np.array([Rot.from_matrix(pk.fk(q)[:3, :3] @ T_r[:3, :3].T).as_rotvec() for q in x["Q"]]),
                     rot_obs=np.array([Rot.from_matrix(pk.fk(q)[:3, :3] @ T_r[:3, :3].T).as_rotvec() for q in Qo]),
                     g_cmd=(x["G"] if x["G"] is not None else np.full(n, g0[s])), g_obs=np.array(go[s]),
                     T_tgt_B=x["T_tgt"], ik_err_m=x["pe"], ik_rot_deg=x["re"], flip=bool(x.get("flip", False)))     # 10-21: robotized target (A') for A/B/C
        stream_path(br, legs_q[2], g0)
        stream_path(br, legs_q[3], g0)
        say("양손 재생 끝났습니다"); print(f"저장 {out}\n다음: .venv/bin/python scripts/evaluation/replay_parity.py {E} {out}/right ; .venv/bin/python scripts/evaluation/replay_parity.py {E} {out}/left --hand 왼손")
    finally:
        if br is not None: br.close()


if __name__ == "__main__":
    main()
