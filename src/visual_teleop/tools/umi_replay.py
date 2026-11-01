"""Replay UMI translation strokes on the robot (2026-10-20, GPT plan step 6: B = robot commanded, C = robot executed). Translation only:
target = READY TCP + (p_B(t) - p_B(hold before the stroke)), orientation and jaw fixed at READY, time stretched by 1/--speed (default 0.5x).
10-20 offline plan: 15 cm back / down from READY hits joint limits and 40+ deg/s, so by default each stroke is CENTRED on READY (start =
READY - half the stroke's main displacement; --no-center = start at READY). Parity uses relative motion only, so this does not change it.
Input = scripts/pose/umi_tcp_traj.py output of an --proto axes take (needs p_B, i.e. robot_cam_v001.json) + its segments.json.
Every step is IK-planned before anything moves (IK error, bridge joint limits, TCP >= --min-z, joint speed <= --max-deg-s); dry-run prints
the plan, --go streams it at --hz through HttpBridge.send (never a hand POST) and returns to READY between strokes.
    ~/xvla-mac/bin/python -m visual_teleop.tools.umi_replay datasets/tag_rig/<take> --hand 오른손 [--axis X] [--rep 1]          # 계획만
    ~/xvla-mac/bin/python -m visual_teleop.tools.umi_replay datasets/tag_rig/<take> --hand 오른손 --go                          # 실제 (사람이 옆에 있을 때만)
-> datasets/replay/<ts>/: meta.json (segments, speed, READY TCP) + seg_XX.npz (src_t_ns, t_cmd_ns, q_cmd, p_cmd, t_obs_ns, q_obs, p_obs)
then: .venv/bin/python scripts/evaluation/replay_parity.py datasets/tag_rig/<take> datasets/replay/<ts>"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .. import config as vcfg
from ..robot.bridge import HttpBridge
from ..robot.ik import ArmIk
from ..robot.ik_pink import PinkIkSolver, PinkParams
from .robot_moves import READY_Q_DEG, ramp_to, say, unpark

ROOT = Path(__file__).resolve().parents[3]
SIDE = {"오른손": "right", "왼손": "left"}


def load(take: Path, side: str, traj: str = ""):
    d = take if (take / f"umi_traj_{side}.npz").exists() else take.parent / (take.name + "_derived")
    z = np.load(d / f"umi_traj_{side}{traj}.npz", allow_pickle=True)          # 10-20: traj "_slam" = scripts/slam_vs_tag.py --save output
    if "p_B" not in z: sys.exit("궤적에 로봇 좌표(p_B)가 없음 -- robot_cam_solve --write 후 umi_tcp_traj 다시")
    return z["t_ns"].astype(np.int64), z["p_B"], json.loads((d / "segments.json").read_text()), (z["R_B_tcp"] if "R_B_tcp" in z else None)


def stroke_path(t, p, sg, hz, speed, smooth, Rm=None, skip_s=0.0):
    """UMI relative displacement over [hold_pre mid, hold_post mid] on a uniform robot-time grid (src time = robot time * speed)"""
    a = (sg["hold_pre"][0] + sg["hold_pre"][1]) // 2 + int(skip_s * 1e9); b = (sg["hold_post"][0] + sg["hold_post"][1]) // 2 if sg["hold_post"] else sg["back"][1]
    m = (t >= a) & (t <= b)
    if m.sum() < 10: return None, None, "UMI 프레임 부족"
    tt, pp = t[m], p[m]; gap = np.diff(tt).max() / 1e9 if len(tt) > 1 else 9
    h0 = (t >= sg["hold_pre"][0]) & (t <= sg["hold_pre"][1]); p0 = np.median(p[h0], 0) if h0.sum() >= 3 else pp[0]
    src = np.arange(a, b, int(1e9 * speed / hz)); P = np.stack([np.interp(src, tt, pp[:, j]) for j in range(3)], 1) - p0
    RV = None
    if Rm is not None:     # 10-20 orientation: SPATIAL relative rotation dR = R(t) R0^T in the robot base (independent of the UMI body frame)
        R0 = Rot.from_matrix(Rm[h0]).mean() if h0.sum() >= 3 else Rot.from_matrix(Rm[m][0])
        rv = (Rot.from_matrix(Rm[m]) * R0.inv()).as_rotvec(); RV = np.stack([np.interp(src, tt, rv[:, j]) for j in range(3)], 1)
        stroke_path.R0 = R0                  # 10-20 --abs-orient: absolute UMI orientation = Rot(RV) * R0
    if smooth > 1:
        k = np.ones(smooth) / smooth; sm = lambda X: np.stack([np.convolve(np.pad(X[:, j], smooth // 2, mode="edge"), k, "valid")[:len(X)] for j in range(3)], 1)
        P = sm(P); RV = sm(RV) if RV is not None else None
    stroke_path.p0 = p0                      # 10-20 --absolute: the stroke start in the robot base
    return src, P, RV, (f"끊김 {gap:.2f}s" if gap > 0.4 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--hand", default="오른손", choices=list(SIDE)); ap.add_argument("--axis", choices=["X", "Y", "Z", "YAW", "PITCH", "ROLL", "FREE"]); ap.add_argument("--orient", action="store_true", help="손목 회전도 재생 (기본: READY 방향 고정)"); ap.add_argument("--rep", type=int)
    ap.add_argument("--speed", type=float, default=0.5); ap.add_argument("--hz", type=float, default=30.0); ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--max-deg-s", type=float, default=30.0); ap.add_argument("--min-z", type=float, default=0.10); ap.add_argument("--obs-every", type=int, default=3)
    ap.add_argument("--lag-ms", type=float, nargs=6, help="관절별 지연 보상 (ms, 명령을 그만큼 앞당김)"); ap.add_argument("--absolute", action="store_true", help="절대 위치 재생: UMI 집게 끝의 로봇 좌표 그대로 (회전은 READY 기준 상대)")
    ap.add_argument("--skip-start", type=float, default=0.0, help="구간 앞부분(초, UMI 시간)을 건너뜀 -- 시작 정지 자세가 관절 한계 밖일 때")
    ap.add_argument("--limit-ik", action="store_true", help="IK 안에서 브리지 관절 한계를 지킴 (위치 우선, 자세 일부 양보; --max-rot-dev 까지 허용)")
    ap.add_argument("--max-rot-dev", type=float, default=20.0)
    ap.add_argument("--clamp-limits", action="store_true", help="관절을 브리지 한계 안으로 잘라냄 (그로 인한 자세 차이를 출력, 위치 15 mm / 자세 15도 넘으면 불가)")
    ap.add_argument("--abs-orient", action="store_true", help="손목 자세도 절대 (umi_gripper_align_v001.json 필요)")
    ap.add_argument("--abs-axes", default="xyz", help="절대로 맞출 축 (예: z = 높이만 절대, x/y 는 READY 중심 상대)")
    ap.add_argument("--table-z", type=float, default=0.003, help="작업대 높이 (로봇 베이스 z, m; 10-20 Orbbec depth 측정 +3 mm)")
    ap.add_argument("--tcp-margin", type=float, default=0.03); ap.add_argument("--link-margin", type=float, default=0.05)
    ap.add_argument("--traj", default="", choices=["", "_slam", "_fused", "_robot"], help="_slam = 손목 SLAM 으로 정렬한 궤적 (slam_vs_tag --save)"); ap.add_argument("--grip-lead-ms", type=float, default=None, help="집게 명령을 이만큼 일찍 (기본: configs/robot/rebot_mit_deploy_v001.json gripper_lead_ms)")
    ap.add_argument("--grip-lead-open-ms", type=float); ap.add_argument("--grip-lead-close-ms", type=float); ap.add_argument("--grip-mode", default="state", choices=["state", "cont"], help="cont = 연속 집게값 (0.1~0.3 사이를 부드럽게)"); ap.add_argument("--grip-only", action="store_true", help="팔은 READY 고정, 집게만 UMI 시점대로 (집게 타이밍 시험)"); ap.add_argument("--gripper", help="umi_jaw record 폴더 (umi_jaw.npz): UMI 집게 열림/닫힘을 로봇 집게로 재생"); ap.add_argument("--no-center", action="store_true"); ap.add_argument("--q-smooth", type=int, default=1, help="IK 관절궤적 이동평균 (자유동작 9 권장)"); ap.add_argument("--go", action="store_true"); ap.add_argument("--offline", action="store_true", help="로봇 없이 READY 에서 계획만")
    a = ap.parse_args(); c = vcfg.load(); s = SIDE[a.hand]
    if a.offline and a.go: sys.exit("--offline 과 --go 는 같이 못 씀")
    if s not in READY_Q_DEG: sys.exit(f"{s} 팔의 READY 자세가 아직 없음 (robot_moves.READY_Q_DEG)")
    t, p, segs, Rm = load(Path(a.take), s, a.traj)
    R_UE = None
    if a.abs_orient:
        al = ROOT / "configs/calibration/umi_gripper_align_v001.json"
        if not al.exists(): sys.exit("UMI-그리퍼 자세 정렬 없음 -- tag_rig_record --proto align 후 scripts/retarget/umi_align.py --write")
        R_UE = np.array(json.loads(al.read_text())["R_U_E"]); a.orient = True
    J = None
    if a.gripper:          # 10-20 gripper timing: UMI jaw state (hysteresis, umi_jaw infer) -> robot g (1 open / 0 closed) at the same source time
        from visual_teleop.tools.umi_jaw import grip_signal
        jz = np.load(Path(a.gripper) / "umi_jaw.npz"); J = (jz["t_ns"].astype(np.int64), grip_signal(jz, a.grip_mode))
        print(f"집게 재생: {a.gripper} (사건 {len(jz['ev_t'])}개)")
        if a.obs_every > 1: a.obs_every = 1; print("집게 재생: 로봇 관측을 매 스텝(30 Hz)으로 (집게 지연 측정 해상도)")
    if a.orient and Rm is None: sys.exit("궤적에 회전(R_B_tcp)이 없음 -- umi_tcp_traj 다시")
    segs = [g for g in segs if g["hand"] in (a.hand, "양손") and (a.axis is None or g["axis"] == a.axis) and (a.rep is None or g["rep"] == a.rep)]
    if not segs: sys.exit("선택한 구간 없음")
    br = None if a.offline else HttpBridge(c.robot["base_url"], live=a.go, timeout_s=c.robot["request_timeout_s"])
    try:
        qr = np.radians(READY_Q_DEG[s])
        if a.offline:
            from ego_teleop.robot.rebot_client import SIDE_SLICE
            jda = np.zeros(14); jda[SIDE_SLICE[s]] = READY_Q_DEG[s]; q_now = qr
        else:
            ob = br.observe()
            if ob.estop: sys.exit("로봇 상태가 E-STOP/미연결/읽기 지연입니다 (설치 전 점검은 --offline)")
            jda, q_now = ob.joints_deg_all, ob.q[s]
        if np.degrees(np.abs(q_now - qr)).max() > 5: sys.exit("팔이 READY 자세가 아님 -- 먼저 visual_teleop.tools.ready_pose --go")
        pk = PinkIkSolver(s, PinkParams(**c.ik["pink"])); pk.sync(jda)
        free = PinkIkSolver(s, PinkParams(max_joint_delta_deg=180, iters=200)); free.sync(jda)
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; lo, hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad); T_r = pk.fk(qr)
        if a.limit_ik:   # 10-20: the URDF allows j4 past the bridge's -80 deg; give Pink the BRIDGE limits so it trades orientation instead
            for k_, iq in enumerate(free.ridx):
                free.model.lowerPositionLimit[iq] = max(free.model.lowerPositionLimit[iq], lo[k_] + np.radians(0.5))
                free.model.upperPositionLimit[iq] = min(free.model.upperPositionLimit[iq], hi[k_] - np.radians(0.5))
        plans, bad = [], []
        for j, sg in enumerate(segs):
            src, P, RV, note = stroke_path(t, p, sg, a.hz, a.speed, a.smooth, Rm if a.orient else None, a.skip_start)
            if src is None: bad.append((j, note)); continue
            if a.grip_only: P = np.zeros_like(P); RV = None          # 10-20: gripper timing alone (the grip take's arm motion exceeded the joint limits)
            off = np.zeros(3); n_clip = 0
            if a.absolute:   # target = absolute TCP in the base: express as an offset from READY so the rest of the code is unchanged
                off = stroke_path.p0 - T_r[:3, 3]
                for k_, ax_ in enumerate("xyz"):     # 10-20: the human worked in front of the robot centre (y ~ 0), outside the right arm's reach
                    if ax_ not in a.abs_axes: off[k_] = -0.5 * (P[:, k_].min() + P[:, k_].max())
                floor = a.table_z + a.tcp_margin; zt = T_r[2, 3] + P[:, 2] + off[2]
                n_clip = int((zt < floor).sum()); P = P.copy(); P[:, 2] = np.maximum(zt, floor) - T_r[2, 3] - off[2]
            elif not a.no_center and sg["axis"] in ("X", "Y", "Z"):
                i_ax = "XYZ".index(sg["axis"]); off[i_ax] = -0.5 * P[np.argmax(np.abs(P[:, i_ax])), i_ax]
            elif not a.no_center and sg["axis"] == "FREE":
                off = -0.5 * (P.min(0) + P.max(0))          # 10-20 free motion: bounding box centred on READY
            T0 = T_r.copy(); T0[:3, 3] += off
            if R_UE is not None and RV is not None: T0[:3, :3] = Rot.from_rotvec(RV[0]).as_matrix() @ stroke_path.R0.as_matrix() @ R_UE
            q, pe0, _ = free.ik(qr, T0)          # stroke start (ramped to before streaming)
            Q, worst, err = [], 0.0, pe0
            for i_, x in enumerate(P + off):
                T = T_r.copy(); T[:3, 3] += x
                if RV is not None:
                    T[:3, :3] = (Rot.from_rotvec(RV[i_]).as_matrix() @ stroke_path.R0.as_matrix() @ R_UE) if R_UE is not None else Rot.from_rotvec(RV[i_]).as_matrix() @ T_r[:3, :3]
                q, pe, re = free.ik(q, T); Q.append(q); err = max(err, pe); worst = max(worst, re)
            Q = np.array(Q); clamp_note = ""
            if a.clamp_limits:   # 10-20: absolute orientation put j4 12 deg past -80 during the first 1.8 s (still hold) of the grip take
                Qc = np.clip(Q, lo + np.radians(1), hi - np.radians(1)); nc = int(np.any(Qc != Q, axis=1).sum())
                if nc:
                    dp_, dr_ = [], []
                    for qa, qb in zip(Q[np.any(Qc != Q, axis=1)], Qc[np.any(Qc != Q, axis=1)]):
                        Ta, Tb = pk.fk(qa), pk.fk(qb); dp_.append(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])); dr_.append(np.degrees(Rot.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude()))
                    clamp_note = f"  [한계 자름 {nc}점: 위치 최대 {max(dp_) * 1000:.0f} mm, 자세 최대 {max(dr_):.0f}°]"; err = max(err, max(dp_) if max(dp_) > 0.015 else 0); worst = max(worst, max(dr_) if max(dr_) > 15 else 0)
                Q = Qc
            if a.q_smooth > 1:   # 10-20 free take: per-step IK convergence left a staircase in j4 (0 -> 1.2 deg per step) = fake 40-70 deg/s at any speed
                k_ = np.ones(a.q_smooth) / a.q_smooth; Q = np.stack([np.convolve(np.pad(Q[:, j], a.q_smooth // 2, mode="edge"), k_, "valid")[:len(Q)] for j in range(6)], 1)
            Qs = Q
            if a.lag_ms:      # 10-20 X replay: j2 led ~-100 ms, j3 lagged ~+200 ms -> Z path error up to 45 mm; send q(t + lag_j) per joint
                tg = np.arange(len(Q)) / a.hz; Qs = np.stack([np.interp(tg + a.lag_ms[k] / 1000, tg, Q[:, k]) for k in range(6)], 1)
            vel = np.degrees(np.abs(np.diff(Qs, axis=0))).max() * a.hz if len(Q) > 1 else 0
            zmin = float(min(T_r[2, 3] + P[:, 2] + off[2])); lim = bool(np.any(Q < lo) or np.any(Q > hi)); link_low = 0.0
            if a.absolute:   # every arm link origin (link4 .. tcp) must stay above the table + --link-margin, also on the ramp from READY
                import pinocchio as pin
                fids = [i for i, f in enumerate(pk.model.frames) if f.name in (f"{s}_link4", f"{s}_link5", f"{s}_link6", f"{s}_gripper_link", f"{s}_gripper_left", f"{s}_gripper_right")]
                def lowest(q6):
                    pin.framesForwardKinematics(pk.model, pk.data, pk._q(q6)); return min(pk.data.oMf[i].translation[2] for i in fids)
                # 10-20: a straight joint ramp READY -> start dipped the TCP to 24 mm. Waypoints at READY height above the start / end instead
                def above(qq):
                    Tq = pk.fk(qq); Tq[2, 3] = max(Tq[2, 3], T_r[2, 3]); return free.ik(qq, Tq)[0]
                up0, up1 = above(Q[0]), above(Q[-1]); way = [(qr, up0), (up0, Q[0]), (Q[-1], up1), (up1, qr)]
                ramp = [qa + u * (qb - qa) for qa, qb in way for u in np.linspace(0, 1, 25)]
                link_low = float(min(lowest(q) for q in list(Q[::5]) + ramp)); ramp_tcp = float(min(pk.fk(q)[2, 3] for q in ramp))
                lim = lim or bool(np.any(np.array([up0, up1]) < lo) or np.any(np.array([up0, up1]) > hi))
                print(f"      절대: 시작점 {np.round(stroke_path.p0 * 1000).astype(int)} mm (READY 에서 {np.linalg.norm(off) * 1000:.0f} mm), 바닥 아래라 올린 점 {n_clip}개, "
                      f"링크 최저 {link_low * 1000:.0f} mm, 접근/복귀 경로 TCP 최저 {ramp_tcp * 1000:.0f} mm (READY 높이 경유)")
                link_low = min(link_low, ramp_tcp)
            min_ok = (zmin >= a.table_z + a.tcp_margin - 1e-6 and link_low >= a.table_z + a.link_margin) if a.absolute else zmin >= a.min_z
            ok = err < (0.01 if a.limit_ik else 0.003) and worst < (a.max_rot_dev if a.limit_ik else 1.5) and not lim and min_ok and vel <= a.max_deg_s
            d = P[np.argmax(np.linalg.norm(P, axis=1))]; rmax = f", 최대 회전 {np.degrees(np.linalg.norm(RV, axis=1)).max():.0f}°" if RV is not None else ""
            print(f"  {j:2d} {sg['hand']} {sg['axis']} {sg['dir']:5s} {sg['rep']}회: {len(P) / a.hz:4.1f}s, 최대 변위 {np.round(d * 100, 1)} cm{rmax}, IK 최대오차 {err * 1e3:.1f} mm/{worst:.1f}°, "
                  f"관절속도 최대 {vel:.0f}°/s, 최저 z {zmin:.3f}{'  관절한계 밖' if lim else ''} {note}{clamp_note} {'OK' if ok else '<- 불가'}")
            if not ok: bad.append((j, "IK/한계/높이/속도")); continue
            G = None
            if J is not None:
                from visual_teleop.tools.umi_jaw import grip_leads, sample_grip
                lo_ms, lc_ms = grip_leads(a, ROOT / "configs/robot/rebot_mit_deploy_v001.json"); a.grip_lead = [lo_ms, lc_ms]
                G = sample_grip(J[0], J[1], src, int(lo_ms * 1e6 * a.speed), int(lc_ms * 1e6 * a.speed))     # real robot ms -> source time at this replay speed
                nev = int(np.sum(np.diff(G > 0.5) != 0)); print(f"      집게({a.grip_mode}): 시작 {'열림' if G[0] > 0.5 else '닫힘'}, 바뀜 {nev}회, 열림 {lo_ms:.0f} / 닫힘 {lc_ms:.0f} ms 먼저 명령")
            plans.append(dict(seg=sg, src=src, P=P, RV=RV, Q=Q, Qs=Qs, off=off, G=G, up=(up0, up1) if a.absolute else None))
        print(f"구간 {len(plans)}/{len(segs)}개 실행 가능, 총 {sum(len(x['P']) for x in plans) / a.hz / 60:.1f}분 ({a.speed}배속)")
        if bad: print("제외:", bad)
        if not a.go: print("계획만 출력함 (--go 로 실제 이동)"); return
        out = ROOT / "datasets/replay" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(dict(take=str(a.take), hand=a.hand, side=s, speed=a.speed, hz=a.hz, smooth=a.smooth, T_ready=T_r.tolist(), absolute=a.absolute, abs_axes=a.abs_axes, abs_orient=a.abs_orient, table_z=a.table_z, orient=a.orient, gripper=a.gripper, grip_lead_ms=getattr(a, "grip_lead", a.grip_lead_ms), grip_mode=a.grip_mode if a.gripper else None, grip_only=a.grip_only, traj=a.traj,
                                                       segments=[x["seg"] for x in plans]), ensure_ascii=False, indent=1))
        say("유미 동작 재생을 시작합니다"); unpark(br, s); ob = br.observe(); g = {x: ob.g(x) for x in ("left", "right")}; other = "left" if s == "right" else "right"
        for j, x in enumerate(plans):
            g0 = dict(g) if x["G"] is None else {s: float(x["G"][0]), other: g[other]}
            if x["up"] is not None: ramp_to(br, s, x["up"][0], 10.0, g=g0)          # absolute: over the start at READY height, then down
            ramp_to(br, s, x["Qs"][0], 10.0, g=g0); time.sleep(0.5 if x["G"] is None else 1.5); say(f"{x['seg']['dir']} {x['seg']['rep']}회")
            q_oth = br.observe().q[other]; tc, to, qo, go = [], [], [], []; t0 = time.monotonic()
            for k, q in enumerate(x["Qs"]):
                gk = g0 if x["G"] is None else {s: float(x["G"][k]), other: g[other]}
                r = br.send({s: q, other: q_oth}, gk); tc.append(time.monotonic_ns())
                if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
                if k % a.obs_every == 0:
                    o = br.observe()
                    if o.estop: br.estop(); sys.exit("E-STOP/읽기 지연 -- 정지")
                    to.append(o.t_ns if getattr(o, "t_ns", None) else time.monotonic_ns()); qo.append(o.q[s].copy()); go.append(o.g(s))
                time.sleep(max(0.0, t0 + (k + 1) / a.hz - time.monotonic()))
            for _ in range(int(a.hz)):                       # 1 s tail: executed motion settles
                o = br.observe(); to.append(o.t_ns if getattr(o, "t_ns", None) else time.monotonic_ns()); qo.append(o.q[s].copy()); go.append(o.g(s)); time.sleep(1 / a.hz)
            qo = np.array(qo)
            np.savez(out / f"seg_{j:02d}.npz", src_t_ns=x["src"], t_cmd_ns=np.array(tc), q_cmd=x["Q"], p_cmd=np.array([pk.fk(q)[:3, 3] for q in x["Q"]]) - T_r[:3, 3] - x["off"], start_offset=x["off"], q_send=x["Qs"], lag_ms=np.array(a.lag_ms or [0] * 6),
                     t_obs_ns=np.array(to), q_obs=qo, p_obs=np.array([pk.fk(q)[:3, 3] for q in qo]) - T_r[:3, 3] - x["off"],
                     rot_cmd=np.array([Rot.from_matrix(pk.fk(q)[:3, :3] @ T_r[:3, :3].T).as_rotvec() for q in x["Q"]]),      # spatial dR vs READY
                     rot_obs=np.array([Rot.from_matrix(pk.fk(q)[:3, :3] @ T_r[:3, :3].T).as_rotvec() for q in qo]),
                     g_cmd=(x["G"] if x["G"] is not None else np.full(len(x["Q"]), g0[s])), g_obs=np.array(go))       # 0 closed .. 1 open
            if x["up"] is not None: ramp_to(br, s, x["up"][1], 10.0, g=g0)          # absolute: lift to READY height first
            ramp_to(br, s, qr, 10.0, g=g0)
        say("재생 끝났습니다"); print(f"저장 {out}\n다음: .venv/bin/python scripts/evaluation/replay_parity.py {a.take} {out}")
    finally:
        if br is not None: br.close()


if __name__ == "__main__":
    main()
