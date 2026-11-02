"""`start`: plan (dry) or drive (--go) READY -> a training episode's start pose, every waypoint checked.
"""
from __future__ import annotations
import json
import time

import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.paths import say, START_FILE


def start_cmd(a):
    """READY -> training start pose.  Target = the start TCPs (pose_abs at frame 0) of ONE training episode (default: the medoid
    episode, so both arms and T_LR come from the same real start), + --start-lift in z.  Path: Cartesian line READY -> target + 10 cm
    (IK seeded step to step), then straight down; every waypoint checked (IK error, joint limits, capsule arm-arm, link-table, TCP
    above table).  Dry-run prints the plan and writes deploy/start_pose.json; --go streams it at --deg-s (umi_replay_bi.stream_path)
    and then simply stops streaming (the bridge holds the last goal)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from visual_teleop.tools.robot_moves import READY_Q_DEG, unpark
    from visual_teleop import config as vcfg
    from visual_teleop.robot.bridge import HttpBridge
    if getattr(a, "label", ""):                    # 10-27 label replay: the label's own first pose (A' vs Cg test)
        lr = core.LabelRunner(a.label); e = a.label; i0 = 0; P0 = {s: lr.T_at(s, 0.0) for s in core.SIDES}
    else:
        root = core.LDT / "data/lerobot" / (getattr(a, "root", "") or "tagrig_laundry_v1_fused_train"); hf = LeRobotDataset("local/x", root=root, video_backend="pyav").hf_dataset.with_format("numpy")   # 10-28 --root (GOLD-281)
        P = np.asarray(hf["pose_abs"]).astype(float); fi = np.asarray(hf["frame_index"]); ep = np.asarray(hf["episode_index"]); st = np.flatnonzero(fi == 0)
        med = np.median(P[st][:, [0, 1, 2, 9, 10, 11]], 0)
        e = int(ep[st[np.argmin(np.linalg.norm(P[st][:, [0, 1, 2, 9, 10, 11]] - med, axis=1))]]) if a.episode < 0 else a.episode
        i0 = int(np.flatnonzero((ep == e) & (fi == 0))[0]); P0 = {s: core.pose9_to_T(P[i0, core.SL[s]]) for s in core.SIDES}
    arms, ra = core.make_arms(); col = core.Collision(arms)
    qr = {s: np.radians(READY_Q_DEG[s]) for s in core.SIDES}
    if a.go:
        c = vcfg.load(); br = HttpBridge(c.robot["base_url"], live=True, timeout_s=0.5); o = br.observe()
        if o.estop: raise SystemExit("로봇 E-STOP/미연결/읽기 지연")
        if max(np.degrees(np.abs(o.q[s] - qr[s])).max() for s in core.SIDES) > 5: raise SystemExit("두 팔이 READY 가 아님")
    tgt = {}
    hc = None
    if getattr(a, "height_calib", False):          # 11-01 user "xy 도 맞추어줘": land the start TCP where the label frame has it (robot FK = B - shift)
        from garment_pipeline.eval_deploy import height_calib as HC; hc = HC.load()
    for s in core.SIDES:
        T = P0[s].copy(); T[2, 3] += a.start_lift
        if hc is not None and "xy_B_minus_robot" in hc["fit"].get(s, {}): T[:2, 3] -= np.asarray(hc["fit"][s]["xy_B_minus_robot"])
        tgt[s] = T
    if hc is not None: say("시작 자세 xy 보정: " + ", ".join(f"{s} {np.round(np.asarray(hc['fit'][s].get('xy_B_minus_robot', [0, 0])) * -1e3, 1).tolist()} mm" for s in core.SIDES))
    def leg(qa, Ta, Tb, n=40):
        from scipy.spatial.transform import Rotation as Rot, Slerp
        sl = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]])); out, q = [], qa.copy()
        for u in np.linspace(0, 1, n):
            T = np.eye(4); T[:3, :3] = sl(u).as_matrix(); T[:3, 3] = (1 - u) * Ta[:3, 3] + u * Tb[:3, 3]; q, pe, re = arms[s_].free.ik(q, T); out.append((q.copy(), pe, re))
        return out
    path = {}; rep = {}
    for s_ in core.SIDES:
        Tr = core.fk(arms, s_, qr[s_]); up = tgt[s_].copy(); up[2, 3] += 0.10
        L1 = leg(qr[s_], Tr, up); L2 = leg(L1[-1][0], up, tgt[s_], 20); path[s_] = L1 + L2
        rep[s_] = dict(target_p=np.round(tgt[s_][:3, 3], 3).tolist(), ik_max_mm=round(max(x[1] for x in path[s_]) * 1e3, 1), rot_max_deg=round(max(x[2] for x in path[s_]), 1),
                       limits_ok=bool(all(np.all(x[0] >= arms[s_].lo) and np.all(x[0] <= arms[s_].hi) for x in path[s_])),
                       q_start_deg=np.round(np.degrees(path[s_][-1][0]), 1).tolist())
    n = max(len(v) for v in path.values()); W = [{s: path[s][min(i, len(path[s]) - 1)][0] for s in core.SIDES} for i in range(n)]
    chk = [col.check(w, table_z=ra.table_z) for w in W]; tcpz = min(core.fk(arms, s, w[s])[2, 3] for w in W for s in core.SIDES)
    jdist = max(float(np.degrees(np.abs(W[i + 1][s] - W[i][s])).max()) for i in range(n - 1) for s in core.SIDES)
    ok = all(r["ik_max_mm"] <= 10 and r["limits_ok"] for r in rep.values()) and min(c_["arm_clear_m"] for c_ in chk) >= 0.01 and min(c_["table_clear_m"] for c_ in chk) >= 0 and tcpz >= ra.table_z + 0.005
    lengths = sum(max(float(np.degrees(np.abs(W[i + 1][s] - W[i][s])).max()) for s in core.SIDES) for i in range(n - 1))
    out = dict(episode=e, frame0_index=i0, start_lift_m=a.start_lift, arms=rep, min_arm_clear_mm=round(min(c_["arm_clear_m"] for c_ in chk) * 1e3), min_table_clear_mm=round(min(c_["table_clear_m"] for c_ in chk) * 1e3),
               min_tcp_z_mm=round(tcpz * 1e3), max_joint_step_deg=round(jdist, 1), duration_s=round(lengths / a.deg_s, 1), ok=bool(ok),
               q={s: path[s][-1][0].tolist() for s in core.SIDES}, T={s: tgt[s].tolist() for s in core.SIDES}, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("q", "T")}, indent=1, ensure_ascii=False))
    if not ok: raise SystemExit("시작 자세 경로 불가 -- 이동 안 함")
    START_FILE.write_text(json.dumps(out, indent=1)); print("저장", START_FILE)
    if not a.go: print("DRY: 이동 안 함. 움직이려면 사람이 옆에서: start --go"); return
    from visual_teleop.tools.umi_replay_bi import stream_path
    for s in core.SIDES: unpark(br, s)
    # [11-02 user: "또 입이 벌려져 있어"] the observed jaw was re-sent here, and a parked (torque-off) jaw relaxes open -> close both
    # jaws on the way to the start (training episodes start closed; g 0 = closed)
    g = {s: 0.0 for s in core.SIDES}; say("학습 시작 자세로 천천히 이동 (집게 닫음)"); stream_path(br, [{s: br.observe().q[s] for s in core.SIDES}] + W, g, deg_s=a.deg_s)
    say("시작 자세 도착 (브리지가 마지막 목표를 유지)"); br.close()
