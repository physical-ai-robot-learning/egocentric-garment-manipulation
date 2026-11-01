"""UMI TCP trajectory from a tag_rig_record take (2026-10-20, replay parity A = "UMI measured"). Per colour frame with >= 1 tag of that UMI:
T_C_rig (src/rebot_prep/parity/tag_rig_fuse.py, rotation imu|tag, optional depth range) -> TCP point p_C = T_C_rig * tcp_in_rig_m (touch-TCP calibration,
configs/calibration/umi_tag_rig_v002.json), then into the robot base with T_B_C (configs/calibration/robot_cam_v001.json) when it exists.
For --proto axes takes it also writes the stroke segments (hand, axis, direction, repeat, outward / return / hold windows in host ns).
Robot base convention used for the expected directions: +X 앞, +Y 왼쪽, +Z 위 (reBot URDF base, right READY TCP ~ (0.45, -0.30, 0.28)).
    .venv/bin/python scripts/pose/umi_tcp_traj.py datasets/tag_rig/<take> --hand 오른손 [--rot imu|tag] [--depth] [--robot-cal <json>]
-> <take>/umi_traj_<right|left>.npz (t_ns, p_C, R_C_tcp, ntags, single_new_tag, guard_rot_deg[, p_B, R_B_tcp]) + <take>/segments.json   (unwritable take -> <take>_derived/)
10-23 v002: single_new_tag=True marks frames whose pose came ONLY from the lone 30 mm camera-back tag (220/221) after the IMU flip guard
(tag_rig_fuse step 4); guard_rot_deg = its rotation residual vs the gyro prediction (NaN elsewhere). Down-weight these downstream."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rebot_prep.parity.tag_rig_fuse import fuse_take, SIDE      # noqa: E402
DIRS = {"앞으로": ("X", +1), "뒤로": ("X", -1), "왼쪽으로": ("Y", +1), "오른쪽으로": ("Y", -1), "위로": ("Z", +1), "아래로": ("Z", -1),
        # 10-20 --proto rot: rotation about the robot-BASE axis through the jaw tip (right-hand rule; READY approach axis = base +X):
        # yaw = about +Z (turn left = +), pitch = about +Y (tip down = +), roll = about +X (clockwise seen from behind = +)
        "왼쪽돌리기": ("YAW", +1), "오른쪽돌리기": ("YAW", -1), "끝아래로": ("PITCH", +1), "끝위로": ("PITCH", -1),
        "시계방향비틀기": ("ROLL", +1), "반시계방향비틀기": ("ROLL", -1)}


def segments(meta: dict) -> list[dict]:
    """axes protocol -> strokes. sched entries: hold '<hand>, <ax>축, <r>회, 멈추세요' -> move '<hand> <dir> <n>센티 천천히' -> hold -> return"""
    t0 = int(meta["t_start_monotonic_ns"]); S = meta["protocol"]; out = []
    ns = lambda e, a=0.0: int(t0 + (e["t0"] + a * e["dur"]) * 1e9)
    for i, e in enumerate(S):
        if ", 자유동작, " in e["text"]:      # 10-20 --proto free: hold -> consecutive '자유:' cues -> hold (one segment per trial)
            j = i + 1
            while j < len(S) and S[j]["text"].startswith("자유:"): j += 1
            if j == i + 1 or j >= len(S): continue
            hp = [ns(S[i]), ns(S[i], 1)]; he = [ns(S[j]), ns(S[j], 1)]
            out.append(dict(hand=e["text"].split(",")[0], axis="FREE", sign=1, dir="자유", phys="free", rep=int(e["text"].split(",")[2].strip().rstrip("회")),
                            cues=[x["text"][4:] for x in S[i + 1:j]], hold_pre=hp, out=[ns(S[i + 1]), ns(S[j - 1], 1)], hold_mid=he, back=None, hold_post=he))
            continue
        w = e["text"].split()
        if not ("센티 천천히" in e["text"] or "도 천천히" in e["text"]) or len(w) < 3 or w[1] not in DIRS: continue
        hold = S[i - 1]["text"].split(","); ax, sg = DIRS[w[1]]
        out.append(dict(hand=w[0], axis=ax, sign=sg, dir=w[1], phys={"YAW": "sweep_left_right(x_E)", "PITCH": "bend_up_down(y_E)", "ROLL": "twist_jaw_axis(z_E)"}.get(ax, "translation"), rep=int(hold[2].strip().rstrip("회")), cm=float(w[2].replace("센티", "")) if "센티" in w[2] else None, deg=float(w[2].replace("도", "")) if w[2].endswith("도") else None,
                        hold_pre=[ns(S[i - 1]), ns(S[i - 1], 1)], out=[ns(e), ns(e, 1)], hold_mid=[ns(S[i + 1]), ns(S[i + 1], 1)], back=[ns(S[i + 2]), ns(S[i + 2], 1)],
                        hold_post=[ns(S[i + 3]), ns(S[i + 3], 1)] if i + 3 < len(S) else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--hand", default="오른손", choices=list(SIDE)); ap.add_argument("--rot", default="imu", choices=["imu", "tag"])
    ap.add_argument("--depth", action="store_true"); ap.add_argument("--imu-fill", action="store_true", help="태그 끊긴 구간 회전을 IMU 로 채움 (위치는 보간, ntags=0 표시)"); ap.add_argument("--tag-size", type=float, default=0.048)
    ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v002.json")); ap.add_argument("--robot-cal", default=str(ROOT / "configs/calibration/robot_cam_v001.json"))
    a = ap.parse_args(); E = Path(a.take); meta = json.loads((E / "meta.json").read_text())
    rig = json.loads(Path(a.rig).read_text()); name, side = SIDE[a.hand]; tcp = np.array(rig[name]["tcp_in_rig_m"])
    F = fuse_take(E, a.hand, rig, a.tag_size, use_depth=a.depth, rot_source=a.rot, fill_gaps=a.imu_fill)
    if F is None: sys.exit(f"{a.hand} 유미 태그가 한 번도 안 보임")
    T = F["T_C_rig"]; p_C = T[:, :3, :3] @ tcp + T[:, :3, 3]
    res = dict(t_ns=F["t_ns"], p_C=p_C, R_C_tcp=T[:, :3, :3], ntags=F["ntags"], rot=a.rot, depth=a.depth, single_new_tag=F["single_new_tag"], guard_rot_deg=F["guard_rot_deg"])
    gs = F["guard_stats"]
    if gs["single_new_tag_frames"]:
        print(f"카메라 뒤 30mm 태그만 보인 프레임 {gs['single_new_tag_frames']}: IMU 뒤집힘 방지 통과 {gs['accepted']}, 거부 회전 {gs['rejected_rot']} / 점프 {gs['rejected_jump']} / "
              f"이전 자세 없음 {gs['rejected_no_prior']} / IMU 없음 {gs['rejected_no_imu']} (통과 프레임은 single_new_tag=True)")
    if a.imu_fill and "t_all_ns" in F:
        # every colour frame: IMU-anchored orientation; TCP position interpolated across tag gaps (marked ntags=0 -> not a measurement)
        ta = F["t_all_ns"]; pa = np.stack([np.interp(ta, F["t_ns"], p_C[:, k]) for k in range(3)], 1)
        na = np.zeros(len(ta), int); idx = np.searchsorted(ta, F["t_ns"]); na[idx] = F["ntags"]; ga = np.full(len(ta), np.nan); ga[idx] = F["guard_rot_deg"]
        sa = np.zeros(len(ta), bool); sa[idx] = F["single_new_tag"]
        res = dict(t_ns=ta, p_C=pa, R_C_tcp=F["R_C_rig_all"], ntags=na, rot=a.rot, depth=a.depth, imu_fill=True, single_new_tag=sa, guard_rot_deg=ga)
        print(f"IMU 로 채움: 프레임 {len(ta)} 중 태그 있음 {int((na > 0).sum())}, 회전만 IMU {int((na == 0).sum())}")
    if Path(a.robot_cal).exists():
        cal = json.loads(Path(a.robot_cal).read_text()); TBC = np.array(cal["T_B_C"])
        res.update(p_B=res["p_C"] @ TBC[:3, :3].T + TBC[:3, 3], R_B_tcp=TBC[:3, :3] @ res["R_C_tcp"], robot_cal=a.robot_cal)
        print(f"로봇 베이스 좌표로 변환 ({a.robot_cal})")
    else: print(f"[주의] 로봇 캘리브레이션 없음 ({a.robot_cal}) -- 카메라 좌표만 저장 (replay_parity 는 --cam-axes 로 근사 비교만 가능)")
    out = E if os.access(E, os.W_OK) else E.parent / (E.name + "_derived"); out.mkdir(exist_ok=True)
    np.savez(out / f"umi_traj_{side}.npz", **res)
    gaps = np.diff(F["t_ns"]) / 1e9; p_C = res["p_C"]
    print(f"{a.hand}: 프레임 {len(p_C)}/{meta['frames']} ({len(p_C) / max(meta['frames'], 1) * 100:.0f}%), 태그 2개 이상 {np.mean(F['ntags'] >= 2) * 100:.0f}%, "
          f"최대 끊김 {gaps.max() if len(gaps) else 0:.2f}s, 태그-융합 회전차 중앙 {np.median(F['tag_vs_fused_deg']):.1f}° -> {out / f'umi_traj_{side}.npz'}")
    if meta.get("proto") in ("axes", "rot", "free", "grip", "task"):
        seg = segments(meta); (out / "segments.json").write_text(json.dumps(seg, ensure_ascii=False, indent=1)); print(f"구간 {len(seg)}개 -> {out / 'segments.json'}")


if __name__ == "__main__":
    main()
