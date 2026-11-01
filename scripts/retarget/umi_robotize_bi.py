"""Stage-A bimanual robotization of a human UMI take (2026-10-21, GPT: human_canonical_tcp -> robot_feasible_tcp before IK).
113243 replayed as-is failed: the right hand crossed the body midline (y up to +0.19 m), hands came within 3-11 cm, both jaws were
pressed below the table. Per demo window (segments.json rep) and per hand, the POSITION is warped continuously (never clipped):
  y   right hand: affine compression toward its outer edge so its p99 stays at <= -sep/2; left hand mirrored (p1 >= +sep/2);
      then a smooth soft-limit for whatever still crosses (no hard clamp)
  z   smooth soft floor at table + clearance (softplus, width --soft-w), so contact segments are lifted without a kink
Orientation is left to the IK (umi_replay_bi --ori-cost relaxes it). Writes <take>/umi_traj_<side>_robot.npz (same keys, p_B replaced,
p_B_human kept) and prints how much was changed. Feasibility (IK / link-link gap / table) is then checked by
    umi_replay_bi <take> --traj _robot --rep N --offline
    .venv/bin/python scripts/retarget/umi_robotize_bi.py datasets/tag_rig/<take> [--sep 0.16] [--table-z 0.003 --clear 0.035]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def soft_floor(z, f, w):
    return f + w * np.logaddexp(0.0, (z - f) / w)


def soft_ceil(y, c, w):
    return -soft_floor(-y, -c, w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--sep", type=float, default=0.16, help="두 손 y 사이 최소 간격 (m), 중앙선 = 0")
    ap.add_argument("--table-z", type=float, default=0.003); ap.add_argument("--clear", type=float, default=0.035, help="TCP 의 테이블 위 최소 높이 (m)")
    ap.add_argument("--soft-w", type=float, default=0.01); ap.add_argument("--min-reach", type=float, default=0.30, help="베이스에서 수평 최소 거리 (m), 0 = 끔")
    ap.add_argument("--base-y", type=float, default=0.315, help="팔 베이스 y (좌 +, 우 -)"); ap.add_argument("--src", default="", help="입력 궤적 접미사 (예: _fused) -> 출력 umi_traj_<손><src>_robot.npz"); ap.add_argument("--min-scale", type=float, default=0.4); a = ap.parse_args()
    E = Path(a.take); segs = json.loads((E / "segments.json").read_text()); reps = sorted({g["rep"] for g in segs})
    Z = {s: dict(np.load(E / f"umi_traj_{s}{a.src}.npz", allow_pickle=True)) for s in ("right", "left")}
    for s, z in Z.items():
        t = z["t_ns"].astype(np.int64); p = z["p_B"].astype(float).copy(); ph = p.copy()
        for r in reps:
            g = [x for x in segs if x["rep"] == r][0]; t0 = g["hold_pre"][0]; t1 = (g["hold_post"] or g["back"])[1]; m = (t >= t0) & (t <= t1)
            if m.sum() < 10: continue
            y = p[m, 1]; half = a.sep / 2
            if s == "right":
                yo, yi = np.percentile(y, 1), np.percentile(y, 99)
                sc = 1.0 if yi <= -half else float(np.clip((-half - yo) / max(yi - yo, 1e-6), a.min_scale, 1.0)); y2 = soft_ceil(yo + sc * (y - yo), -half, a.soft_w)
            else:
                yo, yi = np.percentile(y, 99), np.percentile(y, 1)
                sc = 1.0 if yi >= half else float(np.clip((yo - half) / max(yo - yi, 1e-6), a.min_scale, 1.0)); y2 = soft_floor(yo + sc * (y - yo), half, a.soft_w)
            p[m, 1] = y2; p[m, 2] = soft_floor(p[m, 2], a.table_z + a.clear, a.soft_w)
            if a.min_reach > 0:     # 10-21: left IK failed on targets 18-39 cm from its own base, low (x 0.14-0.35, z 0.04-0.19) -> push radially out
                b0 = np.array([0.0, a.base_y if s == "left" else -a.base_y]); v = p[m, :2] - b0; r_ = np.linalg.norm(v, axis=1)
                r2 = soft_floor(r_, a.min_reach, a.soft_w * 2); p[m, :2] = b0 + v * (r2 / np.maximum(r_, 1e-6))[:, None]
            d = np.linalg.norm(p[m] - ph[m], axis=1)
            print(f"{s} {r}회: y 범위 사람 [{np.percentile(y, 1):+.3f}, {np.percentile(y, 99):+.3f}] -> 로봇 [{np.percentile(p[m, 1], 1):+.3f}, {np.percentile(p[m, 1], 99):+.3f}] (압축 {sc:.2f}), "
                  f"z 최저 {ph[m, 2].min():+.3f} -> {p[m, 2].min():+.3f} | 위치 변경 중앙 {np.median(d) * 1000:.0f} mm p95 {np.percentile(d, 95) * 1000:.0f} mm, 5 mm 넘게 바뀐 프레임 {np.mean(d > 0.005) * 100:.0f}%")
        # 10-21 (GPT, training on robot_feasible_tcp): per-frame quality so a dataset builder can weight / drop frames whose action label
        # moved far from what the wrist video shows. class 0 < 20 mm (train as is), 1 < 50 mm (lower weight), 2 >= 50 mm (replace / drop)
        disp = np.linalg.norm(p - ph, axis=1); lift = p[:, 2] - ph[:, 2]
        z.update(p_B_human=ph, p_B=p, robotize=json.dumps(vars(a)), robotize_disp_m=disp, robotize_lift_z_m=lift, robotize_quality=np.digitize(disp, [0.02, 0.05]).astype(np.int8))
        np.savez(E / f"umi_traj_{s}{a.src}_robot.npz", **z)
        q_ = z["robotize_quality"]; print(f"{s} 프레임 품질: 작은 수정(<20 mm) {np.mean(q_ == 0) * 100:.0f}%, 중간(<50) {np.mean(q_ == 1) * 100:.0f}%, 큰 수정(>=50) {np.mean(q_ == 2) * 100:.0f}%")
    print(f"저장 {E}/umi_traj_{{right,left}}_robot.npz")


if __name__ == "__main__":
    main()
