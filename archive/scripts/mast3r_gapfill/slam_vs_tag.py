"""Wrist SLAM trajectory vs the Orbbec/AprilTag reference (2026-10-20, GPT plan step 2-4: "how well does the UMI TCP trajectory survive
when the tags are hidden"). Inputs: a tag_rig_record take (umi_traj_right.npz from umi_tcp_traj --imu-fill: p_B = TCP in the robot base,
R_B_tcp = rig orientation, ntags = 0 on IMU-filled frames), the MASt3R-SLAM per-frame CSV (T_W_cam, camera -> SLAM world, Sim3 world
units) and frames.json (SLAM frame index -> wrist host monotonic_ns).
Alignment (only TAG frames are truth): T_B_T(t) = A * S_s(T_W_cam(t + dt)) * X
  dt   wrist-vs-Orbbec time offset: angular-speed correlation of SLAM vs the IMU-fused orientation, +-300 ms
  X    camera -> TCP frame rotation: hand-eye on 0.4 s relative rotations;  A rotation = mean of R_B R_X^T R_c^T
  s, t_X, t_A   linear least squares on the TCP positions, then Huber refinement of everything
KPIs (GPT): ATE on tag frames (global fit) and on a held-out second half (fit on the first half); "tags hidden for d s" drift = error of
the SLAM displacement over d = 0.5/1/2 s against the tag displacement; orientation drift over d; 10/20 cm segment length error; SLAM
coverage (valid / lost / reloc); optional jaw events (--jaw umi_jaw.npz): TCP height around each grasp, SLAM vs the tag/IMU-filled path.
    .venv/bin/python scripts/slam_vs_tag.py datasets/tag_rig/<take> <m3.csv> <frames.json> [--jaw <dir>/umi_jaw.npz] [--save]"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def load_slam(csv_path, frames_json):
    fr = {f["i"]: f["t_ns"] for f in json.loads(Path(frames_json).read_text())["frames"]}
    T, P, Q, st = [], [], [], []
    for r in csv.DictReader(open(csv_path)):
        i = int(r["frame_idx"])
        if i not in fr: continue
        T.append(fr[i]); P.append([float(r["x"]), float(r["y"]), float(r["z"])]); Q.append([float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")])
        st.append((r["state"], r["is_lost"] == "true"))
    Q = np.array(Q); zq = np.linalg.norm(Q, axis=1) < 1e-6; Q[zq] = [0, 0, 0, 1]          # invalid frames carry a zero quaternion
    ok = np.array([s in ("INIT", "TRACKING") and not lost for s, lost in st]) & ~zq
    return np.array(T, np.int64), np.array(P), R.from_quat(Q), ok, st


def interp_rot(t_src, Rs, t):
    from scipy.spatial.transform import Slerp
    t = np.clip(t, t_src[0], t_src[-1]); return Slerp(t_src, Rs)(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("csv"); ap.add_argument("frames"); ap.add_argument("--jaw"); ap.add_argument("--save", action="store_true")
    ap.add_argument("--hand", default="right"); a = ap.parse_args()
    z = np.load(Path(a.take) / f"umi_traj_{a.hand}.npz"); tT = z["t_ns"].astype(np.int64); pB = z["p_B"]; RB = R.from_matrix(z["R_B_tcp"]); nt = z["ntags"]
    tS, pS, RS, ok, st = load_slam(a.csv, a.frames)
    n_lost = sum(l for _, l in st); n_reloc = sum(s == "RELOC" for s, _ in st)
    print(f"SLAM: 프레임 {len(tS)}, 유효 {ok.mean() * 100:.0f}% (잃음 {n_lost}, 재위치 {n_reloc}) | 태그 기준 프레임 {int((nt > 0).sum())}/{len(tT)}")
    tS, pS, RS = tS[ok], pS[ok], RS[np.where(ok)[0]]
    # 1) time offset: |w| of SLAM vs |w| of the IMU-fused reference (dense) on a 10 ms grid
    g = np.arange(max(tS[0], tT[0]) + int(0.5e9), min(tS[-1], tT[-1]) - int(0.5e9), int(1e7))
    def speed(t_src, Rs, tt):
        Ri = interp_rot((t_src - t_src[0]) / 1e9, Rs, (tt - t_src[0]) / 1e9); return np.r_[np.linalg.norm((Ri[:-1].inv() * Ri[1:]).as_rotvec(), axis=1) / 0.01, 0]
    wT = speed(tT, RB, g); best = max(np.arange(-0.3e9, 0.3e9, 1e7), key=lambda d: np.corrcoef(wT, speed(tS, RS, g + d))[0, 1]); dt = int(best)
    print(f"시간 오프셋 (손목 - Orbbec): {dt / 1e6:+.0f} ms, 회전속도 상관 {np.corrcoef(wT, speed(tS, RS, g + dt))[0, 1]:.2f}")
    # 2) truth frames = TAG frames inside the SLAM span
    m = (nt > 0) & (tT + dt >= tS[0]) & (tT + dt <= tS[-1]); tt = tT[m]; pb = pB[m]; Rb = RB[np.where(m)[0]]
    Rc = interp_rot((tS - tS[0]) / 1e9, RS, (tt + dt - tS[0]) / 1e9); pc = np.stack([np.interp(tt + dt, tS, pS[:, k]) for k in range(3)], 1)
    def fit(idx):
        pairs = [(i, j) for i in idx for j in [idx[min(np.searchsorted(tt[idx], tt[i] + 4e8), len(idx) - 1)]] if 0.3e9 < tt[j] - tt[i] < 0.6e9]
        A_ = np.array([(Rc[i].inv() * Rc[j]).as_rotvec() for i, j in pairs]); B_ = np.array([(Rb[i].inv() * Rb[j]).as_rotvec() for i, j in pairs])
        big = np.linalg.norm(A_, axis=1) > np.radians(5); A_, B_ = A_[big], B_[big]                  # rv_T = R_X^T rv_c
        U, _, Vt = np.linalg.svd(B_.T @ A_); RXt = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt; RX = R.from_matrix(RXt.T)
        RA = R.from_matrix(np.array([(Rb[i] * RX.inv() * Rc[i].inv()).as_matrix() for i in idx])).mean()
        M = np.vstack([np.c_[RA.apply(pc[i])[:, None], (RA * Rc[i]).as_matrix(), np.eye(3)] for i in idx]); y = np.concatenate([pb[i] for i in idx])
        sol = np.linalg.lstsq(M, y, rcond=None)[0]; s, tX, tA = sol[0], sol[1:4], sol[4:7]
        def res(x):
            RA_, RX_ = R.from_rotvec(x[0:3]), R.from_rotvec(x[3:6]); s_, tX_, tA_ = np.exp(x[6]), x[7:10], x[10:13]
            pp = s_ * RA_.apply(pc[idx]) + (RA_ * Rc[idx]).apply(tX_) + tA_
            rr = ((RA_ * Rc[idx] * RX_).inv() * Rb[idx]).as_rotvec() * 0.05                                    # 1 rad ~ 50 mm
            return np.concatenate([(pp - pb[idx]).ravel(), rr.ravel()])
        x0 = np.r_[RA.as_rotvec(), RX.as_rotvec(), np.log(abs(s)), tX, tA]; x = least_squares(res, x0, loss="huber", f_scale=0.01).x
        return R.from_rotvec(x[0:3]), R.from_rotvec(x[3:6]), np.exp(x[6]), x[7:10], x[10:13]
    def pred(RA, RX, s, tX, tA, i):
        return s * RA.apply(pc[i]) + (RA * Rc[i]).apply(tX) + tA, RA * Rc[i] * RX
    allidx = np.arange(len(tt)); RA, RX, s, tX, tA = fit(allidx); pp, Rp = pred(RA, RX, s, tX, tA, allidx)
    e = np.linalg.norm(pp - pb, axis=1) * 1e3; er = np.degrees((Rp.inv() * Rb).magnitude())
    print(f"정렬: 스케일 {s:.4f} (SLAM 1 단위 = {s * 1000:.0f} mm), 카메라->TCP {np.round(tX * 1000).astype(int)} mm, 회전 {np.degrees(RX.magnitude()):.0f}°")
    print(f"[전체 맞춤] 태그 프레임 {len(tt)}개: 위치 오차 중앙 {np.median(e):.1f} mm, p95 {np.percentile(e, 95):.1f} | 자세 오차 중앙 {np.median(er):.1f}°, p95 {np.percentile(er, 95):.1f}")
    half = tt < tt[0] + (tt[-1] - tt[0]) / 2; F = fit(np.where(half)[0]); pq, Rq = pred(*F, np.where(~half)[0])
    e2 = np.linalg.norm(pq - pb[~half], axis=1) * 1e3; er2 = np.degrees((Rq.inv() * Rb[np.where(~half)[0]]).magnitude())
    print(f"[앞 절반으로 맞추고 뒤 절반 검증] 위치 중앙 {np.median(e2):.1f} mm p95 {np.percentile(e2, 95):.1f} | 자세 중앙 {np.median(er2):.1f}° | 스케일 {F[2]:.4f} (전체 {s:.4f})")
    print("\n태그가 d초 동안 안 보일 때 (SLAM 변위 vs 태그 변위, 시작점 = 태그 프레임) | 기준: <10 mm 매우 좋음, 10-20 usable, >20-30 개선 필요")
    for d in (0.5, 1.0, 2.0):
        j = np.searchsorted(tt, tt + d * 1e9); v = (j < len(tt)); v[v] = np.abs(tt[j[v]] - tt[v] - d * 1e9) < 0.08e9
        i0, i1 = np.where(v)[0], j[v]
        de = np.linalg.norm((pp[i1] - pp[i0]) - (pb[i1] - pb[i0]), axis=1) * 1e3; mv = np.linalg.norm(pb[i1] - pb[i0], axis=1) * 1e3
        dr = np.degrees(((Rp[i0].inv() * Rp[i1]).inv() * (Rb[i0].inv() * Rb[i1])).magnitude())
        print(f"  {d:.1f}초: 위치 오차 중앙 {np.median(de):5.1f} mm, p95 {np.percentile(de, 95):5.1f} (그동안 실제 이동 중앙 {np.median(mv):.0f} mm) | 자세 오차 중앙 {np.median(dr):.1f}°, p95 {np.percentile(dr, 95):.1f} (n={len(i0)})")
    for L in (0.10, 0.20):
        rows = []
        for i in range(0, len(tt), 3):
            k = np.where((tt > tt[i]) & (tt < tt[i] + 5e9))[0]; dd = np.linalg.norm(pb[k] - pb[i], axis=1); c = k[np.abs(dd - L) < 0.01]
            rows += [(np.linalg.norm(pp[x] - pp[i]) - np.linalg.norm(pb[x] - pb[i])) * 1e3 for x in c[:3]]
        if rows: print(f"  {int(L * 100)} cm 구간 길이 오차: 중앙 {np.median(np.abs(rows)):.1f} mm ({np.median(np.abs(rows)) / L / 10:.1f}%), p95 {np.percentile(np.abs(rows), 95):.1f} mm (n={len(rows)})")
    allt = tT[(tT + dt >= tS[0]) & (tT + dt <= tS[-1])]
    Rc_all = interp_rot((tS - tS[0]) / 1e9, RS, (allt + dt - tS[0]) / 1e9); pc_all = np.stack([np.interp(allt + dt, tS, pS[:, k]) for k in range(3)], 1)
    p_all = s * RA.apply(pc_all) + (RA * Rc_all).apply(tX) + tA; zf = np.interp(allt, tT, pB[:, 2]); vis = np.interp(allt, tT, (nt > 0).astype(float)) > 0.5
    j_ = np.clip(np.searchsorted(tS, allt + dt), 1, len(tS) - 1); slam_ok = np.minimum(np.abs(tS[j_] - allt - dt), np.abs(tS[j_ - 1] - allt - dt)) < 0.15e9
    if a.save:      # SLAM-aligned TCP on every Orbbec frame time; slam_valid = False where SLAM was lost (interpolated across)
        out = Path(a.take) / f"umi_traj_{a.hand}_slam.npz"
        np.savez(out, t_ns=allt, p_B=p_all, R_B_tcp=(RA * Rc_all * RX).as_matrix(), ntags=np.interp(allt, tT, nt).round().astype(int), slam_valid=slam_ok, scale=s, dt_ns=dt)
        print(f"저장 {out} (SLAM 정렬 TCP 궤적, SLAM 유효 {slam_ok.mean() * 100:.0f}%)")
    if a.jaw:   # TCP height around each jaw event: SLAM vs the tag/IMU-filled path (where the tags were hidden)
        jz = np.load(a.jaw)
        print("\n집게 사건 주변 높이 (사건 -1.5초 대비 최저 Z 변화): SLAM vs 태그(가려지면 보간)")
        for te, k in zip(jz["ev_t"], jz["ev_kind"]):
            w = (allt > te - 1.5e9) & (allt < te + 1.5e9)
            if w.sum() < 5: continue
            b0 = allt[w][0]; i0 = np.argmin(np.abs(allt - b0))
            print(f"  {'열림' if k == 'open' else '닫힘'} {(te - tT[0]) / 1e9:6.1f}s: 태그 보임 {vis[w].mean() * 100:3.0f}% | 최저 Z 변화 SLAM {(p_all[w, 2].min() - p_all[i0, 2]) * 1000:+5.0f} mm, 태그/보간 {(zf[w].min() - zf[i0]) * 1000:+5.0f} mm")


if __name__ == "__main__":
    main()
