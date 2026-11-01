"""Joint multi-segment hand-eye for restarted MASt3R segments (2026-10-21, GPT: one camera->TCP X for all segments, a Sim3 per segment).
Per-segment hand-eye (umi_fuse_segments step 1) gave X = (23,120,137) / (-23,87,147) mm depending on the segment: each segment's scale /
orientation / origin error was absorbed into X. Here
    T_B_TCP(tag) ~ S_B_S(segment) . T_S_C(slam) . X          X shared, S_B_S per segment, Huber loss
on the GOOD segments only (enough anchors, TCP rotation >= --min-rot deg, travel >= --min-travel m, position-only Sim3 residual < --max-pos-res).
Validation: leave one good segment out, solve X on the others, then fit only that segment's Sim3 (X fixed) on the FIRST half of its anchors
and measure TCP error on the SECOND half (and the same with X re-solved on that segment alone, for contrast).
    .venv/bin/python scripts/umi_joint_handeye.py datasets/tag_rig/<take> <segs/manifest.json> <frames.json> [--hand right] [--save]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam_vs_tag import interp_rot                                        # noqa: E402
from umi_fuse_segments import load_segments, sim3_resid, trim_diverged    # noqa: E402

W_ROT = 0.05          # m per rad, as umi_fuse_segments


def seg_obs(sg, tt, dt):
    """SLAM camera pose of segment sg at tag times tt (+dt)"""
    pc = np.stack([np.interp(tt + dt, sg["t"], sg["p"][:, k]) for k in range(3)], 1)
    Rc = interp_rot((sg["t"] - sg["t"][0]) / 1e9, sg["R"], (tt + dt - sg["t"][0]) / 1e9); return pc, Rc


def init_sim3(pc, pb):
    A = pc - pc.mean(0); B = pb - pb.mean(0); U, S, Vt = np.linalg.svd(B.T @ A / len(A)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ D @ Vt; s = (S * np.diag(D)).sum() / A.var(0).sum(); return R.from_matrix(Rm), s, pb.mean(0) - s * Rm @ pc.mean(0)


def tcp_pred(RA, s, t, pc, Rc, RX, tX):
    return s * RA.apply(pc) + (RA * Rc).apply(tX) + t, RA * Rc * RX


FIX_T = None      # 10-21: --fix-t (camera->TCP translation fixed, only its rotation + the segment Sim3s solved)


def solve(obs, x0X=None, fixX=None):
    """obs: list of (pc, Rc, pb, Rb). Returns RX, tX, [(RA, s, t)]. fixX=(RX, tX) solves only the per-segment Sim3s."""
    n = len(obs); inits = [init_sim3(pc, pb) for pc, _, pb, _ in obs]
    if fixX is None:
        if x0X is None:     # rotation-only hand-eye over all segments: RX = mean of (RA Rc)^-1 Rb with the Umeyama RA
            RX0 = R.from_matrix(np.concatenate([((ia[0] * o[1]).inv() * o[3]).as_matrix() for ia, o in zip(inits, obs)])).mean()
            x0X = np.r_[RX0.as_rotvec(), 0, 0, 0.12]
    def unpack(x):
        if fixX is None: RX, tX, k = R.from_rotvec(x[:3]), (x[3:6] if FIX_T is None else FIX_T), 6
        else: (RX, tX), k = fixX, 0
        return RX, tX, [(R.from_rotvec(x[k + 7 * i:k + 7 * i + 3]), np.exp(x[k + 7 * i + 3]), x[k + 7 * i + 4:k + 7 * i + 7]) for i in range(n)]
    def res(x):
        RX, tX, S = unpack(x); out = []
        for (RA, s, t), (pc, Rc, pb, Rb) in zip(S, obs):
            pp, Rp = tcp_pred(RA, s, t, pc, Rc, RX, tX); out += [(pp - pb).ravel(), (Rp.inv() * Rb).as_rotvec().ravel() * W_ROT]
        return np.concatenate(out)
    x0 = np.concatenate(([] if fixX is not None else [x0X]) + [np.r_[ia[0].as_rotvec(), np.log(ia[1]), ia[2]] for ia in inits])
    x = least_squares(res, x0, loss="huber", f_scale=0.01).x
    return unpack(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("manifest"); ap.add_argument("frames"); ap.add_argument("--hand", default="right"); ap.add_argument("--dt-ms", type=float, default=20.0)
    ap.add_argument("--min-anchors", type=int, default=30); ap.add_argument("--min-rot", type=float, default=25.0); ap.add_argument("--min-travel", type=float, default=0.15)
    ap.add_argument("--max-pos-res", type=float, default=0.015); ap.add_argument("--save", action="store_true")
    ap.add_argument("--fix-t", help="카메라->TCP 이동을 mm 로 고정 (예: -5,34,122)"); a = ap.parse_args()
    global FIX_T
    if a.fix_t: FIX_T = np.array([float(v) for v in a.fix_t.split(",")]) / 1000
    E = Path(a.take); z = np.load(E / f"umi_traj_{a.hand}.npz"); tT = z["t_ns"].astype(np.int64); nt = z["ntags"]; tag = nt > 0
    tt0, pb0, Rb0 = tT[tag], z["p_B"][tag], R.from_matrix(z["R_B_tcp"][tag]); dt = int(a.dt_ms * 1e6)
    segs = trim_diverged(load_segments(a.manifest, a.frames, Path(a.manifest).parent), tt0, pb0)
    good, info = [], []
    for sg in segs:
        idx = np.where((tt0 + dt >= sg["t"][0]) & (tt0 + dt <= sg["t"][-1]))[0]
        if len(idx) < a.min_anchors: continue
        pc, Rc = seg_obs(sg, tt0[idx], dt); rot = np.degrees(np.linalg.norm((Rb0[idx[0]].inv() * Rb0[idx]).as_rotvec(), axis=1)).max()
        trav = np.ptp(pb0[idx], 0).max(); r = float(np.median(sim3_resid(pc, pb0[idx])[0]))
        ok = rot >= a.min_rot and trav >= a.min_travel and r <= a.max_pos_res
        info.append(f"{sg['start']}: 앵커 {len(idx)} 회전 {rot:.0f}° 이동 {trav * 100:.0f}cm 위치Sim3 {r * 1000:.0f}mm{' *' if ok else ''}")
        if ok: good.append((sg["start"], (pc, Rc, pb0[idx], Rb0[idx])))
    print("[구간] " + " | ".join(info)); print(f"[구간] 손눈에 쓸 좋은 구간 {len(good)}개: {[g[0] for g in good]}  (* 표시)")
    if len(good) < 2: sys.exit("좋은 구간이 2개 미만 -- 공동 풀이 불가")
    def err(pp, Rp, pb, Rb): return np.linalg.norm(pp - pb, axis=1), np.degrees(np.linalg.norm((Rp.inv() * Rb).as_rotvec(), axis=1))
    RX, tX, S = solve([o for _, o in good])
    print(f"[공동 손눈] 카메라->TCP {np.round(tX * 1000).astype(int)} mm, 회전 {np.degrees(np.linalg.norm(RX.as_rotvec())):.0f}° (구간 {len(good)}개 공통)")
    for (st, o), (RA, s, t) in zip(good, S):
        e, eR = err(*tcp_pred(RA, s, t, o[0], o[1], RX, tX), o[2], o[3]); print(f"   학습 구간 {st}: TCP 오차 중앙 {np.median(e) * 1000:.1f} mm p95 {np.percentile(e, 95) * 1000:.1f}, 자세 {np.median(eR):.1f}°")
    print("[검증] 한 구간을 빼고 X 를 풀고, 그 구간은 앵커 앞 절반으로 Sim3 만 맞춰 뒤 절반에서 오차 (괄호 = 그 구간 혼자 X 까지 푼 경우)")
    allj, alls = [], []
    for k, (st, o) in enumerate(good):
        RXk, tXk, _ = solve([g[1] for j, g in enumerate(good) if j != k]); h = len(o[0]) // 2
        first = tuple(v[:h] for v in o); second = tuple(v[h:] for v in o)
        _, _, Sk = solve([first], fixX=(RXk, tXk)); e, eR = err(*tcp_pred(*Sk[0], second[0], second[1], RXk, tXk), second[2], second[3])
        RXs, tXs, Ss = solve([first]); e2, _ = err(*tcp_pred(*Ss[0], second[0], second[1], RXs, tXs), second[2], second[3])
        allj += list(e); alls += list(e2)
        print(f"   {st}: X(빼고 푼) {np.round(tXk * 1000).astype(int)} mm -> 뒤 절반 오차 중앙 {np.median(e) * 1000:.1f} mm p95 {np.percentile(e, 95) * 1000:.1f}, 자세 {np.median(eR):.1f}°   (혼자: X {np.round(tXs * 1000).astype(int)}, {np.median(e2) * 1000:.1f} mm)")
    print(f"[검증 전체] 공동 X: 중앙 {np.median(allj) * 1000:.1f} mm p95 {np.percentile(allj, 95) * 1000:.1f}  |  구간별 X: 중앙 {np.median(alls) * 1000:.1f} mm p95 {np.percentile(alls, 95) * 1000:.1f}")
    if a.save:
        out = E / f"umi_handeye_{a.hand}.json"
        out.write_text(json.dumps(dict(R_C_TCP=RX.as_matrix().tolist(), t_C_TCP_m=tX.tolist(), dt_ms=a.dt_ms, segments=[g[0] for g in good], manifest=str(a.manifest),
                                       heldout_median_mm=float(np.median(allj) * 1000), heldout_p95_mm=float(np.percentile(allj, 95) * 1000)), indent=1))
        print(f"저장 {out}")


if __name__ == "__main__":
    main()
