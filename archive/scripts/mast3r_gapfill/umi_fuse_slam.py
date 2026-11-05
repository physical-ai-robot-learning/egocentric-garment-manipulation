"""UMI TCP trajectory fusion v1 (2026-10-20, GPT plan step 5): wrist SLAM relative motion + AprilTag absolute anchors (+ IMU orientation
for the gaps). Offline, per take. Tags are SPARSE, QUALITY-WEIGHTED anchors (never a per-frame overwrite):
  1. dt (wrist vs Orbbec clock) and the camera -> TCP extrinsic X: global hand-eye fit on tag frames (scripts/slam_vs_tag.py model)
  2. SLAM split into SEGMENTS at tracking losses / relocalisations (> --gap-s): MASt3R's scale can jump across them
  3. per segment a TIME-VARYING Sim3 (SLAM world -> robot base): weighted Umeyama of SLAM camera centres onto tag-derived camera centres
     (T_B_cam = T_B_TCP X^-1), Gaussian time window --tau-s around each grid time (0.25 s), anchor weight 1 (2+ tags) / --w-single
     (1 tag); too few anchors -> the window widens up to the whole segment; a segment without anchors -> global Sim3, flagged
  4. fused TCP = Sim3(t) * T_W_cam(t) * X ; SLAM invalid but tags visible -> tag pose ; neither -> IMU-fused orientation + position
     interpolated (source = 2, flagged)
  5. blind validation (GPT): hide every tag anchor inside windows of 0.5 / 1 / 2 s, re-fuse, error at the hidden frames vs the tags
Output <take>/umi_traj_<side>_fused.npz: t_ns, p_B, R_B_tcp, ntags, source (0 slam, 1 tag, 2 interp), tag_anchor_age_s, local_scale,
slam_quality (match fraction), the same keys umi_replay / converters read.
    .venv/bin/python scripts/umi_fuse_slam.py datasets/tag_rig/<take> <m3.csv> <frames.json> [--validate] [--save]"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam_vs_tag import load_slam, interp_rot      # noqa: E402


def umeyama(src, dst, w):
    """weighted Sim3 with dst ~ s R src + t"""
    w = w / w.sum(); ms, md = w @ src, w @ dst; S, D = src - ms, dst - md
    C = (D * w[:, None]).T @ S; U, sig, Vt = np.linalg.svd(C); E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ E @ Vt; var = (w * (S ** 2).sum(1)).sum(); s = (sig * np.diag(E)).sum() / max(var, 1e-12)
    return s, Rm, md - s * Rm @ ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("csv"); ap.add_argument("frames"); ap.add_argument("--hand", default="right")
    ap.add_argument("--tau-s", type=float, default=3.0); ap.add_argument("--w-single", type=float, default=0.5); ap.add_argument("--gap-s", type=float, default=0.3)
    ap.add_argument("--min-anchors", type=int, default=15); ap.add_argument("--validate", action="store_true"); ap.add_argument("--save", action="store_true"); a = ap.parse_args()
    E = Path(a.take); z = np.load(E / f"umi_traj_{a.hand}.npz"); tT = z["t_ns"].astype(np.int64); pB = z["p_B"]; RB = R.from_matrix(z["R_B_tcp"]); nt = z["ntags"]
    tS, pS, RS, ok, st = load_slam(a.csv, a.frames); mf = np.array([float(r["match_frac"]) if r["match_frac"] not in ("", "nan") else np.nan for r in csv.DictReader(open(a.csv))])[:len(ok)]
    tS, pS, RS, mf = tS[ok], pS[ok], RS[np.where(ok)[0]], mf[ok]
    # --- 1. dt + global hand-eye (X fixed afterwards) -------------------------------------------------------------------------------
    g = np.arange(max(tS[0], tT[0]) + int(0.5e9), min(tS[-1], tT[-1]) - int(0.5e9), int(1e7))
    def speed(t_src, Rs, tt):
        Ri = interp_rot((t_src - t_src[0]) / 1e9, Rs, (tt - t_src[0]) / 1e9); return np.r_[np.linalg.norm((Ri[:-1].inv() * Ri[1:]).as_rotvec(), axis=1) / 0.01, 0]
    wT = speed(tT, RB, g); dt = int(max(np.arange(-0.3e9, 0.3e9, 1e7), key=lambda d: np.corrcoef(wT, speed(tS, RS, g + d))[0, 1]))
    tag = (nt > 0) & (tT + dt >= tS[0]) & (tT + dt <= tS[-1]); tt = tT[tag]; pb = pB[tag]; Rb = RB[np.where(tag)[0]]
    Rc = interp_rot((tS - tS[0]) / 1e9, RS, (tt + dt - tS[0]) / 1e9); pc = np.stack([np.interp(tt + dt, tS, pS[:, k]) for k in range(3)], 1)
    pairs = [(i, j) for i in range(len(tt)) for j in [min(np.searchsorted(tt, tt[i] + 4e8), len(tt) - 1)] if 0.3e9 < tt[j] - tt[i] < 0.6e9]
    A_ = np.array([(Rc[i].inv() * Rc[j]).as_rotvec() for i, j in pairs]); B_ = np.array([(Rb[i].inv() * Rb[j]).as_rotvec() for i, j in pairs])
    big = np.linalg.norm(A_, axis=1) > np.radians(5); U, _, Vt = np.linalg.svd(B_[big].T @ A_[big]); RX = R.from_matrix((U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T)
    RA = R.from_matrix(np.array([(Rb[i] * RX.inv() * Rc[i].inv()).as_matrix() for i in range(len(tt))])).mean()
    M = np.vstack([np.c_[RA.apply(pc[i])[:, None], (RA * Rc[i]).as_matrix(), np.eye(3)] for i in range(len(tt))]); sol = np.linalg.lstsq(M, np.concatenate(pb), rcond=None)[0]
    def res(x):
        RA_, RX_ = R.from_rotvec(x[0:3]), R.from_rotvec(x[3:6]); pp = np.exp(x[6]) * RA_.apply(pc) + (RA_ * Rc).apply(x[7:10]) + x[10:13]
        return np.concatenate([(pp - pb).ravel(), ((RA_ * Rc * RX_).inv() * Rb).as_rotvec().ravel() * 0.05])
    x = least_squares(res, np.r_[RA.as_rotvec(), RX.as_rotvec(), np.log(abs(sol[0])), sol[1:4], sol[4:7]], loss="huber", f_scale=0.01).x
    RXf, tX, s_glob, RA_glob, tA_glob = R.from_rotvec(x[3:6]), x[7:10], np.exp(x[6]), R.from_rotvec(x[0:3]), x[10:13]
    print(f"[전역] dt {dt / 1e6:+.0f} ms, 스케일 {s_glob:.4f}, 카메라->TCP {np.round(tX * 1000).astype(int)} mm")
    # anchors: tag-derived CAMERA centre in the base, weights by tag count
    cam_B = pb - (Rb * RXf.inv()).apply(tX); w_tag = np.where(nt[tag] >= 2, 1.0, a.w_single)
    # --- 2. SLAM segments -------------------------------------------------------------------------------------------------------------
    cut = np.where(np.diff(tS) > a.gap_s * 1e9)[0]; seg_id = np.zeros(len(tS), int)
    for c in cut: seg_id[c + 1:] += 1
    seg_of = lambda t: seg_id[np.clip(np.searchsorted(tS, t) - 1, 0, len(tS) - 1)]
    anc_seg = seg_of(tt + dt); nseg = seg_id.max() + 1
    print(f"SLAM 구간 {nseg}개 (끊김 > {a.gap_s}s 기준), 구간별 앵커 수 {np.bincount(anc_seg, minlength=nseg).tolist()}")
    # --- 3/4. time-varying Sim3 per segment + fused trajectory -------------------------------------------------------------------------
    def fuse(mask_anchor=None):
        use = np.ones(len(tt), bool) if mask_anchor is None else ~mask_anchor
        sim = {}                                                    # (seg, grid t) -> (s, R, t)
        out_s, out_R, out_t, qual = np.zeros(len(tS)), [], np.zeros((len(tS), 3)), np.zeros(len(tS))
        for sg in range(nseg):
            idx = np.where(seg_id == sg)[0]; ai = np.where(use & (anc_seg == sg))[0]
            grid = np.arange(tS[idx[0]], tS[idx[-1]] + 1, int(0.25e9)); P = []
            # v2 (10-20): local Sim3 was unstable on short windows (free take: scale 0.10-0.26, p95 125-200 mm). Scale + rotation are now
            # ONE per SLAM segment (what a relocalisation can change); only a slowly varying TRANSLATION correction follows the anchors.
            if len(ai) >= 4: s_seg, R_seg, t_seg = umeyama(pc[ai], cam_B[ai], w_tag[ai])
            else: s_seg, R_seg, t_seg = s_glob, RA_glob.as_matrix(), tA_glob
            r_seg = cam_B[ai] - (s_seg * pc[ai] @ R_seg.T + t_seg) if len(ai) else np.zeros((0, 3))
            for gt in grid:
                if len(ai) < 4: P.append((s_seg, R_seg, t_seg, 0.0)); continue
                tau = a.tau_s
                while True:
                    w = w_tag[ai] * np.exp(-0.5 * ((tt[ai] + dt - gt) / (tau * 1e9)) ** 2)
                    if (w > 0.05).sum() >= a.min_anchors or tau > 60: break
                    tau *= 1.5
                corr = (w[:, None] * r_seg).sum(0) / (w.sum() + 1.0)          # +1: shrink to 0 when the window has few anchors
                P.append((s_seg, R_seg, t_seg + corr, float((w > 0.05).sum())))
            gs = np.array([p[0] for p in P]); gr = R.from_matrix(np.array([p[1] for p in P])); gtt = np.array([p[2] for p in P]); gq = np.array([p[3] for p in P])
            tq = tS[idx]; out_s[idx] = np.interp(tq, grid, gs); out_t[idx] = np.stack([np.interp(tq, grid, gtt[:, k]) for k in range(3)], 1)
            Rg = Slerp((grid - grid[0]) / 1e9, gr)(np.clip((tq - grid[0]) / 1e9, 0, (grid[-1] - grid[0]) / 1e9)) if len(grid) > 1 else R.concatenate([gr[0]] * len(tq))
            out_R.append((idx, Rg)); qual[idx] = np.interp(tq, grid, gq)
        Rsim = [None] * len(tS)
        for idx, Rg in out_R:
            for k, i in enumerate(idx): Rsim[i] = Rg[k]
        Rsim = R.concatenate(Rsim); camB = out_s[:, None] * Rsim.apply(pS) + out_t; RBcam = Rsim * RS
        return camB + (RBcam).apply(tX), RBcam * RXf, out_s, qual
    pF, RF, sF, qF = fuse()
    # error on tag frames (in-sample)
    def at_tag(pF_, RF_, sel):
        ti = tt[sel] + dt; pp = np.stack([np.interp(ti, tS, pF_[:, k]) for k in range(3)], 1); Rp = interp_rot((tS - tS[0]) / 1e9, RF_, (ti - tS[0]) / 1e9)
        inside = np.array([np.min(np.abs(tS - x)) < 0.1e9 for x in ti])
        return np.linalg.norm(pp - pb[sel], axis=1)[inside] * 1e3, np.degrees((Rp.inv() * Rb[np.where(sel)[0]]).magnitude())[inside]
    e, er = at_tag(pF, RF, np.ones(len(tt), bool))
    print(f"[융합] 태그 프레임 위치 오차 중앙 {np.median(e):.1f} mm p95 {np.percentile(e, 95):.1f} | 자세 중앙 {np.median(er):.1f}° | 국소 스케일 범위 {sF.min():.3f}-{sF.max():.3f}")
    if a.validate:   # hide anchors in windows, re-fuse, error at the hidden frames
        rng = np.random.default_rng(0)
        print("가린 시간별 오차 (그 구간 태그 앵커를 빼고 융합 -> 가린 프레임에서 태그와 비교; 기준 <10 매우 좋음, 10-20 usable)")
        for L in (0.5, 1.0, 2.0):
            E1, E2, R1 = [], [], []
            for _ in range(25):
                c0 = tt[rng.integers(0, len(tt))]; hid = (tt >= c0) & (tt < c0 + L * 1e9)
                if hid.sum() < 3: continue
                pF2, RF2, _, _ = fuse(hid); e2, r2 = at_tag(pF2, RF2, hid)
                ti = tt[hid] + dt; pg = s_glob * RA_glob.apply(np.stack([np.interp(ti, tS, pS[:, k]) for k in range(3)], 1)) + (RA_glob * interp_rot((tS - tS[0]) / 1e9, RS, (ti - tS[0]) / 1e9)).apply(tX) + tA_glob
                E1 += list(e2); R1 += list(r2); E2 += list(np.linalg.norm(pg - pb[hid], axis=1) * 1e3)
            if E1: print(f"  {L:.1f}초 가림: 융합 위치 중앙 {np.median(E1):5.1f} mm p95 {np.percentile(E1, 95):5.1f} | 자세 중앙 {np.median(R1):.1f}° | (전역 Sim3 하나: 중앙 {np.median(E2):5.1f} p95 {np.percentile(E2, 95):5.1f})")
    if a.save:
        # on every Orbbec frame time inside the take: SLAM-fused where valid, tag where only tags, else IMU orientation + interpolated position
        tq = tT; src = np.full(len(tq), 2); P = np.zeros((len(tq), 3)); Rq = []
        near = np.array([np.min(np.abs(tS - (x + dt))) < 0.1e9 for x in tq])
        Pf = np.stack([np.interp(tq + dt, tS, pF[:, k]) for k in range(3)], 1); Rf = interp_rot((tS - tS[0]) / 1e9, RF, (tq + dt - tS[0]) / 1e9)
        for i in range(len(tq)):
            if near[i]: P[i] = Pf[i]; Rq.append(Rf[i]); src[i] = 0
            elif nt[i] > 0: P[i] = pB[i]; Rq.append(RB[i]); src[i] = 1
            else: P[i] = np.nan; Rq.append(RB[i])
        bad = np.isnan(P[:, 0]); P[bad] = np.stack([np.interp(tq[bad], tq[~bad], P[~bad, k]) for k in range(3)], 1)
        lt = tq[nt > 0]; age = np.array([(x - lt[np.searchsorted(lt, x, side="right") - 1]) / 1e9 if np.searchsorted(lt, x, side="right") > 0 else np.inf for x in tq])
        out = E / f"umi_traj_{a.hand}_fused.npz"
        np.savez(out, t_ns=tq, p_B=P, R_B_tcp=R.concatenate(Rq).as_matrix(), ntags=nt, source=src, tag_anchor_age_s=age,
                 local_scale=np.interp(tq + dt, tS, sF), anchor_count=np.interp(tq + dt, tS, qF), slam_quality=np.interp(tq + dt, tS, np.nan_to_num(mf, nan=0.0)),
                 dt_ns=dt, tX=tX, R_X=RXf.as_matrix())
        print(f"저장 {out}: SLAM {np.mean(src == 0) * 100:.0f}%, 태그만 {np.mean(src == 1) * 100:.0f}%, 보간 {np.mean(src == 2) * 100:.0f}%")


if __name__ == "__main__":
    main()
