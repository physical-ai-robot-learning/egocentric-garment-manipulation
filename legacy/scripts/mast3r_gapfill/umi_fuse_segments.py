"""UMI TCP fusion over RESTARTED MASt3R-SLAM segments (2026-10-21, GPT: quality-triggered restart, segment stitching). Input: the
manifest of run_segmented.py (each segment = its own SLAM world; frame_idx local to the segment, valid = tracked part before `cut`).
  1. dt + camera->TCP X from the segment with the most tag anchors (hand-eye, as umi_fuse_slam)
  2. each segment: Sim3 (SLAM world -> robot base) by weighted Umeyama of SLAM camera centres onto tag-derived camera centres
     (+ slowly varying translation correction, umi_fuse_slam v2); a segment with too few anchors is CHAINED to an already placed
     neighbour through the overlap frames (weight lower); a segment that can be neither anchored nor chained is dropped
  3. per frame the best covering segment (most anchors; chained after anchored); SLAM-less frames: tag pose, else IMU orientation +
     interpolated position (flagged)
  4. --validate: hide tag anchors in 0.5 / 1 / 2 s windows, refit, error at the hidden frames
Output <take>/umi_traj_<side>_fused.npz (same keys as umi_fuse_slam + seg).
    .venv/bin/python scripts/umi_fuse_segments.py datasets/tag_rig/<take> <manifest.json> <frames.json> [--hand right] [--validate] [--save]"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam_vs_tag import interp_rot          # noqa: E402
from umi_fuse_slam import umeyama           # noqa: E402


def load_segments(manifest, frames_json, run_dir):
    fr = {f["i"]: f["t_ns"] for f in json.loads(Path(frames_json).read_text())["frames"]}; segs = []
    for m in json.loads(Path(manifest).read_text())["segments"]:
        p = Path(run_dir) / Path(m["csv"]).name
        if not p.exists() or m["cut"] < 10: continue
        T, P, Q, MF = [], [], [], []
        for r in csv.DictReader(open(p)):
            k = int(r["frame_idx"])
            if k >= m["cut"] or r["state"] not in ("INIT", "TRACKING"): continue
            q = [float(r[c]) for c in ("q_x", "q_y", "q_z", "q_w")]
            if np.linalg.norm(q) < 1e-6 or (m["start"] + k) not in fr: continue
            T.append(fr[m["start"] + k]); P.append([float(r["x"]), float(r["y"]), float(r["z"])]); Q.append(q)
            MF.append(float(r["match_frac"]) if r["match_frac"] not in ("", "nan") else 0.0)
        if len(T) >= 10: segs.append(dict(t=np.array(T, np.int64), p=np.array(P), R=R.from_quat(np.array(Q)), mf=np.array(MF), start=m["start"], cut=m["cut"]))
    return segs


def sim3_resid(pc, pb):
    """position-only Umeyama Sim3 (SLAM -> base) residual per anchor (m) and the scale"""
    A = pc - pc.mean(0); B = pb - pb.mean(0); U, S, Vt = np.linalg.svd(B.T @ A / len(A)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    s = (S * np.diag(D)).sum() / max(A.var(0).sum(), 1e-12); return np.linalg.norm(s * (U @ D @ Vt @ A.T).T - B, axis=1), s


def trim_diverged(segs, tt0, pb0, max_res=0.04, min_anchors=15):
    """10-21 (113243): MASt3R kept saying TRACKING while 7 of 17 segments blew up (positions 1e1-1e4 SLAM units). Keep the longest prefix
    whose position-only Sim3 onto the tag anchors has median residual < max_res; segments without enough anchors are kept as they are
    (they can still be chained)."""
    out, log = [], []
    for sg in segs:
        idx = np.where((tt0 >= sg["t"][0]) & (tt0 <= sg["t"][-1]))[0]
        if len(idx) < min_anchors: out.append(sg); continue
        pc = np.stack([np.interp(tt0[idx], sg["t"], sg["p"][:, k]) for k in range(3)], 1)
        good = 0
        for n in range(min_anchors, len(idx) + 1, 3):
            if np.median(sim3_resid(pc[:n], pb0[idx[:n]])[0]) < max_res: good = n
            elif good and n - good > 9: break
        if len(idx) - good <= 2: out.append(sg); continue
        if good < min_anchors: log.append(f"{sg['start']}:버림"); continue
        k = np.searchsorted(sg["t"], tt0[idx[good - 1]], side="right")
        out.append({**{kk: (v[:k] if isinstance(v, np.ndarray) else v) for kk, v in sg.items() if kk != "R"}, "R": sg["R"][:k]}); log.append(f"{sg['start']}:{len(sg['t'])}->{k}")
    print(f"[폭주 구간 정리] 태그와 Sim3 중앙 오차 < {max_res * 1000:.0f} mm 인 앞부분만 남김: " + (", ".join(log) or "없음"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("manifest"); ap.add_argument("frames"); ap.add_argument("--hand", default="right")
    ap.add_argument("--tau-s", type=float, default=3.0); ap.add_argument("--w-single", type=float, default=0.5); ap.add_argument("--min-anchors", type=int, default=8)
    ap.add_argument("--wrist-anchors", help="SLAM 입력 폴더 (datasets/m3_frames/<tag>): 손목 카메라가 시작점 태그 400/401 을 본 순간을 절대 앵커로")
    ap.add_argument("--jaw", help="umi_jaw.npz: 집기/놓기 앞뒤 1.5초의 SLAM 지속 비율"); ap.add_argument("--he-seg", type=int, help="손눈 보정에 쓸 구간 시작 프레임 (기본: 자동)"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--save", action="store_true"); a = ap.parse_args()
    E = Path(a.take); z = np.load(E / f"umi_traj_{a.hand}.npz"); tT = z["t_ns"].astype(np.int64); pB = z["p_B"]; RB = R.from_matrix(z["R_B_tcp"]); nt = z["ntags"]
    segs = load_segments(a.manifest, a.frames, Path(a.manifest).parent); cov = sum(len(s["t"]) for s in segs)
    print(f"구간 {len(segs)}개, 추적 프레임 합 {cov} (겹침 포함)")
    tag = nt > 0; tt0 = tT[tag]; pb0 = pB[tag]; Rb0 = RB[np.where(tag)[0]]; wt0 = np.where(nt[tag] >= 2, 1.0, a.w_single)
    segs = trim_diverged(segs, tt0, pb0)
    def anchors_in(sg, dt, mask=None):
        m = (tt0 + dt >= sg["t"][0]) & (tt0 + dt <= sg["t"][-1])
        if mask is not None: m &= ~mask
        idx = np.where(m)[0]; j = np.clip(np.searchsorted(sg["t"], tt0[idx] + dt), 1, len(sg["t"]) - 1)
        near = np.minimum(np.abs(sg["t"][j] - tt0[idx] - dt), np.abs(sg["t"][j - 1] - tt0[idx] - dt)) < 0.08e9
        return idx[near]
    # 1. dt + X on the best segment ------------------------------------------------------------------------------------------------------
    # 10-21: the segment with the MOST anchors was a still one (4.5 cm, 9 deg) -> X unobservable (camera->TCP came out 0.5 m). Pick among
    # segments with real excitation (TCP rotation >= 25 deg and travel >= 15 cm) the one whose position-only Sim3 fits best
    def excite(sg):
        ai = anchors_in(sg, 0)
        if len(ai) < 30: return None
        rot = np.degrees(np.linalg.norm((Rb0[ai[0]].inv() * Rb0[ai]).as_rotvec(), axis=1)).max(); trav = np.ptp(pb0[ai], 0).max()
        pc = np.stack([np.interp(tt0[ai], sg["t"], sg["p"][:, k]) for k in range(3)], 1)
        return (rot, trav, float(np.median(sim3_resid(pc, pb0[ai])[0])))
    cand = [(sg, e) for sg in segs if (e := excite(sg)) and e[0] >= 25 and e[1] >= 0.15]
    best = min(cand, key=lambda c: c[1][2])[0] if cand else max(segs, key=lambda sg: len(anchors_in(sg, 0)))
    if a.he_seg is not None: best = next(sg for sg in segs if sg["start"] == a.he_seg)
    print("[손눈 후보] " + ", ".join(f"{sg['start']}: 회전 {e[0]:.0f}° 이동 {e[1] * 100:.0f}cm 잔차 {e[2] * 1000:.0f}mm" for sg, e in cand))
    g = np.arange(max(best["t"][0], tT[0]) + int(0.3e9), min(best["t"][-1], tT[-1]) - int(0.3e9), int(1e7))
    def speed(t_src, Rs, tt):
        Ri = interp_rot((t_src - t_src[0]) / 1e9, Rs, (tt - t_src[0]) / 1e9); return np.r_[np.linalg.norm((Ri[:-1].inv() * Ri[1:]).as_rotvec(), axis=1) / 0.01, 0]
    wT = speed(tT, RB, g); dt = int(max(np.arange(-0.3e9, 0.3e9, 1e7), key=lambda d: np.nan_to_num(np.corrcoef(wT, speed(best["t"], best["R"], g + d))[0, 1])))
    ai = anchors_in(best, dt); tt, pb, Rb = tt0[ai], pb0[ai], Rb0[ai]
    Rc = interp_rot((best["t"] - best["t"][0]) / 1e9, best["R"], (tt + dt - best["t"][0]) / 1e9); pc = np.stack([np.interp(tt + dt, best["t"], best["p"][:, k]) for k in range(3)], 1)
    pairs = [(i, j) for i in range(len(tt)) for j in [min(np.searchsorted(tt, tt[i] + 4e8), len(tt) - 1)] if 0.25e9 < tt[j] - tt[i] < 0.7e9]
    A_ = np.array([(Rc[i].inv() * Rc[j]).as_rotvec() for i, j in pairs]); B_ = np.array([(Rb[i].inv() * Rb[j]).as_rotvec() for i, j in pairs])
    big = np.linalg.norm(A_, axis=1) > np.radians(4); U, _, Vt = np.linalg.svd(B_[big].T @ A_[big]); RX = R.from_matrix((U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T)
    RA = R.from_matrix(np.array([(Rb[i] * RX.inv() * Rc[i].inv()).as_matrix() for i in range(len(tt))])).mean()
    M = np.vstack([np.c_[RA.apply(pc[i])[:, None], (RA * Rc[i]).as_matrix(), np.eye(3)] for i in range(len(tt))]); sol = np.linalg.lstsq(M, np.concatenate(pb), rcond=None)[0]
    def res(x):
        RA_, RX_ = R.from_rotvec(x[0:3]), R.from_rotvec(x[3:6]); pp = np.exp(x[6]) * RA_.apply(pc) + (RA_ * Rc).apply(x[7:10]) + x[10:13]
        return np.concatenate([(pp - pb).ravel(), ((RA_ * Rc * RX_).inv() * Rb).as_rotvec().ravel() * 0.05])
    x = least_squares(res, np.r_[RA.as_rotvec(), RX.as_rotvec(), np.log(abs(sol[0])), sol[1:4], sol[4:7]], loss="huber", f_scale=0.01).x
    RXf, tX = R.from_rotvec(x[3:6]), x[7:10]
    print(f"[손눈] 최다 앵커 구간 {best['start']}: dt {dt / 1e6:+.0f} ms, 카메라->TCP {np.round(tX * 1000).astype(int)} mm (앵커 {len(ai)})")
    camB0 = pb0 - (Rb0 * RXf.inv()).apply(tX); n_orb = len(tt0)
    if a.wrist_anchors:   # 10-21: the wrist camera sees the start tags (400/401) -> absolute camera pose = T_B_tag * T_tag_C (undistorted SLAM frames)
        import cv2
        sa = json.loads((Path(__file__).resolve().parents[1] / "configs/calibration/start_anchors_v001.json").read_text()); size = float(sa["tag_size_m"])
        fj = json.loads((Path(a.wrist_anchors) / "frames.json").read_text()); Pm = np.array(fj["P"]); K = Pm[:3, :3]
        h = size / 2; obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm); wt, wc, ww = [], [], []
        for f in fj["frames"][::2]:
            img = cv2.imread(str(Path(a.wrist_anchors) / "rgb" / f"{f['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            cs, ids, _ = det.detectMarkers(img)
            if ids is None: continue
            for c, i in zip(cs, ids.ravel()):
                if str(int(i)) not in sa["tags"]: continue
                q = c.reshape(4, 2).astype(np.float64); side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, 0), axis=1)))
                if side < 30: continue
                ok, rv, tv = cv2.solvePnP(obj, q, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok: continue
                T_C_tag = np.eye(4); T_C_tag[:3, :3] = cv2.Rodrigues(rv)[0]; T_C_tag[:3, 3] = tv.ravel()
                tg = sa["tags"][str(int(i))]; T_B_tag = np.eye(4); T_B_tag[:3, :3] = np.array(tg["R_B"]); T_B_tag[:3, 3] = tg["p_B"]
                T_B_C = T_B_tag @ np.linalg.inv(T_C_tag); wt.append(f["t_ns"] - dt); wc.append(T_B_C[:3, 3]); ww.append(1.0 if side > 60 else 0.5)
        if wt:
            o = np.argsort(np.r_[tt0, wt]); tt0 = np.r_[tt0, wt][o]; camB0 = np.vstack([camB0, wc])[o]; wt0 = np.r_[wt0, ww][o]
            is_orb = np.r_[np.ones(n_orb, bool), np.zeros(len(wt), bool)][o]; pb0 = np.vstack([pb0, np.full((len(wt), 3), np.nan)])[o]
            Rb0 = R.concatenate([Rb0, R.identity(len(wt))])[o]
        print(f"[손목 앵커] 시작점 태그가 보인 손목 프레임 {len(wt)}개 (30px 이상) -> 앵커 추가")
    else: is_orb = np.ones(n_orb, bool)
    # 2./3. per-segment Sim3 + merge -------------------------------------------------------------------------------------------------------
    def place(mask=None):
        placed = {}; order = sorted(range(len(segs)), key=lambda i: -len(anchors_in(segs[i], dt, mask)))
        for i in order:
            sg = segs[i]; ai_ = anchors_in(sg, dt, mask)
            if len(ai_) >= a.min_anchors and np.ptp(camB0[ai_], axis=0).max() > 0.03:
                pc_ = np.stack([np.interp(tt0[ai_] + dt, sg["t"], sg["p"][:, k]) for k in range(3)], 1); w = wt0[ai_]
                s_, R_, t_ = umeyama(pc_, camB0[ai_], w); placed[i] = dict(s=s_, R=R_, t=t_, q=float(len(ai_)), ai=ai_, pc=pc_, kind="태그")
        changed = True
        while changed:                       # chain the rest through overlaps with placed neighbours
            changed = False
            for i in range(len(segs)):
                if i in placed: continue
                sg = segs[i]; src_p, dst_p = [], []
                for j, pj in placed.items():
                    if pj["kind"] == "연결" and pj["q"] < 1: continue
                    o = (sg["t"] >= segs[j]["t"][0]) & (sg["t"] <= segs[j]["t"][-1])
                    if o.sum() < 8: continue
                    tj = segs[j]; pj_c = np.stack([np.interp(sg["t"][o], tj["t"], tj["p"][:, k]) for k in range(3)], 1)
                    src_p.append(sg["p"][o]); dst_p.append(pj["s"] * pj_c @ pj["R"].T + pj["t"])
                if src_p:
                    S_, D_ = np.vstack(src_p), np.vstack(dst_p)
                    if np.ptp(D_, axis=0).max() > 0.02:
                        s_, R_, t_ = umeyama(S_, D_, np.ones(len(S_))); placed[i] = dict(s=s_, R=R_, t=t_, q=0.5, ai=np.zeros(0, int), pc=np.zeros((0, 3)), kind="연결"); changed = True
        return placed
    def fused_at(placed, tq):
        """camera->TCP pose at times tq (Orbbec clock) from the best covering placed segment; nan where none"""
        P = np.full((len(tq), 3), np.nan); Rm = [None] * len(tq); sid = np.full(len(tq), -1); bestq = np.full(len(tq), -1.0)
        for i, pl in placed.items():
            sg = segs[i]; m = (tq + dt >= sg["t"][0]) & (tq + dt <= sg["t"][-1])
            if not m.any(): continue
            k = np.where(m)[0]; j = np.clip(np.searchsorted(sg["t"], tq[k] + dt), 1, len(sg["t"]) - 1)
            near = np.minimum(np.abs(sg["t"][j] - tq[k] - dt), np.abs(sg["t"][j - 1] - tq[k] - dt)) < 0.1e9; k = k[near]
            if not len(k): continue
            better = pl["q"] > bestq[k]; k = k[better]
            if not len(k): continue
            pc_ = np.stack([np.interp(tq[k] + dt, sg["t"], sg["p"][:, c]) for c in range(3)], 1); Rc_ = interp_rot((sg["t"] - sg["t"][0]) / 1e9, sg["R"], (tq[k] + dt - sg["t"][0]) / 1e9)
            corr = np.zeros((len(k), 3))
            if len(pl["ai"]) >= a.min_anchors:          # v2 local translation correction from this segment's anchors
                r_ = camB0[pl["ai"]] - (pl["s"] * pl["pc"] @ pl["R"].T + pl["t"])
                for n_, tqq in enumerate(tq[k]):
                    w = wt0[pl["ai"]] * np.exp(-0.5 * ((tt0[pl["ai"]] - tqq) / (a.tau_s * 1e9)) ** 2); corr[n_] = (w[:, None] * r_).sum(0) / (w.sum() + 1.0)
            Rsim = R.from_matrix(pl["R"]); camB = pl["s"] * pc_ @ pl["R"].T + pl["t"] + corr; RBc = Rsim * Rc_
            P[k] = camB + RBc.apply(tX)
            for n_, kk in enumerate(k): Rm[kk] = RBc[n_] * RXf
            sid[k] = i; bestq[k] = pl["q"]
        return P, Rm, sid
    placed = place(); kinds = [placed[i]["kind"] if i in placed else "버림" for i in range(len(segs))]
    print(f"구간 배치: 태그로 {kinds.count('태그')}, 이웃 연결 {kinds.count('연결')}, 버림 {kinds.count('버림')} | 스케일 {[round(placed[i]['s'], 3) for i in sorted(placed)][:12]}")
    P, Rm, sid = fused_at(placed, tT); have = ~np.isnan(P[:, 0])
    e = np.linalg.norm(P[tag & have] - pB[tag & have], axis=1) * 1e3
    print(f"[융합] Orbbec 프레임 중 SLAM 위치 있음 {have.mean() * 100:.0f}% (태그 가려진 프레임 중 {have[~tag].mean() * 100 if (~tag).any() else 0:.0f}%) | 태그 프레임 오차 중앙 {np.median(e):.1f} mm p95 {np.percentile(e, 95):.1f}")
    # GPT KPIs: SLAM continuity where the tags are hidden (esp. around grasps) + segment boundary jumps
    hid = ~tag; print(f"[가림 지속] 태그 가려진 Orbbec 프레임 {int(hid.sum())}개 중 SLAM 위치 있음 {have[hid].mean() * 100 if hid.any() else 0:.0f}%")
    if a.jaw:
        jz = np.load(a.jaw); near_ev = np.zeros(len(tT), bool)
        for te in jz["ev_t"]: near_ev |= np.abs(tT - te) < 1.5e9
        m_ = near_ev & hid
        print(f"  집기/놓기 앞뒤 1.5초 ({len(jz['ev_t'])}개 사건): 태그 가려진 프레임 {int(m_.sum())}개 중 SLAM 있음 {have[m_].mean() * 100 if m_.any() else 0:.0f}% "
              f"(태그 보임 {tag[near_ev].mean() * 100 if near_ev.any() else 0:.0f}%)")
    jumps_p, jumps_r = [], []
    ks = sorted(placed); full_t = np.arange(tT[0], tT[-1], int(1e8))
    for ii, i in enumerate(ks):
        for j in ks[ii + 1:]:
            o0, o1 = max(segs[i]["t"][0], segs[j]["t"][0]), min(segs[i]["t"][-1], segs[j]["t"][-1])
            if o1 - o0 < 0.2e9: continue
            tq = full_t[(full_t + dt >= o0) & (full_t + dt <= o1)]
            if not len(tq): continue
            Pi, Ri_, _ = fused_at({i: placed[i]}, tq); Pj, Rj_, _ = fused_at({j: placed[j]}, tq); ok_ = ~np.isnan(Pi[:, 0]) & ~np.isnan(Pj[:, 0])
            if not ok_.any(): continue
            jumps_p.append(float(np.median(np.linalg.norm(Pi[ok_] - Pj[ok_], axis=1)) * 1e3))
            jumps_r.append(float(np.median([np.degrees((Ri_[k].inv() * Rj_[k]).magnitude()) for k in np.where(ok_)[0]])))
    if jumps_p: print(f"[구간 경계] 겹친 구간 {len(jumps_p)}쌍: 위치 차이 중앙 {np.median(jumps_p):.1f} mm (최대 {max(jumps_p):.1f}), 자세 차이 중앙 {np.median(jumps_r):.1f}° (최대 {max(jumps_r):.1f}) | 기준 <10 mm, <3°")
    if a.validate:
        rng = np.random.default_rng(0)
        for L in (0.5, 1.0, 2.0):
            Es = []
            for _ in range(25):
                c0 = tt0[is_orb][rng.integers(0, int(is_orb.sum()))]; hid = (tt0 >= c0) & (tt0 < c0 + L * 1e9) & is_orb
                if hid.sum() < 3: continue
                pl2 = place(hid); P2, _, _ = fused_at(pl2, tt0[hid]); ok = ~np.isnan(P2[:, 0])
                Es += list(np.linalg.norm(P2[ok] - pb0[hid][ok], axis=1) * 1e3)
            if Es: print(f"  {L:.1f}초 가림: 위치 오차 중앙 {np.median(Es):5.1f} mm p95 {np.percentile(Es, 95):5.1f} (n={len(Es)})")
    if a.save:
        src = np.where(have, 0, np.where(tag, 1, 2)); Pf = np.where(have[:, None], P, np.where(tag[:, None], pB, np.nan)); Rq = [Rm[i] if have[i] else RB[i] for i in range(len(tT))]
        bad = np.isnan(Pf[:, 0]); Pf[bad] = np.stack([np.interp(tT[bad], tT[~bad], Pf[~bad, k]) for k in range(3)], 1)
        lt = tT[tag]; age = np.array([(x - lt[np.searchsorted(lt, x, side="right") - 1]) / 1e9 if np.searchsorted(lt, x, side="right") > 0 else np.inf for x in tT])
        out = E / f"umi_traj_{a.hand}_fused.npz"
        np.savez(out, t_ns=tT, p_B=Pf, R_B_tcp=R.concatenate(Rq).as_matrix(), ntags=nt, source=src, seg=sid, tag_anchor_age_s=age, dt_ns=dt, tX=tX, R_X=RXf.as_matrix())
        print(f"저장 {out}: SLAM {np.mean(src == 0) * 100:.0f}%, 태그만 {np.mean(src == 1) * 100:.0f}%, 보간 {np.mean(src == 2) * 100:.0f}%")


if __name__ == "__main__":
    main()
