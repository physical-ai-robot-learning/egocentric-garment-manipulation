"""SLAM-as-primary-trajectory evaluation for one hand of a restarted-MASt3R take (2026-10-21, GPT: decide on 4 numbers).
  - hand-eye: camera->TCP translation FIXED (tape / joint solve: right -5,34,122 mm), rotation + per-segment Sim3 solved on ALIGNMENT tag
    frames only; VALIDATION tag frames (1 s hidden every 4 s) never touch the fit
  - tail cut: a segment ends at the first run of >= --cut-run alignment anchors with residual > --cut-mm (then refit on the prefix)
  - KPIs: coverage (frames with a SLAM pose) | pickup/place survival = tag-HIDDEN frames within +-1.5 s of jaw events that have a SLAM
    pose | hidden error = TCP error on the validation frames | boundary jump = position difference where two kept segments overlap
  - diagnostics: per segment start/end/kept/scale, segment ends near jaw events (+-1 s), image brightness + hand speed just before an end
    .venv/bin/python scripts/umi_slam_eval.py datasets/tag_rig/<take> datasets/m3_runs/<tag>/segs/manifest.json datasets/m3_frames/<tag> \
        --hand right --fix-t=-5,34,122 [--jaw datasets/tag_rig/<take>/wrist/umi_jaw.npz] [--save]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/Users/jeonghwanlee/ego_collector/scripts")
import umi_joint_handeye as JH                                           # noqa: E402
from umi_fuse_segments import load_segments, sim3_resid                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("manifest"); ap.add_argument("frames_dir"); ap.add_argument("--hand", default="right")
    ap.add_argument("--fix-t", required=True, help="camera->TCP mm, e.g. -5,34,122"); ap.add_argument("--dt-ms", type=float, default=20.0)
    ap.add_argument("--jaw"); ap.add_argument("--cut-mm", type=float, default=25.0); ap.add_argument("--cut-run", type=int, default=4)
    ap.add_argument("--hide-every", type=float, default=4.0); ap.add_argument("--hide-s", type=float, default=1.0); ap.add_argument("--save", action="store_true"); ap.add_argument("--handeye-in"); ap.add_argument("--handeye-out")
    ap.add_argument("--max-dev", type=float, default=0.15); ap.add_argument("--good-rot", type=float, default=25.0); ap.add_argument("--good-travel", type=float, default=0.15); ap.add_argument("--good-res", type=float, default=0.015)
    a = ap.parse_args(); E = Path(a.take); dt = int(a.dt_ms * 1e6)
    JH.FIX_T = np.array([float(v) for v in a.fix_t.split(",")]) / 1000
    z = np.load(E / f"umi_traj_{a.hand}.npz"); tT = z["t_ns"].astype(np.int64); nt = z["ntags"]; tag = nt > 0
    tt0, pb0, Rb0 = tT[tag], z["p_B"][tag], R.from_matrix(z["R_B_tcp"][tag])
    rel = (tt0 - tt0[0]) / 1e9; val = (rel % a.hide_every) < a.hide_s; aln = ~val
    segs = load_segments(a.manifest, str(Path(a.frames_dir) / "frames.json"), Path(a.manifest).parent)
    print(f"[{a.hand}] 구간 {len(segs)}개 | 태그 프레임 {tag.sum()} (정렬용 {aln.sum()}, 검증용 {val.sum()})")

    def obs(sg, idx):
        pc, Rc = JH.seg_obs(sg, tt0[idx], dt); return (pc, Rc, pb0[idx], Rb0[idx])
    def in_seg(sg, mask):
        return np.where(mask & (tt0 + dt >= sg["t"][0]) & (tt0 + dt <= sg["t"][-1]))[0]
    # 1. hand-eye rotation on the good segments (alignment frames only)
    good = []
    for sg in segs:
        ia = in_seg(sg, aln)
        if len(ia) < 30: continue
        pc, _ = JH.seg_obs(sg, tt0[ia], dt); rot = np.degrees(np.linalg.norm((Rb0[ia[0]].inv() * Rb0[ia]).as_rotvec(), axis=1)).max()
        if rot >= a.good_rot and np.ptp(pb0[ia], 0).max() >= a.good_travel and np.median(sim3_resid(pc, pb0[ia])[0]) < a.good_res: good.append(obs(sg, ia))
    if a.handeye_in:
        hj = json.loads(Path(a.handeye_in).read_text()); RX = R.from_rotvec(hj["rotvec"]); tX = np.array(hj["t"]); print("  [handeye borrowed from", a.handeye_in, "]")
    else:
        if not good: sys.exit("손눈 회전을 풀 좋은 구간이 없음")
        RX, tX, _ = JH.solve(good)
    if a.handeye_out: Path(a.handeye_out).write_text(json.dumps(dict(rotvec=RX.as_rotvec().tolist(), t=np.asarray(tX).tolist())))
    print(f"  손눈: 이동 고정 {np.round(tX * 1000).astype(int)} mm, 회전 {np.degrees(np.linalg.norm(RX.as_rotvec())):.0f}° (좋은 구간 {len(good)}개)")
    # 2. per segment: Sim3 with X fixed, continuous-residual tail cut, refit
    kept = []
    for sg in segs:
        ia = in_seg(sg, aln)
        if len(ia) < 8: continue
        (RA, s, t), = JH.solve([obs(sg, ia)], fixX=(RX, tX))[2]
        pp, _ = JH.tcp_pred(RA, s, t, *JH.seg_obs(sg, tt0[ia], dt), RX, tX); r = np.linalg.norm(pp - pb0[ia], axis=1)
        bad = r > a.cut_mm / 1000; cut_t = sg["t"][-1]; reason = "끝까지"
        run = 0
        for k, b in enumerate(bad):
            run = run + 1 if b else 0
            if run >= a.cut_run: cut_t = int(tt0[ia[k - run + 1]]) + dt; reason = f"잔차 {a.cut_mm:.0f} mm 초과 연속"; break
        if cut_t < sg["t"][-1]:
            ia = ia[tt0[ia] + dt < cut_t]
            if len(ia) < 8: print(f"  구간 {sg['start']}: 버림 ({reason}, 남은 앵커 {len(ia)})"); continue
            (RA, s, t), = JH.solve([obs(sg, ia)], fixX=(RX, tX))[2]
        if s < 0.05 or s > 3.0: print(f"  구간 {sg['start']}: 버림 (스케일 {s:.3f} 비정상)"); continue
        # 10-21: a segment with few anchors can run away where no tag sees it (right 171057: z -1.68 m). Cut it at the first frame that
        # leaves the tag/IMU baseline (umi_traj_<hand>, interpolated in gaps) by more than --max-dev.
        tt_s = sg["t"][(sg["t"] >= sg["t"][0]) & (sg["t"] <= cut_t)] - dt
        if len(tt_s):
            pp_s = JH.tcp_pred(RA, s, t, *JH.seg_obs(sg, tt_s, dt), RX, tX)[0]
            base = np.stack([np.interp(tt_s, tT, z["p_B"][:, k]) for k in range(3)], 1); dev = np.linalg.norm(pp_s - base, axis=1)
            if (dev > a.max_dev).any():
                k0 = int(np.argmax(dev > a.max_dev))
                if k0 < 10: print(f"  구간 {sg['start']}: 버림 (처음부터 기준 궤적과 {dev.max() * 100:.0f} cm 차이)"); continue
                cut_t = int(tt_s[k0] + dt); reason = f"기준 궤적과 {a.max_dev * 100:.0f} cm 넘게 벌어짐"
        kept.append(dict(sg=sg, RA=RA, s=s, t=t, t0=sg["t"][0], t1=min(cut_t, sg["t"][-1]), n_al=len(ia), reason=reason))
    print(f"  유지 구간 {len(kept)}개 / {len(segs)}")
    # 3. per-frame fused position over every Orbbec frame
    def pred_at(times):
        P = np.full((len(times), 3), np.nan); src = np.full(len(times), -1)
        for k_, kp in sorted(enumerate(kept), key=lambda x: x[1]["n_al"]):        # most anchors wins (written last)
            m = (times + dt >= kp["t0"]) & (times + dt <= kp["t1"])
            if m.any():
                pc, Rc = JH.seg_obs(kp["sg"], times[m], dt); P[m] = JH.tcp_pred(kp["RA"], kp["s"], kp["t"], pc, Rc, RX, tX)[0]; src[m] = k_
        return P, src
    P_all, src_all = pred_at(tT); cov = np.mean(~np.isnan(P_all[:, 0]))
    Pv, _ = pred_at(tt0[val]); ev = np.linalg.norm(Pv - pb0[val], axis=1); ev = ev[~np.isnan(ev)]
    jumps = []
    for i, ka in enumerate(kept):
        for kb in kept[i + 1:]:
            lo, hi = max(ka["t0"], kb["t0"]), min(ka["t1"], kb["t1"])
            if hi - lo < 2e8: continue
            tm = np.linspace(lo, hi, 10).astype(np.int64) - dt
            pa = JH.tcp_pred(ka["RA"], ka["s"], ka["t"], *JH.seg_obs(ka["sg"], tm, dt), RX, tX)[0]; pb = JH.tcp_pred(kb["RA"], kb["s"], kb["t"], *JH.seg_obs(kb["sg"], tm, dt), RX, tX)[0]
            jumps.append(float(np.median(np.linalg.norm(pa - pb, axis=1))))
    surv, near_end = None, []
    if a.jaw and Path(a.jaw).exists():
        jz = np.load(a.jaw); evt = jz["ev_t"].astype(np.int64)
        hid = ~tag; near = np.zeros(len(tT), bool)
        for e in evt: near |= np.abs(tT - e) < 1.5e9
        hm = hid & near; surv = (float(np.mean(~np.isnan(P_all[hm, 0]))), int(hm.sum()))
        for kp in kept:
            if kp["reason"] != "끝까지" or kp["t1"] < kp["sg"]["t"][-1]: near_end.append(float(np.min(np.abs(evt - kp["t1"])) / 1e9) if len(evt) else 9.0)
    # 4. diagnostics just before each segment end: brightness + hand speed
    import cv2
    fj = json.loads((Path(a.frames_dir) / "frames.json").read_text()); ft = np.array([f["t_ns"] for f in fj["frames"]]); fi = np.array([f["i"] for f in fj["frames"]])
    diag = []
    for kp in kept:
        k = int(np.searchsorted(ft, kp["t1"])) - 3
        img = cv2.imread(str(Path(a.frames_dir) / "rgb" / f"{int(fi[max(k, 0)]):06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = (tt0 > kp["t1"] - 1e9) & (tt0 <= kp["t1"]); spd = float(np.linalg.norm(np.diff(pb0[m], axis=0), axis=1).sum() / max((tt0[m][-1] - tt0[m][0]) / 1e9, 1e-3)) if m.sum() > 2 else float("nan")
        diag.append((kp["sg"]["start"], float(round((kp["t0"] - tT[0]) / 1e9, 1)), float(round((kp["t1"] - tT[0]) / 1e9, 1)), kp["n_al"], round(float(kp["s"]), 3), kp["reason"],
                     None if img is None else round(float(img.mean()), 0), None if np.isnan(spd) else round(spd * 100, 1)))
    print("  구간 (시작프레임, 시작s, 끝s, 정렬앵커, 스케일, 끝난 이유, 끝 직전 밝기, 끝 직전 손 속도 cm/s):")
    for d in diag: print("   ", d)
    if near_end: print(f"  잘린 구간의 끝이 집게 사건 ±1초 안: {np.mean(np.array(near_end) < 1.0) * 100:.0f}% ({len(near_end)}개)")
    sc = np.array([d[4] for d in diag] or [np.nan]); print(f"  구간 스케일: 중앙 {np.median(sc):.3f}, 범위 {sc.min():.3f}-{sc.max():.3f}")
    print(f"[핵심 {a.hand}] 커버리지 {cov * 100:.0f}% | 집기·놓기 주변 태그 가려진 프레임 SLAM 생존 " + (f"{surv[0] * 100:.0f}% (n={surv[1]})" if surv else "jaw 없음")
          + f" | 숨긴 구간 오차 중앙 {np.median(ev) * 1000:.1f} mm p95 {np.percentile(ev, 95) * 1000:.1f} (n={len(ev)})" if len(ev) else " | 숨긴 구간 오차 없음")
    print(f"[핵심 {a.hand}] 구간 경계 차이 중앙 {np.median(jumps) * 1000:.1f} mm (쌍 {len(jumps)})" if jumps else f"[핵심 {a.hand}] 겹치는 구간 없음")
    if a.save:
        out = E / f"umi_traj_{a.hand}_fused.npz"; ok = ~np.isnan(P_all[:, 0])
        np.savez(out, t_ns=tT, p_B=np.where(ok[:, None], P_all, z["p_B"]), R_B_tcp=z["R_B_tcp"], ntags=nt, slam_ok=ok, seg_src=src_all,
                 handeye_R=RX.as_matrix(), handeye_t=tX, note="10-21 umi_slam_eval: SLAM where kept, else tag/IMU (umi_traj_<hand>)")
        print(f"저장 {out} (SLAM {ok.mean() * 100:.0f}%, 나머지 태그/IMU)")


if __name__ == "__main__":
    main()
