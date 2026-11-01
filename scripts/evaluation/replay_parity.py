"""UMI -> robot replay parity (2026-10-20, GPT plan step 6). Per outward stroke (hold before -> hold after, displacements relative to the
stroke start, robot base frame): A = UMI measured (umi_tcp_traj p_B), B = robot commanded (FK of the replayed q), C = robot executed (FK of
the observed q). Compared on NORMALIZED PROGRESS s in [0,1] (projection on A's chord, each series normalised by its own length, monotone),
so lag / speed differences do not count as error.
Per stroke: main-axis ratio C/A (and B/A), cross-axis leakage |perp(C(s)-A(s))| / |A main|, endpoint |dC-dA|, RMSE / p95 |C(s)-A(s)|,
B-A (transform + IK) and C-B (robot tracking) split, A's own straightness and angle to the robot axis (human, informational).
Per direction: 3D endpoint repeat std of A and C over the repeats.
Gates: ratio 0.9-1.1, leakage < 10% (better < 5%), endpoint < 20 mm, RMSE < 20 mm, p95 < 30 mm, repeat std C < 10 mm.
    .venv/bin/python scripts/evaluation/replay_parity.py datasets/tag_rig/<take> datasets/replay/<run>        # A/B/C
    .venv/bin/python scripts/evaluation/replay_parity.py datasets/tag_rig/<take> --umi-only [--hand 오른손]     # A only (before the robot), axes from the data
    .venv/bin/python scripts/evaluation/replay_parity.py --synthetic                                            # self-test"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
AXR = {"YAW": AX["Z"], "PITCH": AX["Y"], "ROLL": AX["X"]}     # 10-20 --proto rot: rotation about the robot-base axis through the jaw tip
# 10-20 GPT gates for rotation: ratio 0.9-1.1, geodesic median < 5 deg, p95 < 10 deg, unwanted axes < 10 %, fake TCP translation < 10 mm,
# repeat < 3 deg. ("mm" fields hold DEGREES for rotation strokes.)
GATE_ROT = dict(ratio=(0.9, 1.1), leak=0.10, end_mm=5.0, med_deg=5.0, p95_mm=10.0, fake_mm=10.0, rep_deg=3.0, sat_deg_s=40.0)
# gripper-local meaning at READY (R_B_E: x_E ~ base +Z up, y_E ~ base -Y, z_E = approach ~ base +X), user/GPT 10-20: record the physical axis
PHYS = {"YAW": "sweep 좌우 (x_E 축)", "PITCH": "bend 상하 (y_E 축)", "ROLL": "twist 집게축 (z_E 축)"}
S = np.linspace(0, 1, 101)
GATE = dict(ratio=(0.9, 1.1), leak=0.10, end_mm=20, rmse_mm=20, p95_mm=30, rep_mm=10)


def on_progress(P, u):
    """displacements P (n,3) relative to their start, resampled at S by the monotone projection on u, normalised by its own end"""
    P = P - P[0]; x = P @ u; L = x[-1] if abs(x[-1]) > 1e-6 else 1e-6; s = np.maximum.accumulate(np.clip(x / L, 0, None))
    s, idx = np.unique(s, return_index=True); P = P[idx]
    if len(s) < 3: return None
    return np.stack([np.interp(S, s, P[:, j]) for j in range(3)], 1)


def perp(v, e): return v - np.outer(v @ e, e)


def stroke(A, e, B=None, C=None):
    u = A[-1] - A[0]; u = u / max(np.linalg.norm(u), 1e-9); a = on_progress(A, u); r = {}
    if a is None: return None
    mA = float((A[-1] - A[0]) @ e); r.update(main_A_mm=mA * 1e3, leak_A=float(np.linalg.norm(perp(a, e), axis=1).max() / abs(mA)),
                                             angle_A_deg=float(np.degrees(np.arccos(np.clip(abs(u @ e), 0, 1)))), end_A=(A[-1] - A[0]))
    for nm, X in (("B", B), ("C", C)):
        if X is None: continue
        x = on_progress(X, u)
        if x is None: continue
        d = np.linalg.norm(x - a, axis=1) * 1e3
        r.update({f"ratio_{nm}": float((X[-1] - X[0]) @ e / mA), f"leak_{nm}": float(np.linalg.norm(perp(x - a, e), axis=1).max() / abs(mA)),
                  f"end_{nm}_mm": float(np.linalg.norm((X[-1] - X[0]) - (A[-1] - A[0])) * 1e3), f"rmse_{nm}_mm": float(np.sqrt((d ** 2).mean())),
                  f"p95_{nm}_mm": float(np.percentile(d, 95)), f"end_{nm}": X[-1] - X[0]})
        if nm == "C" and "ratio_B" in r: r["rmse_CB_mm"] = float(np.sqrt((np.linalg.norm(x - on_progress(B, u), axis=1) ** 2).mean()) * 1e3)
    return r


def judge(r, rot=False):
    if rot:
        g = GATE_ROT; f = []
        if not g["ratio"][0] <= r["ratio_C"] <= g["ratio"][1]: f.append("비율")
        if r["leak_C"] >= g["leak"]: f.append("축혼입")
        if r["end_C_mm"] >= g["end_mm"]: f.append("끝각도")
        if r["geo_med_deg"] >= g["med_deg"] or r["p95_C_mm"] >= g["p95_mm"]: f.append("경로각도")
        if r.get("fake_mm", 0) >= g["fake_mm"]: f.append("가짜이동")
        if r.get("sat_deg_s", 0) >= g["sat_deg_s"]: f.append("포화")
        return f
    g = GATE; f = []
    if not g["ratio"][0] <= r["ratio_C"] <= g["ratio"][1]: f.append("비율")
    if r["leak_C"] >= g["leak"]: f.append("새어나감")
    if r["end_C_mm"] >= g["end_mm"]: f.append("끝점")
    if r["rmse_C_mm"] >= g["rmse_mm"] or r["p95_C_mm"] >= g["p95_mm"]: f.append("경로")
    return f


def summarize(rows, mode):
    if any(k.split()[0] in AXR for k, _ in rows):
        print("\n회전 축 의미 (READY 그리퍼 기준): " + ", ".join(f"{k} = {v}" for k, v in PHYS.items()) + "\n회전 행: A 이동 = 사람 회전각, 끝점/RMSE/p95/C-B = 도(°), 새어남 = 다른 축 회전 비율")
    print(f"\n{'구간':14s} {'A 이동':>7s} {'A 각도':>6s} {'A 휨':>5s}" + ("" if mode == "A" else f" {'C/A':>5s} {'B/A':>5s} {'새어남':>6s} {'끝점':>6s} {'RMSE':>6s} {'p95':>6s} {'C-B':>6s}  판정"))
    fails = 0
    for k, r in rows:
        line = f"{k:14s} {r['main_A_mm']:6.0f}mm {r['angle_A_deg']:5.1f}° {r['leak_A'] * 100:4.0f}%"
        rot = k.split()[0] in AXR; u = "°" if rot else "mm"
        if rot: line = f"{k:14s} {r['main_A_mm']:6.1f}° {r['angle_A_deg']:5.1f}° {r['leak_A'] * 100:4.0f}%"
        if mode != "A" and "ratio_C" in r:
            f = judge(r, rot); fails += bool(f)
            line += (f" {r['ratio_C']:5.2f} {r.get('ratio_B', np.nan):5.2f} {r['leak_C'] * 100:5.1f}% {r['end_C_mm']:5.1f}{u} {r['rmse_C_mm']:5.1f}{u} {r['p95_C_mm']:5.1f}{u} "
                     f"{r.get('rmse_CB_mm', np.nan):5.1f}{u}  " + ("PASS" if not f else "FAIL(" + ",".join(f) + ")"))
            if rot: line += f" | 각오차 중앙 {r['geo_med_deg']:.1f}° p95 {r['p95_C_mm']:.1f}° | 가짜이동 최대 {r['fake_mm']:.0f} mm (중앙 {r['fake_med_mm']:.0f}) | 사람 끝 흔들림 {r['tip_A_mm']:.0f} mm | 관절속도 p99 {r['sat_deg_s']:.0f}°/s"
        print(line)
    print("\n방향별 반복 (끝점 3D 표준편차; A = 사람 동작 편차, 참고):")
    groups = {}
    for k, r in rows: groups.setdefault(k.rsplit(" ", 1)[0], []).append(r)
    rep_bad = 0
    for g, L in groups.items():
        if len(L) < 2: continue
        sA = float(np.sqrt(np.trace(np.cov(np.array([x["end_A"] for x in L]).T))) * 1e3); u = "°" if g.split()[0] in AXR else "mm"
        msg = f"  {g:12s} 반복 {len(L)}회: A {sA:5.1f} {u}"
        if mode != "A" and all("end_C" in x for x in L):
            # 10-20: each repeat replays a DIFFERENT human stroke, so C's own spread just follows A's; the robot's repeatability is the
            # spread of the endpoint ERROR (C - A) over the repeats
            sC = float(np.sqrt(np.trace(np.cov(np.array([x["end_C"] - x["end_A"] for x in L]).T))) * 1e3); rep_bad += sC >= GATE["rep_mm"]
            lim_ = GATE_ROT["rep_deg"] if u == "°" else GATE["rep_mm"]; rep_bad += (sC >= lim_) - (sC >= GATE["rep_mm"])
            msg += f", 로봇 끝점오차(C-A) {sC:5.1f} {u}" + ("" if sC < lim_ else f"  <- {lim_:.0f} {u} 이상")
        print(msg)
    if mode != "A":
        allr = [r for _, r in rows if "ratio_C" in r]
        d = np.array([r["rmse_C_mm"] for r in allr])
        print(f"\n전체: 구간 {len(allr)}개, 통과 {len(allr) - fails}, RMSE 중앙 {np.median(d):.1f} mm, 최대 {d.max():.1f} mm, 반복 기준 넘은 방향 {rep_bad}")
        print("최종:", "PASS" if fails == 0 and rep_bad == 0 else "FAIL",
              "| B-A 가 크면 좌표변환/IK 문제, C-B 가 크면 로봇 추종 문제, A 각도가 크면 사람이 로봇 축과 다르게 움직인 것(오류 아님)")
        return fails == 0 and rep_bad == 0
    return True


def windows(sg):
    a = (sg["hold_pre"][0] + sg["hold_pre"][1]) // 2; b = (sg["hold_mid"][0] + sg["hold_mid"][1]) // 2; return a, b


def sel(t, P, a, b, h0=None):
    m = (t >= a) & (t <= b)
    if m.sum() < 5: return None
    X = P[m].copy()
    if h0 is not None:                          # start = median of the hold before the stroke (operator jitter)
        hm = (t >= h0[0]) & (t <= h0[1])
        if hm.sum() >= 3: X[0] = np.median(P[hm], 0)
    return X


def rot_stroke(z, R_, g, sp, w0, w1):
    """10-20 rotation stroke: spatial relative rotation (robot base) of A (UMI), B (commanded FK), C (executed FK) on normalised progress.
    Rotvecs are scaled deg/1000 so stroke()'s "mm" fields read in DEGREES. Also the jaw-tip drift during the rotation (A vs C)."""
    from scipy.spatial.transform import Rotation as Rr
    if "R_B_tcp" not in z or "rot_cmd" not in R_: print(f"  {g['axis']} {g['dir']} {g['rep']}: 회전 데이터 없음 (--orient 재생 + R_B_tcp 필요)"); return None
    tA = z["t_ns"].astype(np.int64); src = R_["src_t_ns"].astype(np.int64); tC = src[0] + ((R_["t_obs_ns"] - R_["t_cmd_ns"][0]) * sp).astype(np.int64)
    h0 = (tA >= g["hold_pre"][0]) & (tA <= g["hold_pre"][1]); mA = (tA >= w0) & (tA <= w1)
    if mA.sum() < 5: return None
    R0 = Rr.from_matrix(z["R_B_tcp"][h0]).mean() if h0.sum() >= 3 else Rr.from_matrix(z["R_B_tcp"][mA][0])
    A = (Rr.from_matrix(z["R_B_tcp"][mA]) * R0.inv()).as_rotvec(); A = np.vstack([np.zeros(3), A])
    def rel(t, rv, a_, b_):
        m = (t >= a_) & (t <= b_)
        if m.sum() < 5: return None, m
        Rs = Rr.from_rotvec(rv[m]); return (Rs * Rs[0].inv()).as_rotvec(), m
    e = g["sign"] * AXR[g["axis"]]; mb = (src >= w0) & (src <= w1); xb = R_["rot_cmd"][mb] @ e; xc = R_["rot_obs"] @ e
    lag = min(np.arange(0, 1.0e9, 2e7), key=lambda L: np.mean((np.interp(src[mb] + L, tC, xc) - xb) ** 2)) if mb.sum() > 5 else 0   # both vs READY
    B, _ = rel(src, R_["rot_cmd"], w0, w1); C, mc = rel(tC, R_["rot_obs"], w0 + int(lag), w1 + int(lag))
    if B is None or C is None: return None
    sc = lambda X: np.degrees(X) / 1000
    r = stroke(sc(A), e, sc(B), sc(C))
    if r is None: return None
    u = A[-1] - A[0]; u = u / max(np.linalg.norm(u), 1e-9); a_, c_ = on_progress(A, u), on_progress(C, u)
    geo = np.degrees((Rr.from_rotvec(c_) * Rr.from_rotvec(a_).inv()).magnitude()) if a_ is not None and c_ is not None else np.array([np.nan])
    # fake TCP translation: executed tip displacement minus the UMI's own tip displacement, matched in (lag-shifted) time
    tAm = tA[mA]; pA = z["p_B"][mA] - z["p_B"][mA][0]; tCm = tC[mc] - int(lag); pC = R_["p_obs"][mc] - R_["p_obs"][mc][0]
    pCi = np.stack([np.interp(tAm, tCm, pC[:, k]) for k in range(3)], 1); fake = np.linalg.norm(pCi - pA, axis=1) * 1e3
    qc = np.degrees(R_["q_cmd"]); dt = np.diff(R_["t_cmd_ns"]) / 1e9; vq = np.abs(np.diff(qc, axis=0)) / np.maximum(dt, 1e-3)[:, None]
    r.update(geo_med_deg=float(np.nanmedian(geo)), p95_C_mm=float(np.nanpercentile(geo, 95)), fake_mm=float(fake.max()), fake_med_mm=float(np.median(fake)),
             tip_A_mm=float(np.linalg.norm(pA, axis=1).max() * 1e3), sat_deg_s=float(np.percentile(vq.max(1), 99)))
    return r


GATE_FREE = dict(pos_rmse=20.0, pos_p95=30.0, rot_med=5.0, rot_p95=10.0, ratio=(0.9, 1.1), jumps=0)


def free_stroke(z, R_, g, sp, plot_path=None):
    """10-20 combined free motion (GPT): whole hold->motion->hold window, A = UMI, B = commanded FK, C = executed FK, C time-shifted by its
    fitted lag (3D position), so lag is not error. Position RMSE/p95, orientation geodesic median/p95, translation and rotation amplitude
    ratios (RMS distance from the start pose), path-length ratio, start/end pose error, C-B residual, discontinuities, overlay plot."""
    from scipy.spatial.transform import Rotation as Rr
    tA = z["t_ns"].astype(np.int64); src = R_["src_t_ns"].astype(np.int64); tC0 = src[0] + ((R_["t_obs_ns"] - R_["t_cmd_ns"][0]) * sp).astype(np.int64)
    a0 = (g["hold_pre"][0] + g["hold_pre"][1]) // 2; a1 = (g["hold_post"][0] + g["hold_post"][1]) // 2
    tg = src[(src >= a0) & (src <= a1)]
    if len(tg) < 20: return None
    h0 = (tA >= g["hold_pre"][0]) & (tA <= g["hold_pre"][1])
    pA0 = np.median(z["p_B"][h0], 0) if h0.sum() >= 3 else z["p_B"][np.searchsorted(tA, a0)]
    RA0 = Rr.from_matrix(z["R_B_tcp"][h0]).mean() if h0.sum() >= 3 else Rr.from_matrix(z["R_B_tcp"][np.searchsorted(tA, a0)])
    itp = lambda t, T, X: np.stack([np.interp(t, T, X[:, k]) for k in range(X.shape[1])], 1)
    A = itp(tg, tA, z["p_B"]) - pA0; rA = itp(tg, tA, (Rr.from_matrix(z["R_B_tcp"]) * RA0.inv()).as_rotvec())
    B = itp(tg, src, R_["p_cmd"]); B -= B[0]; rB = itp(tg, src, R_["rot_cmd"]) if "rot_cmd" in R_ else np.zeros_like(A)
    lag = min(np.arange(0, 1.5e9, 2e7), key=lambda L: np.mean(np.sum((itp(tg + L, tC0, R_["p_obs"]) - itp(tg, src, R_["p_cmd"])) ** 2, 1)))
    C = itp(tg + lag, tC0, R_["p_obs"]); C -= C[0]; rC = itp(tg + lag, tC0, R_["rot_obs"]) if "rot_obs" in R_ else np.zeros_like(A)
    rel = lambda rv: (Rr.from_rotvec(rv) * Rr.from_rotvec(rv[0]).inv()).as_rotvec()
    rA, rB, rC = rel(rA), rel(rB), rel(rC)
    dp = np.linalg.norm(C - A, axis=1) * 1e3; dr = np.degrees((Rr.from_rotvec(rC) * Rr.from_rotvec(rA).inv()).magnitude())
    rms = lambda X: float(np.sqrt(np.mean(np.sum(X ** 2, 1))))
    plen = lambda X: float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())
    sm = lambda X, k=9: np.stack([np.convolve(np.pad(X[:, j], k // 2, mode="edge"), np.ones(k) / k, "valid") for j in range(3)], 1)
    vC = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.diff(tg / 1e9); vB = np.linalg.norm(np.diff(B, axis=0), axis=1) / np.diff(tg / 1e9)
    jumps = int(np.sum(vC > max(0.25, 3 * np.percentile(vB, 99))))
    r = dict(pos_rmse=float(np.sqrt(np.mean(dp ** 2))), pos_p95=float(np.percentile(dp, 95)), rot_med=float(np.median(dr)), rot_p95=float(np.percentile(dr, 95)),
             tr_ratio=rms(C) / max(rms(A), 1e-6), rot_ratio=rms(rC) / max(rms(rA), 1e-6), path_ratio=plen(sm(C)) / max(plen(sm(A)), 1e-6),
             end_mm=float(np.linalg.norm(C[-1] - A[-1]) * 1e3), end_deg=float(dr[-1]), cb_rmse=float(np.sqrt(np.mean(np.sum((C - B) ** 2, 1))) * 1e3),
             ba_rmse=float(np.sqrt(np.mean(np.sum((B - A) ** 2, 1))) * 1e3), jumps=jumps, lag_ms=float(lag / 1e6), amp_A_mm=float(np.linalg.norm(A, axis=1).max() * 1e3),
             amp_A_deg=float(np.degrees(np.linalg.norm(rA, axis=1).max())), tagged=float(np.mean(np.interp(tg, tA, (z["ntags"] > 0).astype(float)) > 0.5)) if "ntags" in z else 1.0)
    if plot_path is not None:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
        tt = (tg - tg[0]) / 1e9; fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for k, nm in enumerate("XYZ"):
            for X, lab, st in ((A, "A UMI", "-"), (B, "B 명령", "--"), (C, "C 실행", ":")): ax[0, k].plot(tt, X[:, k] * 1e3, st, label=lab)
            ax[0, k].set_title(f"위치 {nm} (mm)")
            for X, lab, st in ((rA, "A", "-"), (rB, "B", "--"), (rC, "C", ":")): ax[1, k].plot(tt, np.degrees(X[:, k]), st, label=lab)
            ax[1, k].set_title(f"회전 {['ROLL(X)', 'PITCH(Y)', 'YAW(Z)'][k]} (deg)"); ax[1, k].set_xlabel("s")
        ax[0, 0].legend(); fig.suptitle(f"{g['hand']} 자유동작 {g['rep']}회: 위치 RMSE {r['pos_rmse']:.1f} mm, 회전 중앙 {r['rot_med']:.1f}°")
        fig.tight_layout(); fig.savefig(plot_path, dpi=90); plt.close(fig)
    return r


GATE_GRIP = dict(phase=0.05, ms=100.0)        # GPT 10-20: phase error < 5 % (target 2-3 %), actual time < 100 ms, missed 0, spurious 0


def grip_events(R_, g, sp, jaw_dir):
    """10-20 gripper timing on normalised phase s in [0,1] of the hold->motion->hold window (UMI source time):
    UMI events (umi_jaw infer: hysteresis on the wrist-fisheye classifier) vs robot COMMAND (g_cmd steps) vs robot ACTUAL (encoder g_obs
    crossing 0.5 with 0.35/0.65 hysteresis). Phase is primary (arm lag ~0.2 s is not a gripper error); times are secondary:
    UMI->command in UMI seconds, command->actual in real robot ms."""
    jz = np.load(Path(jaw_dir) / "umi_jaw.npz"); a0 = (g["hold_pre"][0] + g["hold_pre"][1]) // 2; a1 = (g["hold_post"][0] + g["hold_post"][1]) // 2
    ph = lambda t: (np.asarray(t, float) - a0) / (a1 - a0)
    U = [(int(t), k) for t, k in zip(jz["ev_t"], jz["ev_kind"]) if a0 <= t <= a1]
    src = R_["src_t_ns"].astype(np.int64); gc = R_["g_cmd"]; tcm = R_["t_cmd_ns"].astype(np.int64)
    Cm = [(int(src[i + 1]), int(tcm[i + 1]), "open" if gc[i + 1] > gc[i] else "close") for i in np.where(np.diff(gc) != 0)[0]]
    go = R_["g_obs"]; to = R_["t_obs_ns"].astype(np.int64)[:len(go)]; st = int(go[0] > 0.5); Ac = []
    def cross(i, lvl):          # linear interpolation of the 0.5 crossing between samples i-1 and i (sample spacing up to 100 ms)
        f = (lvl - go[i - 1]) / (go[i] - go[i - 1]) if go[i] != go[i - 1] else 1.0
        return int(to[i - 1] + np.clip(f, 0, 1) * (to[i] - to[i - 1]))
    for i in range(1, len(go)):
        if st == 0 and go[i] > 0.65: st = 1; tx = cross(i, 0.5); Ac.append((int(src[0] + (tx - tcm[0]) * sp), tx, "open"))
        elif st == 1 and go[i] < 0.35: st = 0; tx = cross(i, 0.5); Ac.append((int(src[0] + (tx - tcm[0]) * sp), tx, "close"))
    rows, used_c, used_a = [], set(), set()
    for tu, ku in U:
        ic = min((i for i in range(len(Cm)) if Cm[i][2] == ku and i not in used_c), key=lambda i: abs(Cm[i][0] - tu), default=None)
        ia = min((i for i in range(len(Ac)) if Ac[i][2] == ku and i not in used_a), key=lambda i: abs(Ac[i][0] - tu), default=None)
        if ic is not None and abs(ph(Cm[ic][0]) - ph(tu)) > 0.15: ic = None
        if ia is not None and abs(ph(Ac[ia][0]) - ph(tu)) > 0.15: ia = None
        if ic is not None: used_c.add(ic)
        if ia is not None: used_a.add(ia)
        rows.append(dict(kind=ku, s_umi=float(ph(tu)), s_cmd=float(ph(Cm[ic][0])) if ic is not None else None, s_act=float(ph(Ac[ia][0])) if ia is not None else None,
                         umi_cmd_s=(Cm[ic][0] - tu) / 1e9 if ic is not None else None,
                         cmd_act_ms=(Ac[ia][1] - Cm[ic][1]) / 1e6 if (ic is not None and ia is not None) else None))
    return rows, len(Cm) - len(used_c), len(Ac) - len(used_a)


def judge_free(r):
    g = GATE_FREE; f = []
    if r["pos_rmse"] >= g["pos_rmse"] or r["pos_p95"] >= g["pos_p95"]: f.append("위치")
    if r["rot_med"] >= g["rot_med"] or r["rot_p95"] >= g["rot_p95"]: f.append("회전")
    if not (g["ratio"][0] <= r["tr_ratio"] <= g["ratio"][1]): f.append("이동비율")
    if r["amp_A_deg"] > 10 and not (g["ratio"][0] <= r["rot_ratio"] <= g["ratio"][1]): f.append("회전비율")
    if r["jumps"] > g["jumps"]: f.append("끊김")
    return f


def run_files(a):
    take = Path(a.take); side = {"오른손": "right", "왼손": "left"}[a.hand]
    d = take if (take / f"umi_traj_{side}.npz").exists() else take.parent / (take.name + "_derived")
    traj = json.loads((Path(a.run) / "meta.json").read_text()).get("traj", "") if a.run else ""    # 10-20: compare against the trajectory that was replayed
    z = np.load(d / f"umi_traj_{side}{traj}.npz", allow_pickle=True); segs = json.loads((d / "segments.json").read_text())
    if traj: print(f"A = {d.name}/umi_traj_{side}{traj}.npz")
    tA = z["t_ns"].astype(np.int64); rows = []
    if a.umi_only:
        PA = z["p_B"] if "p_B" in z else z["p_C"]; segs = [g for g in segs if g["hand"] == a.hand]
        E = dict(AX) if "p_B" in z else None
        if E is None:                            # no robot calibration: axes from the data (sign-corrected mean chords, Gram-Schmidt)
            raw = {}
            for g in segs:
                X = sel(tA, PA, *windows(g))
                if X is not None: raw.setdefault(g["axis"], []).append(g["sign"] * (X[-1] - X[0]) / np.linalg.norm(X[-1] - X[0]))
            raw = {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in raw.items()}
            if len(raw) == 3:
                ang = lambda p, q: np.degrees(np.arccos(np.clip(raw[p] @ raw[q], -1, 1)))
                print(f"데이터에서 잡은 축 사이 각도 (90° 가 이상적): X-Y {ang('X', 'Y'):.1f}°, X-Z {ang('X', 'Z'):.1f}°, Y-Z {ang('Y', 'Z'):.1f}° (카메라 좌표, 로봇 캘리브 전)")
            E = raw
        for g in segs:
            X = sel(tA, PA, *windows(g), g["hold_pre"])
            if X is None or g["axis"] not in E: continue
            r = stroke(X, g["sign"] * E[g["axis"]])
            if r: rows.append((f"{g['axis']} {g['dir']} {g['rep']}", r))
        return summarize(rows, "A")
    run = Path(a.run); meta = json.loads((run / "meta.json").read_text()); sp = meta["speed"]
    if all(g["axis"] == "FREE" for g in meta["segments"]):
        print(f"\n{'구간':10s} {'사람 범위':>14s} {'위치RMSE':>8s} {'p95':>6s} {'회전중앙':>7s} {'p95':>6s} {'이동비':>5s} {'회전비':>5s} {'경로비':>5s} {'끝점':>12s} {'C-B':>6s} {'B-A':>6s} {'끊김':>4s}  판정")
        nf = 0; rs = []
        for j, g in enumerate(meta["segments"]):
            f = run / f"seg_{j:02d}.npz"
            if not f.exists() or meta.get("grip_only"): continue      # 10-20 --grip-only: the arm stayed at READY, only the gripper is compared
            r = free_stroke(z, np.load(f), g, sp, plot_path=run / f"free_{j:02d}.png")
            if r is None: print(f"  자유 {g['rep']}: 데이터 부족"); continue
            fl = judge_free(r); nf += bool(fl); rs.append(r)
            print(f"자유 {g['rep']}회     {r['amp_A_mm']:5.0f}mm/{r['amp_A_deg']:3.0f}° {r['pos_rmse']:6.1f}mm {r['pos_p95']:5.1f} {r['rot_med']:6.1f}° {r['rot_p95']:5.1f} {r['tr_ratio']:5.2f} {r['rot_ratio']:5.2f} "
                  f"{r['path_ratio']:5.2f} {r['end_mm']:4.0f}mm/{r['end_deg']:3.1f}° {r['cb_rmse']:5.1f} {r['ba_rmse']:5.1f} {r['jumps']:4d}  " + ("PASS" if not fl else "FAIL(" + ",".join(fl) + ")")
                  + f"  (지연 {r['lag_ms']:.0f} ms, 태그 보임 {r['tagged'] * 100:.0f}%)")
        print(f"\n전체: {len(rs)}개 중 통과 {len(rs) - nf} | 기준: 위치 RMSE<20 p95<30 mm, 회전 중앙<5° p95<10°, 이동/회전 비율 0.9-1.1, 끊김 0 | 그림: {run}/free_XX.png")
        if meta.get("gripper"):
            print(f"\n집게 타이밍 (진행률 s, UMI = 손목 어안 판정, 명령 = 로봇에 보낸 g, 실제 = 엔코더) | 기준: 진행률 오차 < {GATE_GRIP['phase'] * 100:.0f}%, 명령->실제 < {GATE_GRIP['ms']:.0f} ms, 누락 0, 헛동작 0")
            gfail = 0; allc = []
            for j, g in enumerate(meta["segments"]):
                f = run / f"seg_{j:02d}.npz"
                if not f.exists() or "g_cmd" not in np.load(f): continue
                rows, sp_c, sp_a = grip_events(np.load(f), g, sp, meta["gripper"])
                for e in rows:
                    miss = e["s_cmd"] is None or e["s_act"] is None
                    d_uc = None if e["s_cmd"] is None else e["s_cmd"] - e["s_umi"]; d_ua = None if e["s_act"] is None else e["s_act"] - e["s_umi"]
                    bad = miss or abs(d_ua) >= GATE_GRIP["phase"] or (e["cmd_act_ms"] is not None and e["cmd_act_ms"] >= GATE_GRIP["ms"]); gfail += bad
                    if e["cmd_act_ms"] is not None: allc.append(e["cmd_act_ms"])
                    fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
                    uc = "" if d_uc is None else f"{d_uc * 100:+.1f}%"; ucs = "" if e["umi_cmd_s"] is None else f"{e['umi_cmd_s'] * 1000:+.0f} ms"
                    ua = "누락" if d_ua is None else f"{d_ua * 100:+.1f}%"; ca = "" if e["cmd_act_ms"] is None else f"{e['cmd_act_ms']:.0f} ms"
                    print(f"  자유 {g['rep']}회 {'열림' if e['kind'] == 'open' else '닫힘'}: UMI s={e['s_umi']:.3f} | 명령 s={fmt(e['s_cmd'])} | 실제 s={fmt(e['s_act'])} | "
                          f"UMI->명령 {uc} ({ucs}), UMI->실제 {ua}, 명령->실제 {ca}  {'FAIL' if bad else 'PASS'}")
                if sp_c or sp_a: print(f"  자유 {g['rep']}회: 헛동작 명령 {sp_c}개, 실제 {sp_a}개  FAIL"); gfail += sp_c + sp_a
            print(f"집게 판정: {'PASS' if gfail == 0 else 'FAIL'}" + (f" | 명령->실제 지연 중앙 {np.median(allc):.0f} ms (로봇 집게 구동 지연)" if allc else ""))
        return nf == 0
    for j, g in enumerate(meta["segments"]):
        f = run / f"seg_{j:02d}.npz"
        if not f.exists(): continue
        R_ = np.load(f); w0, w1 = windows(g)
        if g["axis"] in AXR:
            r = rot_stroke(z, R_, g, sp, w0, w1)
            if r: rows.append((f"{g['axis']} {g['dir']} {g['rep']}", r))
            continue
        src = R_["src_t_ns"].astype(np.int64); tC = src[0] + ((R_["t_obs_ns"] - R_["t_cmd_ns"][0]) * sp).astype(np.int64)
        # 10-20: C window = B window shifted by C's measured lag (was a fixed +0.5 s: the operator often kept moving in the 'hold', so the
        # extra 0.5 s of C showed up as a fake 40 mm endpoint error). lag = best match of C(t + lag) to B(t) along the stroke direction
        u_ = R_["p_cmd"][-1] - R_["p_cmd"][0]; Bf = sel(src, R_["p_cmd"], w0, w1)
        if Bf is not None and np.linalg.norm(Bf[-1] - Bf[0]) > 1e-3: u_ = (Bf[-1] - Bf[0]) / np.linalg.norm(Bf[-1] - Bf[0])
        mb = (src >= w0) & (src <= w1); xb = R_["p_cmd"][mb] @ u_; xc = R_["p_obs"] @ u_
        lag = min(np.arange(0, 1.0e9, 2e7), key=lambda L: np.mean((np.interp(src[mb] + L, tC, xc) - xb) ** 2)) if mb.sum() > 5 else 0
        A = sel(tA, z["p_B"], w0, w1, g["hold_pre"]); B = sel(src, R_["p_cmd"], w0, w1); C = sel(tC, R_["p_obs"], w0 + int(lag), w1 + int(lag))
        if A is None or B is None or C is None: print(f"  {g['axis']} {g['dir']} {g['rep']}: 데이터 부족"); continue
        r = stroke(A, g["sign"] * AX[g["axis"]], B, C)
        if r: rows.append((f"{g['axis']} {g['dir']} {g['rep']}", r))
    return summarize(rows, "ABC")


def synthetic():
    rng = np.random.default_rng(3); ok_all = True
    for case, leak, scale in (("정상 로봇 (C = 0.97배, 1.5% 새어남)", 0.015, 0.97), ("불량 로봇 (X 이동 때 Z 로 15% 새어남)", 0.15, 1.0)):
        print(f"\n=== 합성: {case}"); rows = []
        for ax in "XYZ":
            for sg in (+1, -1):
                for rep in range(3):
                    n = 60; tt = np.linspace(0, 1, n); mj = 10 * tt ** 3 - 15 * tt ** 4 + 6 * tt ** 5
                    A = np.outer(mj, AX[ax] * sg * 0.15) + np.outer(np.sin(np.pi * tt), rng.normal(0, 0.006, 3)) + rng.normal(0, 0.0015, (n, 3))
                    B = A.copy(); m2 = 10 * np.linspace(0, 1, 2 * n) ** 3 - 15 * np.linspace(0, 1, 2 * n) ** 4 + 6 * np.linspace(0, 1, 2 * n) ** 5
                    Bs = np.stack([np.interp(np.linspace(0, 1, 2 * n), tt, B[:, j]) for j in range(3)], 1)
                    lag = np.r_[np.zeros(6), m2[:-6]]                                   # executed: lags, scaled, maybe leaks
                    C = scale * np.stack([np.interp(lag, m2, Bs[:, j]) for j in range(3)], 1) + rng.normal(0, 0.0005, (2 * n, 3))
                    if ax == "X": C[:, 2] += leak * 0.15 * lag * sg * (1 if leak < 0.1 else 1)
                    r = stroke(A, sg * AX[ax], Bs, C); rows.append((f"{ax} {'+' if sg > 0 else '-'} {rep + 1}", r))
        res = summarize(rows, "ABC"); ok_all &= (res if "정상" in case else not res)
    print("\n합성 시험", "PASS (정상은 통과, 불량은 걸러냄)" if ok_all else "FAIL"); return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?"); ap.add_argument("run", nargs="?"); ap.add_argument("--hand", default="오른손", choices=["오른손", "왼손"])
    ap.add_argument("--umi-only", action="store_true"); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    if a.synthetic: sys.exit(0 if synthetic() else 1)
    if not a.take or (not a.run and not a.umi_only): ap.error("take 와 run (또는 --umi-only) 필요")
    run_files(a)


if __name__ == "__main__":
    main()
