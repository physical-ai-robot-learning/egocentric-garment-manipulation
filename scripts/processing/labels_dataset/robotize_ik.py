#!/usr/bin/env python3
"""[2026-10-22] Robotization mode "ik_v1" (user option B): NO midline y-compression.  The human canonical TCP is kept wherever the
reBot can follow it; only short offending spans are warped, smoothly, and grasp windows have position priority.

Per demo window and per arm (arm-labelled inputs in data/tag_rig_derived/<take><src>/umi_traj_<arm>.npz, see tag_rig_to_lerobot.py):
  human target   T_h = [R_B_tcp @ R_U_E(rig) | p_B] on the 20 Hz grid (5-frame position / rotation-vector smoothing); where the
                 finger corner would hit the table the tool is first tilted toward vertical (umi_replay_bi.level, <= --level-max 30 deg,
                 keeps the human TCP height), any remainder is lifted (corner_lift, --tip-clear 0.005, +0.005 left)
  IK             umi_replay_bi's PinkIkSolver with the BRIDGE joint limits (0.5 deg inside), joint offset / base / TCP overrides,
                 orientation_cost 0.1 (position first), 7.5 deg/frame bound (= its 5 deg at 30 Hz), first frame unbounded, wrist-roll
                 yield above |j6| 115 deg, tool-axis 180 deg flip chosen per arm (fewer > 10 mm frames)
  inter-arm gap  umi_replay_bi's link-origin set (link3-6, gripper_link, fingers, tcp) of both arms; min distance >= --arm-gap (0.12 m).
                 Each violating span is fixed by pushing ONE arm's target away from the other along base y with a raised-cosine bump
                 (flat on the span, 0.5 s ramps).  The arm with fewer grasp-window frames in the span moves; if both grasp, both move half.
                 Up to --rounds re-solves.
  label          robot-feasible TCP = FK(q) of the (5-frame smoothed) joint path, in reBot EEF axes (same frame as R_B_tcp @ R_U_E)
  10-26 contract (extra keys, old keys unchanged): <arm>_p/R_fused (fusion + axes + 0.25 s smoothing), <arm>_p/R_rob = + table clearance =
                 action_cartesian_robotized (POLICY TARGET, before IK; tool flip / wrist-roll yield / arm-gap push are runtime corrections and
                 NOT in it), <arm>_p_iktgt (+ gap push), IK QA: err_pos_rob_mm, err_rot_rob_deg (jaw-symmetric), roll_relax_deg, lim_margin_deg,
                 gap_push_mm, ik_valid (< 10 mm vs the IK target); <arm>_p/R = action_cartesian_ik_fk (old label, comparison)
Outputs <derived>/ik_v1/demoNN.npz: t_ns, per arm {p_h, R_h, p, R, q, ik_mm, lim (any joint within 1 deg of a limit), grasp},
gap (min link distance per frame, m), meta json.  Run with ~/xvla-mac/bin/python (pinocchio + pink).
usage: robotize_ik.py <take_dir> <derived_take_dir> [--demos 0,1] [--workers 6]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

EGO = Path.home() / "ego_collector"; sys.path.insert(0, str(EGO))
DT = 0.05
ALIGN = {"right": EGO / "configs/calibration/umi_gripper_align_v001.json", "left": EGO / "configs/calibration/umi_gripper_align_left_v001.json"}


def demos_of(take):
    m = json.loads((take / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
    return [dict(idx=i, t0_ns=t0 + int(p["t0"] * 1e9), t1_ns=t0 + int((p["t0"] + p["dur"]) * 1e9)) for i, p in enumerate([p for p in m["protocol"] if p["text"].startswith("자유:")])]


def smooth(X, k):
    if k <= 1: return X
    w = np.ones(k) / k; return np.stack([np.convolve(np.pad(X[:, j], k // 2, mode="edge"), w, "valid")[:len(X)] for j in range(X.shape[1])], 1)


class Arm:
    def __init__(self, side, a):
        from visual_teleop import config as vcfg
        from visual_teleop.robot.ik import ArmIk
        from visual_teleop.robot.ik_pink import PinkIkSolver, PinkParams
        from visual_teleop.tools.robot_moves import READY_Q_DEG
        import pinocchio as pin
        c = vcfg.load(); self.s, self.a, self.pin = side, a, pin
        jda = np.zeros(14)
        from ego_teleop.robot.rebot_client import SIDE_SLICE
        for s_ in ("left", "right"): jda[SIDE_SLICE[s_]] = READY_Q_DEG[s_]
        self.qr = np.radians(READY_Q_DEG[side])
        pk = PinkIkSolver(side, PinkParams(**c.ik["pink"])); pk.sync(jda)
        mk = lambda step, it: PinkIkSolver(side, PinkParams(max_joint_delta_deg=step, iters=it, orientation_cost=a.ori_cost))
        self.free, self.f0 = mk(a.ik_step, 200), mk(180, 400); self.fine = mk(2.0, 60)
        L = ArmIk.from_cfg(c, pk).gate.cfg.limits; self.lo, self.hi = np.asarray(L.lower_rad), np.asarray(L.upper_rad)
        for sv in (self.free, self.f0, self.fine):
            sv.sync(jda)
            for k_, iq in enumerate(sv.ridx):
                jo_ = sv.joff[side][k_]
                sv.model.lowerPositionLimit[iq] = max(sv.model.lowerPositionLimit[iq], self.lo[k_] + jo_ + np.radians(0.5))
                sv.model.upperPositionLimit[iq] = min(sv.model.upperPositionLimit[iq], self.hi[k_] + jo_ - np.radians(0.5))
        self.pk = pk
        self.fids = [i for i, f in enumerate(pk.model.frames) if f.name in (f"{side}_link3", f"{side}_link4", f"{side}_link5", f"{side}_link6",
                                                                              f"{side}_gripper_link", f"{side}_gripper_left", f"{side}_gripper_right", f"{side}_tcp")]

    def corner_lift(self, T, g_):
        a = self.a; low = T[2, 3] - a.jaw_half * abs(T[2, 0]) - (0.008 + 0.028 * float(np.clip(g_, 0, 1))) * abs(T[2, 1])
        return max(0.0, a.table_z + a.tip_clear + (a.tip_clear_left_extra if self.s == "left" else 0.0) - low)

    def level(self, T, g_):
        """umi_replay_bi.level (10-21): where the lowest finger corner would hit the table, tilt the tool axis toward straight down by
        the SMALLEST angle (<= --level-max) that clears it, keeping the TCP position (human height); elsewhere untouched"""
        from scipy.spatial.transform import Rotation as Rot
        a = self.a
        if a.level_max <= 0 or self.corner_lift(T, g_) <= 0: return T
        ax = T[:3, 2]; c = np.cross(ax, [0.0, 0.0, -1.0]); n_ = np.linalg.norm(c)
        if n_ < 1e-6: return T
        amax = min(np.arccos(np.clip(-ax[2], -1, 1)), np.radians(a.level_max)); u = c / n_
        def rot(al): T2 = T.copy(); T2[:3, :3] = Rot.from_rotvec(u * al).as_matrix() @ T[:3, :3]; return T2
        if self.corner_lift(rot(amax), g_) > 0: return rot(amax)
        lo, hi = 0.0, amax
        for _ in range(20):
            md = (lo + hi) / 2; lo, hi = (md, hi) if self.corner_lift(rot(md), g_) > 0 else (lo, md)
        return rot(hi)

    def _multistart(self, q_prev, T, rng, n_rand):
        """unbounded solve from ready + previous + n_rand uniform seeds inside the bridge limits -> best (q, pe)"""
        seeds = [self.qr, q_prev] + [rng.uniform(self.lo + 0.05, self.hi - 0.05) for _ in range(n_rand)]
        best = None
        for sd in seeds:
            q, pe, _ = self.f0.ik(sd, T)
            if best is None or pe < best[1]: best = (q, pe)
            if pe < 1e-3: break
        return best

    def _multistart_pref(self, q_prev, T, rng, n_rand, tol=0.010):
        """multi-start; among results within tol of the best residual prefer the one closest to q_prev (joint space)"""
        seeds = [self.qr, q_prev] + [rng.uniform(self.lo + 0.05, self.hi - 0.05) for _ in range(n_rand)]
        res = [self.f0.ik(sd, T)[:2] for sd in seeds]; best = min(pe for _, pe in res)
        cand = [(np.abs(q - q_prev).max(), q, pe) for q, pe in res if pe <= best + tol]
        _, q, pe = min(cand, key=lambda x: x[0]); return q, pe

    def _bridge(self, T, Q, PE):
        """[2026-10-23] offline bidirectional recovery bridge on a forward sequential path (in place on copies) -> Q, PE, bridge mask, log"""
        a = self.a; n = len(T); Q = Q.copy(); PE = PE.copy(); B = np.zeros(n, bool); log = []
        thr = a.reseed_mm / 1000; N = a.reseed_n; W = int(round(a.bridge_window_s / 0.05)); vmax = np.radians(a.ik_step); rng = np.random.default_rng(0)
        i = 0
        while i < n:
            bad = PE[i:] > thr; run = 0; o = None
            for k, b in enumerate(bad):
                run = run + 1 if b else 0
                if run >= N: o = i + k - N + 1; break
            if o is None: break
            d = o + N - 1
            qs, pes = self._multistart_pref(Q[d - 1] if d > 0 else self.qr, T[d], rng, a.seeds)
            if pes > thr: log.append(dict(onset=int(o), detect=int(d), ok=False, why="no reachable seed")); i = d + 1; continue
            lo_ = max(o - W, 0)
            qb = {d: qs}; peb = {d: pes}
            for t in range(d - 1, lo_ - 1, -1):                                  # backward bounded pass from the new branch
                q_, pe_, _ = self.free.ik(qb[t + 1], T[t]); qb[t] = q_; peb[t] = pe_
            best = None
            for tf in range(lo_, d):
                for tb in range(tf + 1, d + 1):
                    gap = np.abs(qb[tb] - Q[tf]).max(); L = tb - tf
                    if gap * 1.875 / L > vmax: continue                          # min-jerk peak speed = 1.875 x mean
                    cost = (L, gap)
                    if best is None or cost < best[0]: best = (cost, tf, tb)
                    break                                                         # shortest feasible tb for this tf
            if best is None: log.append(dict(onset=int(o), detect=int(d), ok=False, why="no feasible bridge")); i = d + 1; continue
            _, tf, tb = best; Lb = tb - tf; qa, qz = Q[tf].copy(), qb[tb].copy(); prev = qa
            for j, t in enumerate(range(tf + 1, tb)):
                u = (j + 1) / Lb; u = 10 * u**3 - 15 * u**4 + 6 * u**5; qi = qa + u * (qz - qa)
                qr_, pe_, _ = self.fine.ik(qi, T[t])                              # small refine toward the Cartesian target
                step = np.clip(qr_ - prev, -vmax, vmax); qn = prev + step
                Q[t] = qn; PE[t] = pose_err(self.pk.fk(qn), T[t]); B[t] = True; prev = qn
            for t in range(tb, d + 1): Q[t] = qb[t]; PE[t] = peb[t]
            q = qs                                                                # continue forward from the new branch
            for t in range(d + 1, n):
                q, pe_, _ = self.free.ik(q, T[t]); Q[t] = q; PE[t] = pe_
            seg = Q[max(tf - 1, 0): min(tb + 2, n)]; dq = np.degrees(np.abs(np.diff(seg, axis=0))).max() if len(seg) > 1 else 0.0
            log.append(dict(onset=int(o), detect=int(d), t_f=int(tf), t_b=int(tb), ok=True, bridge_s=round(Lb * 0.05, 2), shift_s=round((o - tf) * 0.05, 2),
                            max_dq_deg=round(float(dq), 2), seed_err_mm=round(pes * 1000, 1)))
            i = d + 1
        return Q, PE, B, log

    def _minjerk(self, Q, PE, B, T, tf, tb, qa, qz):
        """min-jerk joint ramp qa (at tf) -> qz (at tb), refined toward T and clipped to --ik-step; writes Q/PE/B on (tf, tb)"""
        vmax = np.radians(self.a.ik_step); Lb = tb - tf; prev = qa
        for j, t in enumerate(range(tf + 1, tb)):
            u = (j + 1) / Lb; u = 10 * u**3 - 15 * u**4 + 6 * u**5; qi = qa + u * (qz - qa)
            qr_, _, _ = self.fine.ik(qi, T[t]); qn = prev + np.clip(qr_ - prev, -vmax, vmax)
            Q[t] = qn; PE[t] = pose_err(self.pk.fk(qn), T[t]); B[t] = True; prev = qn

    def _join(self, A, Bq, ta_lo, ta_hi, tb_hi, vmax):
        """first feasible min-jerk join from path A at tf in [ta_lo, ta_hi) to path Bq at tb in (tf, tb_hi]: min (length, gap)"""
        best = None
        for tf in range(ta_lo, ta_hi):
            for tb in range(tf + 1, tb_hi + 1):
                if tb not in Bq or tf not in A: continue
                gap = np.abs(Bq[tb] - A[tf]).max(); L = tb - tf
                if gap * 1.875 / L <= vmax:
                    if best is None or (L, gap) < best[0]: best = ((L, gap), tf, tb)
                    break
        return best

    def _bridge2(self, T, Q, PE):
        """[2026-10-23] detour bridge (bridge v2): for every lock-out run of the forward path (>= --reseed-n frames > --reseed-mm), solve a
        fresh branch at detection (multi-start, nearest to the old pose), track it backward (bridge-in) and forward (bridge-out), join the old
        path with min-jerk ramps (<= --ik-step) and REJOIN the old path once it has recovered, so clean spans outside detours are unchanged.
        Join windows: --bridge-window-s before the onset, --rejoin-window-s after the recovery.  No rejoin -> the new branch is kept."""
        a = self.a; n = len(T); Q = Q.copy(); PE = PE.copy(); B = np.zeros(n, bool); log = []
        thr = a.reseed_mm / 1000; ok_thr = 0.030; N = a.reseed_n; Wi = int(round(a.bridge_window_s / 0.05)); Wo = int(round(a.rejoin_window_s / 0.05))
        vmax = np.radians(a.ik_step); rng = np.random.default_rng(0); i = 0
        while i < n:
            bad = PE[i:] > thr; run = 0; o = None
            for k, b in enumerate(bad):
                run = run + 1 if b else 0
                if run >= N: o = i + k - N + 1; break
            if o is None: break
            d = o + N - 1
            e = d + 1                                                             # old path recovers: first frame >= d with N clean frames
            while e < n and not (PE[e: e + N] <= ok_thr).all(): e += 1
            e = min(e, n)
            qs, pes = self._multistart_pref(Q[d - 1] if d > 0 else self.qr, T[d], rng, a.seeds)
            if pes > thr: log.append(dict(onset=int(o), detect=int(d), recover=int(e), ok=False, why="no reachable seed")); i = d + 1; continue
            lo_ = max(o - Wi, 0); qb = {d: qs}; peb = {d: pes}
            for t in range(d - 1, lo_ - 1, -1):
                q_, pe_, _ = self.free.ik(qb[t + 1], T[t]); qb[t] = q_; peb[t] = pe_
            hi_ = min(e + Wo, n - 1); qf = {d: qs}; pef = {d: pes}
            for t in range(d + 1, hi_ + 1):
                q_, pe_, _ = self.free.ik(qf[t - 1], T[t]); qf[t] = q_; pef[t] = pe_
            alt_bad = float(np.mean([pef[t] > thr for t in range(d, e)])); old_bad = float(np.mean(PE[d:e] > thr))
            if alt_bad >= old_bad: log.append(dict(onset=int(o), detect=int(d), recover=int(e), ok=False, why="new branch not better")); i = e; continue
            bi = self._join({t: Q[t] for t in range(lo_, d)}, qb, lo_, d, d, vmax)
            if bi is None: log.append(dict(onset=int(o), detect=int(d), recover=int(e), ok=False, why="no feasible bridge-in")); i = d + 1; continue
            _, tf, tb = bi
            old = {t: Q[t].copy() for t in range(e, n)}
            bo = self._join(qf, old, e if e < n else hi_, hi_ + 1 if e < n else hi_, min(hi_ + Wo, n - 1), vmax) if e < n else None
            self._minjerk(Q, PE, B, T, tf, tb, Q[tf].copy(), qb[tb].copy())
            for t in range(tb, d + 1): Q[t] = qb[t]; PE[t] = peb[t]
            if bo is not None:
                _, tf2, tb2 = bo
                for t in range(d + 1, tf2 + 1): Q[t] = qf[t]; PE[t] = pef[t]
                self._minjerk(Q, PE, B, T, tf2, tb2, qf[tf2].copy(), old[tb2].copy())
                B[tb:tf2 + 1] = True; nxt = tb2
            else:                                                                 # keep the new branch to the end
                q = qs
                for t in range(d + 1, n):
                    q, pe_, _ = self.free.ik(q, T[t]); Q[t] = q; PE[t] = pe_
                B[tb:n] = True; tf2 = tb2 = None; nxt = d + 1
            log.append(dict(onset=int(o), detect=int(d), recover=int(e), t_f=int(tf), t_b=int(tb), ok=True, rejoin=bo is not None,
                            t_f2=None if tf2 is None else int(tf2), t_b2=None if tb2 is None else int(tb2), bridge_s=round((tb - tf) * 0.05, 2),
                            shift_s=round((o - tf) * 0.05, 2), seed_err_mm=round(pes * 1000, 1)))
            i = max(nxt, d + 1)
        return Q, PE, B, log

    def solve(self, Th, G):
        """IK over targets Th (N,4,4) -> Q, pos err (m), flip, reseed mask.  --ik-mode (10-23 lock-out audit; default seq = production):
        seq     sequential, --ik-step bound from the previous pose (umi_replay_bi.solve_path behaviour)
        reseed  seq, but after --reseed-n consecutive frames with error > --reseed-mm the bound is dropped for that frame and it is solved
                fresh from ready + previous + --seeds random seeds (best kept), then bounded again
        indep   every frame independently, multi-start, no continuity (oracle reachability diagnostic)"""
        a = self.a; Rz180 = np.diag([-1.0, -1.0, 1.0]); j6_hi = np.radians(a.roll_thr); best = None; mode = getattr(a, "ik_mode", "seq")
        rng = np.random.default_rng(0)
        for flip in ((False, True) if mode != "indep" or getattr(a, "indep_flip", None) is None else (bool(a.indep_flip),)):
            q = self.qr; Q, PE, RS, phi, bad = [], [], [], 0.0, 0; PHI = []                 # 10-26: per-frame wrist-roll yield (runtime correction)
            for i in range(len(Th)):
                T = Th[i].copy(); T[:3, :3] = T[:3, :3] @ (Rz180 if flip else np.eye(3))
                c_, s_ = np.cos(phi), np.sin(phi); T[:3, :3] = T[:3, :3] @ np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
                rs = False
                if mode == "indep":
                    q, pe = self._multistart(q, T, rng, a.seeds)
                else:
                    q_try, pe, _ = (self.f0 if i == 0 else self.free).ik(q, T)
                    bad = bad + 1 if pe > a.reseed_mm / 1000 else 0
                    if mode == "reseed" and bad >= a.reseed_n:
                        q2, pe2 = self._multistart(q, T, rng, a.seeds)
                        if pe2 < pe: q_try, pe, rs = q2, pe2, True
                        bad = 0
                    q = q_try
                over = abs(q[5]) - j6_hi
                if mode != "indep":
                    if over > 0: phi -= np.sign(q[5]) * over
                    elif phi != 0: phi -= np.sign(phi) * min(abs(phi), np.radians(a.roll_relax), max(0.0, -over))
                Q.append(q); PE.append(pe); RS.append(rs); PHI.append(phi)
            Q, PE = np.array(Q), np.array(PE); BR, LOG = np.zeros(len(Q), bool), []
            if mode in ("bridge", "bridge2"):
                Tl = []
                for i in range(len(Th)):                                           # same targets as the forward pass (no roll yield in the bridge)
                    T = Th[i].copy(); T[:3, :3] = T[:3, :3] @ (Rz180 if flip else np.eye(3)); Tl.append(T)
                Q, PE, BR, LOG = (self._bridge if mode == "bridge" else self._bridge2)(Tl, Q, PE)
            score = (float(np.mean(PE > 0.01)), float(PE.max()))
            if best is None or score < best[0]: best = (score, Q, PE, flip, np.array(RS), BR, LOG, np.degrees(np.array(PHI)))
        self.last_reseed = best[4]; self.last_bridge = best[5]; self.last_log = best[6]; self.last_phi = best[7]
        return best[1], best[2], best[3]

    def links(self, q6):
        self.pin.framesForwardKinematics(self.pk.model, self.pk.data, self.pk._q(q6)); return np.array([self.pk.data.oMf[i].translation for i in self.fids])


def pose_err(F, T):
    return float(np.linalg.norm(F[:3, 3] - T[:3, 3]))


def bump(n, s, e, ramp):
    w = np.zeros(n); w[s:e] = 1.0
    for k in range(1, ramp + 1):
        v = 0.5 * (1 + np.cos(np.pi * k / (ramp + 1)))
        if s - k >= 0: w[s - k] = max(w[s - k], v)
        if e - 1 + k < n: w[e - 1 + k] = max(w[e - 1 + k], v)
    return w


def run_demo(args):
    # 11-01: one demo with NaN positions (A' fusion gap, take 152254 demo 2: SVD did not converge) killed the whole pool -> per-demo failure
    try: return _run_demo(args)
    except Exception as e: return str(Path(args[1]) / "ik_v1" / f"demo{args[2]:02d}.npz"), f"FAILED {type(e).__name__}: {e}"


def _run_demo(args):
    take, dtake, di, a = args
    from scipy.spatial.transform import Rotation as Rot
    take, dtake = Path(take), Path(dtake); asg = json.loads((dtake / "assignment.json").read_text()); d = demos_of(take)[di]
    out = dtake / "ik_v1" / f"demo{di:02d}.npz"
    if out.exists() and not a.force: return str(out), "exists"
    Z = {s: np.load(dtake / f"umi_traj_{s}.npz", allow_pickle=True) for s in ("left", "right")}
    lo = max(d["t0_ns"], *(int(z["t_ns"][0]) for z in Z.values())); hi = min(d["t1_ns"], *(int(z["t_ns"][-1]) for z in Z.values()))
    if hi - lo < 5e9: return str(out), "short"
    g_ns = lo + (np.arange(int((hi - lo) / 1e9 / DT) + 1) * DT * 1e9).astype(np.int64); n = len(g_ns)
    arms, Th, G, GW, P0, R0 = {}, {}, {}, {}, {}, {}
    for s in ("left", "right"):
        z = Z[s]; rig = str(z["source_rig_label"]); R_UE = np.array(json.loads(ALIGN[rig].read_text())["R_U_E"]); t = z["t_ns"].astype(np.int64)
        P = np.stack([np.interp(g_ns, t, z["p_B"][:, j]) for j in range(3)], 1)
        R = z["R_B_tcp"] @ R_UE; rv = Rot.from_matrix(R); k = np.clip(np.searchsorted(t, g_ns), 1, len(t) - 1)
        # SLERP-free: nearest + rotation-vector smoothing relative to the first frame (small steps at 15 -> 20 Hz)
        Rg = rv[np.where(np.abs(t[k - 1] - g_ns) <= np.abs(t[k] - g_ns), k - 1, k)]
        Rref = Rg[0]; rvec = (Rref.inv() * Rg).as_rotvec(); rvec = np.unwrap(rvec, axis=0) if False else rvec
        P = smooth(P, a.smooth); Rs = Rref * Rot.from_rotvec(smooth(rvec, a.smooth))
        J = np.load(take / asg["cam_of_arm"][s] / "umi_jaw.npz"); gs = J["state"][np.clip(np.searchsorted(J["t_ns"], g_ns, side="right") - 1, 0, len(J["t_ns"]) - 1)].astype(float)
        T = np.tile(np.eye(4), (n, 1, 1)); T[:, :3, :3] = Rs.as_matrix(); T[:, :3, 3] = P
        arms[s] = Arm(s, a)
        for i in range(n):
            T[i] = arms[s].level(T[i], gs[i]); T[i, 2, 3] += arms[s].corner_lift(T[i], gs[i])      # tilt first (keeps height), lift the rest
        Th[s], G[s] = T, gs; P0[s] = P.copy(); R0[s] = Rs.as_matrix()
        ce = np.flatnonzero(np.diff((gs > 0.5).astype(int)) < 0) + 1; gw = np.zeros(n, bool)
        for c in ce: gw[max(c - 10, 0): c + 11] = True
        GW[s] = gw
    tgt = {s: Th[s].copy() for s in Th}; sol = {s: arms[s].solve(tgt[s], G[s]) for s in arms}; hist = []
    for rnd in range(a.rounds + 1):
        Qs = {s: smooth(sol[s][0], a.q_smooth) for s in sol}
        L = {s: np.array([arms[s].links(q) for q in Qs[s]]) for s in arms}
        gap = np.linalg.norm(L["left"][:, :, None] - L["right"][:, None], axis=3).min(axis=(1, 2))
        viol = gap < a.arm_gap; hist.append(dict(round=rnd, viol_frames=int(viol.sum()), min_gap_mm=round(float(gap.min()) * 1000, 1)))
        if not viol.any() or rnd == a.rounds: break
        x = np.r_[0, viol.astype(int), 0]; ss, ee = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1); moved = set()
        for s0, e0 in zip(ss, ee):
            deficit = a.arm_gap - float(gap[s0:e0].min()) + 0.01
            sup = slice(max(s0 - a.ramp, 0), min(e0 + a.ramp, n))          # grasp priority over the WHOLE bump support (ramps included)
            gl, gr = GW["left"][sup].sum(), GW["right"][sup].sum()
            share = {"left": 0.5, "right": 0.5} if (gl > 0 and gr > 0) else ({"left": 0.0, "right": 1.0} if gl > gr else {"left": 1.0, "right": 0.0} if gr > gl else
                     ({"left": 1.0, "right": 0.0} if abs(tgt["left"][s0:e0, 1, 3].mean()) < abs(tgt["right"][s0:e0, 1, 3].mean()) else {"left": 0.0, "right": 1.0}))
            w = bump(n, s0, e0, a.ramp)
            for s, f in share.items():
                if f > 0: tgt[s][:, 1, 3] += (1 if s == "left" else -1) * f * deficit * w; moved.add(s)
        for s in moved: sol[s] = arms[s].solve(tgt[s], G[s])
    res = dict(t_ns=g_ns, gap=gap, rounds=json.dumps(hist), mode="ik_v1", args=json.dumps({k: v for k, v in vars(a).items() if k not in ("take", "derived")}))
    for s in arms:
        Q = Qs[s]; F = np.array([arms[s].pk.fk(q) for q in Q]); err = np.linalg.norm(F[:, :3, 3] - tgt[s][:, :3, 3], axis=1)
        lim = np.any((Q < arms[s].lo + np.radians(1)) | (Q > arms[s].hi - np.radians(1)), axis=1)
        res.update({f"{s}_p_h": P0[s], f"{s}_R_h": R0[s], f"{s}_p": F[:, :3, 3], f"{s}_R": F[:, :3, :3], f"{s}_q": Q, f"{s}_ik_mm": err * 1000,
                    f"{s}_lim": lim, f"{s}_grasp": GW[s], f"{s}_flip": sol[s][2], f"{s}_g": G[s],
                    f"{s}_q_raw": sol[s][0], f"{s}_reseed": getattr(arms[s], "last_reseed", np.zeros(n, bool)),
                    f"{s}_bridge": getattr(arms[s], "last_bridge", np.zeros(n, bool)), f"{s}_bridge_log": json.dumps(getattr(arms[s], "last_log", []))})
        # 10-26 label contract: Cartesian labels before IK + IK as a feasibility QA (flip / roll yield / gap push = runtime corrections)
        Rz = np.diag([-1.0, -1.0, 1.0]); Rr = Th[s][:, :3, :3]; RF = F[:, :3, :3]
        ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(A, 1, 2) @ B, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        res.update({f"{s}_p_fused": P0[s], f"{s}_R_fused": R0[s],                                   # action_cartesian_fused (= _p_h / _R_h)
                    f"{s}_p_rob": Th[s][:, :3, 3], f"{s}_R_rob": Rr,                                # action_cartesian_robotized (POLICY TARGET)
                    f"{s}_p_iktgt": tgt[s][:, :3, 3], f"{s}_gap_push_mm": np.linalg.norm(tgt[s][:, :3, 3] - Th[s][:, :3, 3], axis=1) * 1000,
                    f"{s}_err_pos_rob_mm": np.linalg.norm(F[:, :3, 3] - Th[s][:, :3, 3], axis=1) * 1000,
                    f"{s}_err_rot_rob_deg": np.minimum(ang(Rr, RF), ang(Rr @ Rz, RF)),             # jaw 180 deg symmetric
                    f"{s}_roll_relax_deg": getattr(arms[s], "last_phi", np.zeros(n)),
                    f"{s}_lim_margin_deg": np.degrees(np.minimum(Q - arms[s].lo, arms[s].hi - Q).min(axis=1)),
                    f"{s}_ik_valid": err < 0.01})
    out.parent.mkdir(parents=True, exist_ok=True); np.savez(out, **res)
    dg = {s: np.linalg.norm(res[f"{s}_p"] - P0[s], axis=1) * 1000 for s in arms}
    return str(out), (f"min gap {gap.min() * 1000:.0f} mm, rounds {len(hist) - 1}, " + ", ".join(
        f"{s}: grasp disp max {dg[s][GW[s]].max() if GW[s].any() else 0:.0f} mm, free p50 {np.median(dg[s][~GW[s]]):.0f} mm, ik>10mm {np.mean(res[f'{s}_ik_mm'] > 10) * 100:.0f}%, lim {np.mean(res[f'{s}_lim']) * 100:.0f}%" for s in arms))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("derived"); ap.add_argument("--demos", default=""); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arm-gap", type=float, default=0.12); ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=5); ap.add_argument("--q-smooth", type=int, default=5); ap.add_argument("--ik-step", type=float, default=7.5)
    ap.add_argument("--ori-cost", type=float, default=0.1); ap.add_argument("--roll-thr", type=float, default=115.0); ap.add_argument("--roll-relax", type=float, default=1.5)
    ap.add_argument("--table-z", type=float, default=0.003); ap.add_argument("--tip-clear", type=float, default=0.005); ap.add_argument("--tip-clear-left-extra", type=float, default=0.005)
    ap.add_argument("--ik-mode", default="seq", choices=["seq", "reseed", "indep", "bridge", "bridge2"], help="seq = production; reseed / indep / bridge / bridge2 = 10-23 lock-out audit")
    ap.add_argument("--reseed-n", type=int, default=5); ap.add_argument("--reseed-mm", type=float, default=50.0); ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--bridge-window-s", type=float, default=1.0)
    ap.add_argument("--rejoin-window-s", type=float, default=2.0, help="bridge2: how far after the old path recovers the detour may rejoin it")
    ap.add_argument("--jaw-half", type=float, default=0.020); ap.add_argument("--level-max", type=float, default=0.0, help="10-22 test 123614 demo01: 15-30 deg tilt made right-arm grasp IK fail (39-103 mm) -> off; table grasps get the 26-33 mm lift instead"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    take = Path(a.take).expanduser().resolve(); dtake = Path(a.derived).expanduser().resolve()
    idx = [int(x) for x in a.demos.split(",") if x.strip()] if a.demos else [d["idx"] for d in demos_of(take)]
    jobs = [(str(take), str(dtake), i, a) for i in idx]; t0 = time.time()
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for o, msg in pool.imap_unordered(run_demo, jobs): print(f"{Path(o).name}: {msg}", flush=True)
    else:
        for j in jobs: o, msg = run_demo(j); print(f"{Path(o).name}: {msg}", flush=True)
    print(f"done {take.name} {len(jobs)} demos in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
