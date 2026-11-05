"""[2026-10-22] Robotization mode "ik_v1" (user option B): NO midline y-compression (dataset stage, IK labels). The human
canonical TCP is kept wherever the reBot can follow it; only short offending spans are warped, smoothly, and grasp windows have
position priority.

Per demo window and per arm (arm-labelled inputs in data/tag_rig_derived/<take><src>/umi_traj_<arm>.npz, see arm_assignment.py):
  human target   T_h = [R_B_tcp @ R_U_E(rig) | p_B] on the 20 Hz grid (5-frame position / rotation-vector smoothing); where the
                 finger corner would hit the table the tool is first tilted toward vertical (umi_replay_bi.level, <= --level-max
                 30 deg, keeps the human TCP height), any remainder is lifted (cornerLift, --tip-clear 0.005, +0.005 left)
  IK             umi_replay_bi's PinkIkSolver with the BRIDGE joint limits (0.5 deg inside), joint offset / base / TCP overrides,
                 orientation_cost 0.1 (position first), 7.5 deg/frame bound (= its 5 deg at 30 Hz), first frame unbounded,
                 wrist-roll yield above |j6| 115 deg, tool-axis 180 deg flip chosen per arm (fewer > 10 mm frames)
                 (robot_arm.Arm)
  inter-arm gap  umi_replay_bi's link-origin set (link3-6, gripper_link, fingers, tcp) of both arms; min distance >= --arm-gap
  (0.12
                 m). Each violating span is fixed by pushing ONE arm's target away from the other along base y with a
                 raised-cosine bump (flat on the span, 0.5 s ramps). The arm with fewer grasp-window frames in the span moves; if
                 both grasp, both move half. Up to --rounds re-solves.
  label          robot-feasible TCP = FK(q) of the (5-frame smoothed) joint path, in reBot EEF axes (same frame as R_B_tcp @
  R_U_E) 10-26 contract (extra keys, old keys unchanged): <arm>_p/R_fused (fusion + axes + 0.25 s smoothing), <arm>_p/R_rob = +
  table
                 clearance = action_cartesian_robotized (POLICY TARGET, before IK; tool flip / wrist-roll yield / arm-gap push are
                 runtime corrections and NOT in it), <arm>_p_iktgt (+ gap push), IK QA: err_pos_rob_mm, err_rot_rob_deg
                 (jaw-symmetric), roll_relax_deg, lim_margin_deg, gap_push_mm, ik_valid (< 10 mm vs the IK target);
                 <arm>_p/R = action_cartesian_ik_fk (old label, comparison)
Outputs <derived>/ik_v1/demoNN.npz: t_ns, per arm {p_h, R_h, p, R, q, ik_mm, lim (any joint within 1 deg of a limit), grasp},
gap (min link distance per frame, m), meta json. Run with the IK interpreter (laundry_common.paths.IK_PYTHON: pinocchio + pink).
usage: robotize_ik.py <take_dir> <derived_take_dir> [--demos 0,1] [--workers 6]
    python -m laundry_dataset.robotize_ik TAKE DERIVED_TAKE [--demos 0,1] [--workers 6]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import calibration
from .robot_arm import DT_S, RZ180, Arm
from .takes import demoWindows

SIDES = ("left", "right")
MIN_WINDOW_NS = 5e9                  # demo window inside the recording shorter than 5 s -> "short"
NS_PER_S = 1e9
JAW_OPEN = 0.5                       # jaw state above this = open
GRASP_HALF_FRAMES = 10               # grasp window = +-0.5 s (10 frames) around each jaw close
GAP_PUSH_MARGIN_M = 0.01             # each arm-gap push clears the violation by 10 mm more than needed
LIMIT_NEAR_DEG = 1.0                 # `lim`: some joint within 1 deg of its bridge limit
IK_REPORT_MM = 10                    # run summary: share of frames with an IK error above this
PCT = 100
HALF = 0.5                           # both arms grasp -> each moves half the deficit
IK_VALID_M = 0.01                    # ik_valid: FK within 10 mm of the IK target
MM_PER_M = 1000
CLI_ONLY_ARGS = ("take", "derived")  # parser entries not stored in the npz `args`


def smooth(X, k):
    """centred k-frame box filter per column, edge-padded"""
    if k <= 1:
        return X
    tW = np.ones(k) / k
    return np.stack([np.convolve(np.pad(X[:, tJ], k // 2, mode="edge"), tW, "valid")[:len(X)] for tJ in range(X.shape[1])], 1)


def bump(n, s, e, ramp):
    """raised-cosine bump: 1 on [s, e), cosine ramps of `ramp` frames on both sides, 0 elsewhere"""
    tW = np.zeros(n)
    tW[s:e] = 1.0
    for tK in range(1, ramp + 1):
        tV = 0.5 * (1 + np.cos(np.pi * tK / (ramp + 1)))
        if s - tK >= 0:
            tW[s - tK] = max(tW[s - tK], tV)
        if e - 1 + tK < n:
            tW[e - 1 + tK] = max(tW[e - 1 + tK], tV)
    return tW


def _humanTarget(z, gridNs, args):
    """one arm on the grid -> (P, Rs) smoothed human TCP and the jaw state; SLERP-free: nearest + rotation-vector smoothing
    relative to the first frame (small steps at 15 -> 20 Hz)"""
    from scipy.spatial.transform import Rotation as Rot
    tRig = str(z["source_rig_label"])
    tRue = calibration.rigToEef(tRig)
    tT = z["t_ns"].astype(np.int64)
    tP = np.stack([np.interp(gridNs, tT, z["p_B"][:, tJ]) for tJ in range(3)], 1)
    tRv = Rot.from_matrix(z["R_B_tcp"] @ tRue)
    tK = np.clip(np.searchsorted(tT, gridNs), 1, len(tT) - 1)
    tRg = tRv[np.where(np.abs(tT[tK - 1] - gridNs) <= np.abs(tT[tK] - gridNs), tK - 1, tK)]
    tRref = tRg[0]
    tRvec = (tRref.inv() * tRg).as_rotvec()
    tP = smooth(tP, args.smooth)
    tRs = tRref * Rot.from_rotvec(smooth(tRvec, args.smooth))
    return tP, tRs


def _jawState(take, asg, side, gridNs):
    """binary jaw state of an arm's wrist camera, last sample at or before each grid time"""
    tJaw = np.load(take / asg["cam_of_arm"][side] / "umi_jaw.npz")
    return tJaw["state"][np.clip(np.searchsorted(tJaw["t_ns"], gridNs, side="right") - 1, 0, len(tJaw["t_ns"]) - 1)].astype(float)


def _graspWindow(g, n):
    """+-0.5 s around every jaw close"""
    tCloses = np.flatnonzero(np.diff((g > JAW_OPEN).astype(int)) < 0) + 1
    tWin = np.zeros(n, bool)
    for tC in tCloses:
        tWin[max(tC - GRASP_HALF_FRAMES, 0): tC + GRASP_HALF_FRAMES + 1] = True
    return tWin


def _pushShare(gl, gr, tgt, s0, e0):
    """which arm moves for one violating span: fewer grasp frames moves, both grasp -> half each, tie -> the arm nearer the
    midline"""
    if gl > 0 and gr > 0:
        return {"left": HALF, "right": HALF}
    if gl > gr:
        return {"left": 0.0, "right": 1.0}
    if gr > gl:
        return {"left": 1.0, "right": 0.0}
    if abs(tgt["left"][s0:e0, 1, 3].mean()) < abs(tgt["right"][s0:e0, 1, 3].mean()):
        return {"left": 1.0, "right": 0.0}
    return {"left": 0.0, "right": 1.0}


def _solveWithGap(arms, Th, G, GW, args, n):
    """solve both arms, then push targets apart along base y until the link gap >= --arm-gap (<= --rounds re-solves)
    -> (IK targets, solutions, smoothed joint paths, gap per frame, round history)"""
    tTgt = {tS: Th[tS].copy() for tS in Th}
    tSol = {tS: arms[tS].solve(tTgt[tS], G[tS]) for tS in arms}
    tHist = []
    for tRound in range(args.rounds + 1):
        tQs = {tS: smooth(tSol[tS][0], args.q_smooth) for tS in tSol}
        tLinks = {tS: np.array([arms[tS].links(tQ) for tQ in tQs[tS]]) for tS in arms}
        tGap = np.linalg.norm(tLinks["left"][:, :, None] - tLinks["right"][:, None], axis=3).min(axis=(1, 2))
        tViol = tGap < args.arm_gap
        tHist.append(dict(round=tRound, viol_frames=int(tViol.sum()), min_gap_mm=round(float(tGap.min()) * MM_PER_M, 1)))
        if not tViol.any() or tRound == args.rounds:
            break
        tX = np.r_[0, tViol.astype(int), 0]
        tStarts, tEnds = np.flatnonzero(np.diff(tX) == 1), np.flatnonzero(np.diff(tX) == -1)
        tMoved = set()
        for tS0, tE0 in zip(tStarts, tEnds):
            tDeficit = args.arm_gap - float(tGap[tS0:tE0].min()) + GAP_PUSH_MARGIN_M
            # grasp priority over the WHOLE bump support (ramps included)
            tSup = slice(max(tS0 - args.ramp, 0), min(tE0 + args.ramp, n))
            tShare = _pushShare(GW["left"][tSup].sum(), GW["right"][tSup].sum(), tTgt, tS0, tE0)
            tW = bump(n, tS0, tE0, args.ramp)
            for tS, tF in tShare.items():
                if tF > 0:
                    tTgt[tS][:, 1, 3] += (1 if tS == "left" else -1) * tF * tDeficit * tW
                    tMoved.add(tS)
        for tS in SIDES:
            if tS in tMoved:
                tSol[tS] = arms[tS].solve(tTgt[tS], G[tS])
    return tTgt, tSol, tQs, tGap, tHist


def _rotAngleDeg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(A, 1, 2) @ B, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def runDemo(job):
    """pool entry: job = (take, derived_take, demo index, args) -> (out path, message). 11-01: one demo with NaN positions (A'
    fusion gap, take 152254 demo 2: SVD did not converge) killed the whole pool -> per-demo failure (was run_demo)"""
    try:
        return _runDemo(job)
    except Exception as tE:
        return str(Path(job[1]) / "ik_v1" / f"demo{job[2]:02d}.npz"), f"FAILED {type(tE).__name__}: {tE}"


def _runDemo(job):
    tTake, tDtake, tDi, tArgs = job
    tTake, tDtake = Path(tTake), Path(tDtake)
    tAsg = json.loads((tDtake / "assignment.json").read_text())
    tDemo = demoWindows(tTake)[tDi]
    tOut = tDtake / "ik_v1" / f"demo{tDi:02d}.npz"
    if tOut.exists() and not tArgs.force:
        return str(tOut), "exists"
    tZ = {tS: np.load(tDtake / f"umi_traj_{tS}.npz", allow_pickle=True) for tS in SIDES}
    tLo = max(tDemo.mT0Ns, *(int(tTraj["t_ns"][0]) for tTraj in tZ.values()))
    tHi = min(tDemo.mT1Ns, *(int(tTraj["t_ns"][-1]) for tTraj in tZ.values()))
    if tHi - tLo < MIN_WINDOW_NS:
        return str(tOut), "short"
    tGrid = tLo + (np.arange(int((tHi - tLo) / NS_PER_S / DT_S) + 1) * DT_S * NS_PER_S).astype(np.int64)
    tN = len(tGrid)
    tArms, tTh, tGrip, tGW, tP0, tR0 = {}, {}, {}, {}, {}, {}
    for tS in SIDES:
        tP, tRs = _humanTarget(tZ[tS], tGrid, tArgs)
        tG = _jawState(tTake, tAsg, tS, tGrid)
        tT = np.tile(np.eye(4), (tN, 1, 1))
        tT[:, :3, :3] = tRs.as_matrix()
        tT[:, :3, 3] = tP
        tArms[tS] = Arm(tS, tArgs)
        for tI in range(tN):
            # tilt first (keeps height), lift the rest
            tT[tI] = tArms[tS].level(tT[tI], tG[tI])
            tT[tI, 2, 3] += tArms[tS].cornerLift(tT[tI], tG[tI])
        tTh[tS], tGrip[tS] = tT, tG
        tP0[tS] = tP.copy()
        tR0[tS] = tRs.as_matrix()
        tGW[tS] = _graspWindow(tG, tN)
    tTgt, tSol, tQs, tGap, tHist = _solveWithGap(tArms, tTh, tGrip, tGW, tArgs, tN)
    tRes = dict(t_ns=tGrid, gap=tGap, rounds=json.dumps(tHist), mode="ik_v1",
                args=json.dumps({tK: tV for tK, tV in vars(tArgs).items() if tK not in CLI_ONLY_ARGS}))
    for tS in tArms:
        tArm = tArms[tS]
        tQ = tQs[tS]
        tF = np.array([tArm.mPk.fk(tQRow) for tQRow in tQ])
        tErr = np.linalg.norm(tF[:, :3, 3] - tTgt[tS][:, :3, 3], axis=1)
        tLim = np.any((tQ < tArm.mLo + np.radians(LIMIT_NEAR_DEG)) | (tQ > tArm.mHi - np.radians(LIMIT_NEAR_DEG)), axis=1)
        tRes.update({f"{tS}_p_h": tP0[tS], f"{tS}_R_h": tR0[tS], f"{tS}_p": tF[:, :3, 3], f"{tS}_R": tF[:, :3, :3], f"{tS}_q": tQ,
                     f"{tS}_ik_mm": tErr * MM_PER_M,
                     f"{tS}_lim": tLim, f"{tS}_grasp": tGW[tS], f"{tS}_flip": tSol[tS][2], f"{tS}_g": tGrip[tS],
                     f"{tS}_q_raw": tSol[tS][0], f"{tS}_reseed": tArm.mLastReseed, f"{tS}_bridge": tArm.mLastBridge,
                     f"{tS}_bridge_log": json.dumps(tArm.mLastLog)})
        # 10-26 label contract: Cartesian labels before IK + IK as a feasibility QA (flip / roll yield / gap push = runtime
        # corrections)
        tRr = tTh[tS][:, :3, :3]
        tRf = tF[:, :3, :3]
        # action_cartesian_fused (= _p_h / _R_h)
        tRes.update({f"{tS}_p_fused": tP0[tS], f"{tS}_R_fused": tR0[tS],
                     # action_cartesian_robotized (POLICY TARGET)
                     f"{tS}_p_rob": tTh[tS][:, :3, 3], f"{tS}_R_rob": tRr,
                     f"{tS}_p_iktgt": tTgt[tS][:, :3, 3],
                     f"{tS}_gap_push_mm": np.linalg.norm(tTgt[tS][:, :3, 3] - tTh[tS][:, :3, 3], axis=1) * MM_PER_M,
                     f"{tS}_err_pos_rob_mm": np.linalg.norm(tF[:, :3, 3] - tTh[tS][:, :3, 3], axis=1) * MM_PER_M,
                     # jaw 180 deg symmetric
                     f"{tS}_err_rot_rob_deg": np.minimum(_rotAngleDeg(tRr, tRf), _rotAngleDeg(tRr @ RZ180, tRf)),
                     f"{tS}_roll_relax_deg": tArm.mLastPhi,
                     f"{tS}_lim_margin_deg": np.degrees(np.minimum(tQ - tArm.mLo, tArm.mHi - tQ).min(axis=1)),
                     f"{tS}_ik_valid": tErr < IK_VALID_M})
    tOut.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tOut, **tRes)
    tDisp = {tS: np.linalg.norm(tRes[f"{tS}_p"] - tP0[tS], axis=1) * MM_PER_M for tS in tArms}
    tArmMsgs = ", ".join(f"{tS}: grasp disp max {tDisp[tS][tGW[tS]].max() if tGW[tS].any() else 0:.0f} mm, "
                         f"free p50 {np.median(tDisp[tS][~tGW[tS]]):.0f} mm, "
                         f"ik>10mm {np.mean(tRes[f'{tS}_ik_mm'] > IK_REPORT_MM) * PCT:.0f}%, "
                         f"lim {np.mean(tRes[f'{tS}_lim']) * PCT:.0f}%" for tS in tArms)
    return str(tOut), f"min gap {tGap.min() * MM_PER_M:.0f} mm, rounds {len(tHist) - 1}, " + tArmMsgs


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take")
    tAp.add_argument("derived")
    tAp.add_argument("--demos", default="")
    tAp.add_argument("--workers", type=int, default=1)
    tAp.add_argument("--arm-gap", type=float, default=0.12)
    tAp.add_argument("--rounds", type=int, default=6)
    tAp.add_argument("--ramp", type=int, default=10)
    tAp.add_argument("--smooth", type=int, default=5)
    tAp.add_argument("--q-smooth", type=int, default=5)
    tAp.add_argument("--ik-step", type=float, default=7.5)
    tAp.add_argument("--ori-cost", type=float, default=0.1)
    tAp.add_argument("--roll-thr", type=float, default=115.0)
    tAp.add_argument("--roll-relax", type=float, default=1.5)
    tAp.add_argument("--table-z", type=float, default=0.003)
    tAp.add_argument("--tip-clear", type=float, default=0.005)
    tAp.add_argument("--tip-clear-left-extra", type=float, default=0.005)
    tAp.add_argument("--ik-mode", default="seq", choices=["seq", "reseed", "indep", "bridge", "bridge2"],
                     help="seq = production; reseed / indep / bridge / bridge2 = 10-23 lock-out audit")
    tAp.add_argument("--reseed-n", type=int, default=5)
    tAp.add_argument("--reseed-mm", type=float, default=50.0)
    tAp.add_argument("--seeds", type=int, default=8)
    tAp.add_argument("--bridge-window-s", type=float, default=1.0)
    tAp.add_argument("--rejoin-window-s", type=float, default=2.0,
                     help="bridge2: how far after the old path recovers the detour may rejoin it")
    tAp.add_argument("--jaw-half", type=float, default=0.020)
    tAp.add_argument("--level-max", type=float, default=0.0,
                     help="10-22 test 123614 demo01: 15-30 deg tilt made right-arm grasp IK fail (39-103 mm) -> off; "
                          "table grasps get the 26-33 mm lift instead")
    tAp.add_argument("--force", action="store_true")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tA = buildParser().parse_args(argv)
    tTake = Path(tA.take).expanduser().resolve()
    tDerived = Path(tA.derived).expanduser().resolve()
    tIdx = [int(tX) for tX in tA.demos.split(",") if tX.strip()] if tA.demos else [tD.mIdx for tD in demoWindows(tTake)]
    tJobs = [(str(tTake), str(tDerived), tI, tA) for tI in tIdx]
    tStart = time.time()
    if tA.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(tA.workers) as tPool:
            for tOutPath, tMsg in tPool.imap_unordered(runDemo, tJobs):
                print(f"{Path(tOutPath).name}: {tMsg}", flush=True)
    else:
        for tJob in tJobs:
            tOutPath, tMsg = runDemo(tJob)
            print(f"{Path(tOutPath).name}: {tMsg}", flush=True)
    print(f"done {tTake.name} {len(tJobs)} demos in {time.time() - tStart:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
