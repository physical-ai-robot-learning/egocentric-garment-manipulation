"""One reBot arm for offline robotization and for the deploy: Pink IK with the bridge joint limits, the finger-corner table floor,
tool levelling, sequential / re-seeded / bridged IK over a target path, and the link-origin set of the inter-arm gap (dataset
stage; was robotize_ik.Arm, 10-22 .. 10-26).

Arm(side, args) takes the robotize_ik argparse Namespace (or any object with the attributes it reads: ik_step, ori_cost, jaw_half,
table_z, tip_clear, tip_clear_left_extra, level_max, roll_thr, roll_relax, reseed_mm, reseed_n, seeds, bridge_window_s,
rejoin_window_s, and optionally ik_mode / indep_flip). Solvers (laundry_shared PinkIkSolver, umi_replay_bi settings):
  mFree   bounded: --ik-step deg per frame, 200 iterations, orientation_cost --ori-cost (position first)
  mF0     unbounded first-frame / multi-start solver: 180 deg, 400 iterations
  mFine   small refine toward a Cartesian target inside a min-jerk ramp: 2 deg, 60 iterations
  mPk     the configured solver (config ik.pink), used for FK
All three bounded solvers get the BRIDGE joint limits 0.5 deg inside (+ the per-joint offsets of the URDF override).
"""
from __future__ import annotations

import numpy as np

DT_S = 0.05                          # 20 Hz label grid
N_ARM_JOINTS = 14                    # both arms in the solver's joint vector
BRIDGE_LIMIT_MARGIN_DEG = 0.5        # solver limits = bridge limits, this far inside
FREE_ITERS = 200
F0_STEP_DEG, F0_ITERS = 180, 400     # effectively unbounded
FINE_STEP_DEG, FINE_ITERS = 2.0, 60
SEED_LIMIT_MARGIN_RAD = 0.05         # random multi-start seeds stay this far inside the bridge limits
GOOD_RESIDUAL_M = 1e-3               # multi-start stops at the first seed this close
PREFER_NEAR_TOL_M = 0.010            # multi-start: results within this of the best residual compete on joint distance
MIN_JERK_PEAK = 1.875                # peak speed of a min-jerk ramp = 1.875 x its mean speed
RECOVERED_M = 0.030                  # bridge2: the old path counts as recovered after reseed_n frames below this
LEVEL_BISECT_ITERS = 20
DEGENERATE_AXIS = 1e-6               # tool axis already (anti)parallel to straight down
FLIP_BAD_M = 0.01                    # tool-flip choice: fewer frames with an IK residual above this
# finger corner below the TCP: jaw half-width along the tool x axis, 8 mm + 28 mm x opening along the tool y axis
CORNER_Y_CLOSED_M, CORNER_Y_OPEN_M = 0.008, 0.028
STRAIGHT_DOWN = (0.0, 0.0, -1.0)
RZ180 = np.diag([-1.0, -1.0, 1.0])   # tool-axis 180 deg flip (the jaw is symmetric)
WRIST_ROLL_JOINT = 5                 # j6, the joint the wrist-roll yield watches
MM_PER_M = 1000
LINK_FRAMES = ("link3", "link4", "link5", "link6", "gripper_link", "gripper_left", "gripper_right", "tcp")


def poseErr(F, T) -> float:
    """position residual (m) of a solved pose F against the target T (was pose_err)"""
    return float(np.linalg.norm(F[:3, 3] - T[:3, 3]))


def _minJerk(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5


class Arm:
    def __init__(self, side, args):
        from laundry_shared.ego_teleop.robot.rebot_client import SIDE_SLICE
        from laundry_shared.visual_teleop import config as vcfg
        from laundry_shared.visual_teleop.robot.ik import ArmIk
        from laundry_shared.visual_teleop.robot.ik_pink import PinkIkSolver, PinkParams
        from laundry_shared.visual_teleop.tools.robot_moves import READY_Q_DEG
        import pinocchio as pin
        tCfg = vcfg.load()
        self.mSide, self.mArgs, self.mPin = side, args, pin
        tReadyDeg = np.zeros(N_ARM_JOINTS)
        for tSide in ("left", "right"):
            tReadyDeg[SIDE_SLICE[tSide]] = READY_Q_DEG[tSide]
        self.mQReady = np.radians(READY_Q_DEG[side])
        tPk = PinkIkSolver(side, PinkParams(**tCfg.ik["pink"]))
        tPk.sync(tReadyDeg)
        tMake = lambda step, iters: PinkIkSolver(side, PinkParams(max_joint_delta_deg=step, iters=iters,
                                                                  orientation_cost=args.ori_cost))
        self.mFree, self.mF0 = tMake(args.ik_step, FREE_ITERS), tMake(F0_STEP_DEG, F0_ITERS)
        self.mFine = tMake(FINE_STEP_DEG, FINE_ITERS)
        tLimits = ArmIk.from_cfg(tCfg, tPk).gate.cfg.limits
        self.mLo, self.mHi = np.asarray(tLimits.lower_rad), np.asarray(tLimits.upper_rad)
        for tSolver in (self.mFree, self.mF0, self.mFine):
            tSolver.sync(tReadyDeg)
            for tK, tIq in enumerate(tSolver.ridx):
                tOffset = tSolver.joff[side][tK]
                tSolver.model.lowerPositionLimit[tIq] = max(tSolver.model.lowerPositionLimit[tIq],
                                                            self.mLo[tK] + tOffset + np.radians(BRIDGE_LIMIT_MARGIN_DEG))
                tSolver.model.upperPositionLimit[tIq] = min(tSolver.model.upperPositionLimit[tIq],
                                                            self.mHi[tK] + tOffset - np.radians(BRIDGE_LIMIT_MARGIN_DEG))
        self.mPk = tPk
        tNames = tuple(f"{side}_{tN}" for tN in LINK_FRAMES)
        self.mLinkFrameIds = [tI for tI, tF in enumerate(tPk.model.frames) if tF.name in tNames]
        self.mLastReseed = self.mLastBridge = self.mLastLog = self.mLastPhi = None

    def cornerLift(self, T, g) -> float:
        """lift (m) that puts the lowest finger corner of tool pose T (jaw opening g) at --table-z + --tip-clear (+ left extra)"""
        tA = self.mArgs
        tCornerY = (CORNER_Y_CLOSED_M + CORNER_Y_OPEN_M * float(np.clip(g, 0, 1))) * abs(T[2, 1])
        tLow = T[2, 3] - tA.jaw_half * abs(T[2, 0]) - tCornerY
        return max(0.0, tA.table_z + tA.tip_clear + (tA.tip_clear_left_extra if self.mSide == "left" else 0.0) - tLow)

    def level(self, T, g):
        """umi_replay_bi.level (10-21): where the lowest finger corner would hit the table, tilt the tool axis toward straight
        down by the SMALLEST angle (<= --level-max) that clears it, keeping the TCP position (human height); elsewhere
        untouched"""
        from scipy.spatial.transform import Rotation as Rot
        tA = self.mArgs
        if tA.level_max <= 0 or self.cornerLift(T, g) <= 0:
            return T
        tAxis = T[:3, 2]
        tCross = np.cross(tAxis, list(STRAIGHT_DOWN))
        tNorm = np.linalg.norm(tCross)
        if tNorm < DEGENERATE_AXIS:
            return T
        tMaxAngle = min(np.arccos(np.clip(-tAxis[2], -1, 1)), np.radians(tA.level_max))
        tUnit = tCross / tNorm

        def rot(angle):
            tT2 = T.copy()
            tT2[:3, :3] = Rot.from_rotvec(tUnit * angle).as_matrix() @ T[:3, :3]
            return tT2
        if self.cornerLift(rot(tMaxAngle), g) > 0:
            return rot(tMaxAngle)
        tLo, tHi = 0.0, tMaxAngle
        for _ in range(LEVEL_BISECT_ITERS):
            tMid = (tLo + tHi) / 2
            tLo, tHi = (tMid, tHi) if self.cornerLift(rot(tMid), g) > 0 else (tLo, tMid)
        return rot(tHi)

    def _seeds(self, qPrev, rng, nRand):
        return [self.mQReady, qPrev] + [rng.uniform(self.mLo + SEED_LIMIT_MARGIN_RAD,
                                                    self.mHi - SEED_LIMIT_MARGIN_RAD) for _ in range(nRand)]

    def _multiStart(self, qPrev, T, rng, nRand):
        """unbounded solve from ready + previous + nRand uniform seeds inside the bridge limits -> best (q, pe)"""
        tBest = None
        for tSeed in self._seeds(qPrev, rng, nRand):
            tQ, tPe, _ = self.mF0.ik(tSeed, T)
            if tBest is None or tPe < tBest[1]:
                tBest = (tQ, tPe)
            if tPe < GOOD_RESIDUAL_M:
                break
        return tBest

    def _multiStartNear(self, qPrev, T, rng, nRand, tol=PREFER_NEAR_TOL_M):
        """multi-start; among results within tol of the best residual prefer the one closest to qPrev (joint space)"""
        tRes = [self.mF0.ik(tSeed, T)[:2] for tSeed in self._seeds(qPrev, rng, nRand)]
        tBest = min(tPeCand for _, tPeCand in tRes)
        tCand = [(np.abs(tQCand - qPrev).max(), tQCand, tPeCand) for tQCand, tPeCand in tRes if tPeCand <= tBest + tol]
        _, tQ, tPe = min(tCand, key=lambda x: x[0])
        return tQ, tPe

    @staticmethod
    def _lockOut(pe, start, thr, nBad):
        """first frame >= start that opens a run of nBad frames with residual > thr, else None"""
        tRun = 0
        for tK, tBad in enumerate(pe[start:] > thr):
            tRun = tRun + 1 if tBad else 0
            if tRun >= nBad:
                return start + tK - nBad + 1
        return None

    def _backward(self, T, qStart, peStart, d, lo):
        """bounded backward pass from the new branch at d down to lo -> {t: q}, {t: pe}"""
        tQb, tPeb = {d: qStart}, {d: peStart}
        for tT in range(d - 1, lo - 1, -1):
            tQ, tPe, _ = self.mFree.ik(tQb[tT + 1], T[tT])
            tQb[tT] = tQ
            tPeb[tT] = tPe
        return tQb, tPeb

    def _bridge(self, T, Q, PE):
        """[2026-10-23] offline bidirectional recovery bridge on a forward sequential path (in place on copies) -> Q, PE, bridge
        mask, log"""
        tA = self.mArgs
        tFrames = len(T)
        Q = Q.copy()
        PE = PE.copy()
        tB = np.zeros(tFrames, bool)
        tLog = []
        tThr = tA.reseed_mm / MM_PER_M
        tN = tA.reseed_n
        tW = int(round(tA.bridge_window_s / DT_S))
        tVmax = np.radians(tA.ik_step)
        tRng = np.random.default_rng(0)
        tI = 0
        while tI < tFrames:
            tO = self._lockOut(PE, tI, tThr, tN)
            if tO is None:
                break
            tD = tO + tN - 1
            tQs, tPes = self._multiStartNear(Q[tD - 1] if tD > 0 else self.mQReady, T[tD], tRng, tA.seeds)
            if tPes > tThr:
                tLog.append(dict(onset=int(tO), detect=int(tD), ok=False, why="no reachable seed"))
                tI = tD + 1
                continue
            tLo = max(tO - tW, 0)
            tQb, tPeb = self._backward(T, tQs, tPes, tD, tLo)
            tBest = None
            for tTf in range(tLo, tD):
                for tTb in range(tTf + 1, tD + 1):
                    tGap = np.abs(tQb[tTb] - Q[tTf]).max()
                    tLen = tTb - tTf
                    if tGap * MIN_JERK_PEAK / tLen > tVmax:
                        continue
                    tCost = (tLen, tGap)
                    if tBest is None or tCost < tBest[0]:
                        tBest = (tCost, tTf, tTb)
                    break                                                         # shortest feasible tb for this tf
            if tBest is None:
                tLog.append(dict(onset=int(tO), detect=int(tD), ok=False, why="no feasible bridge"))
                tI = tD + 1
                continue
            _, tTf, tTb = tBest
            tLen = tTb - tTf
            tQa, tQz = Q[tTf].copy(), tQb[tTb].copy()
            tPrev = tQa
            for tJ, tT in enumerate(range(tTf + 1, tTb)):
                tU = _minJerk((tJ + 1) / tLen)
                tQi = tQa + tU * (tQz - tQa)
                tQr, _, _ = self.mFine.ik(tQi, T[tT])                              # small refine toward the Cartesian target
                tStep = np.clip(tQr - tPrev, -tVmax, tVmax)
                tQn = tPrev + tStep
                Q[tT] = tQn
                PE[tT] = poseErr(self.mPk.fk(tQn), T[tT])
                tB[tT] = True
                tPrev = tQn
            for tT in range(tTb, tD + 1):
                Q[tT] = tQb[tT]
                PE[tT] = tPeb[tT]
            tQ = tQs                                                               # continue forward from the new branch
            for tT in range(tD + 1, tFrames):
                tQ, tPe, _ = self.mFree.ik(tQ, T[tT])
                Q[tT] = tQ
                PE[tT] = tPe
            tSeg = Q[max(tTf - 1, 0): min(tTb + 2, tFrames)]
            tDq = np.degrees(np.abs(np.diff(tSeg, axis=0))).max() if len(tSeg) > 1 else 0.0
            tLog.append(dict(onset=int(tO), detect=int(tD), t_f=int(tTf), t_b=int(tTb), ok=True, bridge_s=round(tLen * DT_S, 2),
                             shift_s=round((tO - tTf) * DT_S, 2), max_dq_deg=round(float(tDq), 2),
                             seed_err_mm=round(tPes * MM_PER_M, 1)))
            tI = tD + 1
        return Q, PE, tB, tLog

    def _minJerkRamp(self, Q, PE, B, T, tf, tb, qa, qz):
        """min-jerk joint ramp qa (at tf) -> qz (at tb), refined toward T and clipped to --ik-step; writes Q/PE/B on (tf, tb)"""
        tVmax = np.radians(self.mArgs.ik_step)
        tLen = tb - tf
        tPrev = qa
        for tJ, tT in enumerate(range(tf + 1, tb)):
            tU = _minJerk((tJ + 1) / tLen)
            tQi = qa + tU * (qz - qa)
            tQr, _, _ = self.mFine.ik(tQi, T[tT])
            tQn = tPrev + np.clip(tQr - tPrev, -tVmax, tVmax)
            Q[tT] = tQn
            PE[tT] = poseErr(self.mPk.fk(tQn), T[tT])
            B[tT] = True
            tPrev = tQn

    @staticmethod
    def _join(pathA, pathB, taLo, taHi, tbHi, vmax):
        """first feasible min-jerk join from pathA at tf in [taLo, taHi) to pathB at tb in (tf, tbHi]: min (length, gap)"""
        tBest = None
        for tTf in range(taLo, taHi):
            for tTb in range(tTf + 1, tbHi + 1):
                if tTb not in pathB or tTf not in pathA:
                    continue
                tGap = np.abs(pathB[tTb] - pathA[tTf]).max()
                tLen = tTb - tTf
                if tGap * MIN_JERK_PEAK / tLen <= vmax:
                    if tBest is None or (tLen, tGap) < tBest[0]:
                        tBest = ((tLen, tGap), tTf, tTb)
                    break
        return tBest

    def _detourBridge(self, T, Q, PE):
        """[2026-10-23] detour bridge (bridge v2, was _bridge2): for every lock-out run of the forward path (>= --reseed-n frames
        > --reseed-mm), solve a fresh branch at detection (multi-start, nearest to the old pose), track it backward (bridge-in)
        and forward (bridge-out), join the old path with min-jerk ramps (<= --ik-step) and REJOIN the old path once it has
        recovered, so clean spans outside detours are unchanged. Join windows: --bridge-window-s before the onset,
        --rejoin-window-s after the recovery. No rejoin -> the new branch is kept."""
        tA = self.mArgs
        tFrames = len(T)
        Q = Q.copy()
        PE = PE.copy()
        tB = np.zeros(tFrames, bool)
        tLog = []
        tThr = tA.reseed_mm / MM_PER_M
        tN = tA.reseed_n
        tWin = int(round(tA.bridge_window_s / DT_S))
        tWout = int(round(tA.rejoin_window_s / DT_S))
        tVmax = np.radians(tA.ik_step)
        tRng = np.random.default_rng(0)
        tI = 0
        while tI < tFrames:
            tO = self._lockOut(PE, tI, tThr, tN)
            if tO is None:
                break
            tD = tO + tN - 1
            # old path recovers: first frame >= d with N clean frames
            tE = tD + 1
            while tE < tFrames and not (PE[tE: tE + tN] <= RECOVERED_M).all():
                tE += 1
            tE = min(tE, tFrames)
            tQs, tPes = self._multiStartNear(Q[tD - 1] if tD > 0 else self.mQReady, T[tD], tRng, tA.seeds)
            if tPes > tThr:
                tLog.append(dict(onset=int(tO), detect=int(tD), recover=int(tE), ok=False, why="no reachable seed"))
                tI = tD + 1
                continue
            tLo = max(tO - tWin, 0)
            tQb, tPeb = self._backward(T, tQs, tPes, tD, tLo)
            tHi = min(tE + tWout, tFrames - 1)
            tQf, tPef = {tD: tQs}, {tD: tPes}
            for tT in range(tD + 1, tHi + 1):
                tQ, tPe, _ = self.mFree.ik(tQf[tT - 1], T[tT])
                tQf[tT] = tQ
                tPef[tT] = tPe
            tAltBad = float(np.mean([tPef[tT] > tThr for tT in range(tD, tE)]))
            tOldBad = float(np.mean(PE[tD:tE] > tThr))
            if tAltBad >= tOldBad:
                tLog.append(dict(onset=int(tO), detect=int(tD), recover=int(tE), ok=False, why="new branch not better"))
                tI = tE
                continue
            tIn = self._join({tT: Q[tT] for tT in range(tLo, tD)}, tQb, tLo, tD, tD, tVmax)
            if tIn is None:
                tLog.append(dict(onset=int(tO), detect=int(tD), recover=int(tE), ok=False, why="no feasible bridge-in"))
                tI = tD + 1
                continue
            _, tTf, tTb = tIn
            tOld = {tT: Q[tT].copy() for tT in range(tE, tFrames)}
            tOut = None
            if tE < tFrames:
                tOut = self._join(tQf, tOld, tE, tHi + 1, min(tHi + tWout, tFrames - 1), tVmax)
            self._minJerkRamp(Q, PE, tB, T, tTf, tTb, Q[tTf].copy(), tQb[tTb].copy())
            for tT in range(tTb, tD + 1):
                Q[tT] = tQb[tT]
                PE[tT] = tPeb[tT]
            if tOut is not None:
                _, tTf2, tTb2 = tOut
                for tT in range(tD + 1, tTf2 + 1):
                    Q[tT] = tQf[tT]
                    PE[tT] = tPef[tT]
                self._minJerkRamp(Q, PE, tB, T, tTf2, tTb2, tQf[tTf2].copy(), tOld[tTb2].copy())
                tB[tTb:tTf2 + 1] = True
                tNext = tTb2
            else:                                                                 # keep the new branch to the end
                tQ = tQs
                for tT in range(tD + 1, tFrames):
                    tQ, tPe, _ = self.mFree.ik(tQ, T[tT])
                    Q[tT] = tQ
                    PE[tT] = tPe
                tB[tTb:tFrames] = True
                tTf2 = tTb2 = None
                tNext = tD + 1
            tLog.append(dict(onset=int(tO), detect=int(tD), recover=int(tE), t_f=int(tTf), t_b=int(tTb), ok=True,
                             rejoin=tOut is not None,
                             t_f2=None if tTf2 is None else int(tTf2), t_b2=None if tTb2 is None else int(tTb2),
                             bridge_s=round((tTb - tTf) * DT_S, 2), shift_s=round((tO - tTf) * DT_S, 2),
                             seed_err_mm=round(tPes * MM_PER_M, 1)))
            tI = max(tNext, tD + 1)
        return Q, PE, tB, tLog

    def solve(self, Th, G):
        """IK over targets Th (N,4,4) -> Q, pos err (m), flip; mLastReseed / mLastBridge / mLastLog / mLastPhi of the chosen flip.
        --ik-mode (10-23 lock-out audit; default seq = production):
        seq      sequential, --ik-step bound from the previous pose (umi_replay_bi.solve_path behaviour)
        reseed   seq, but after --reseed-n consecutive frames with error > --reseed-mm the bound is dropped for that frame and it
        is
                 solved fresh from ready + previous + --seeds random seeds (best kept), then bounded again
        indep    every frame independently, multi-start, no continuity (oracle reachability diagnostic)
        bridge   seq + the recovery bridge (_bridge); bridge2 = seq + the detour bridge (_detourBridge)
        The tool-axis 180 deg flip is tried both ways (indep with indep_flip: only that one); the flip with fewer frames > 10 mm
        (then the smaller max error) wins. 10-26: wrist-roll yield above |j6| --roll-thr is a per-frame runtime correction."""
        tA = self.mArgs
        tJ6Hi = np.radians(tA.roll_thr)
        tBest = None
        tMode = getattr(tA, "ik_mode", "seq")
        tRng = np.random.default_rng(0)
        tFlips = (False, True) if tMode != "indep" or getattr(tA, "indep_flip", None) is None else (bool(tA.indep_flip),)
        for tFlip in tFlips:
            tQNow = self.mQReady
            tQ, tPe, tRs, tPhiAll = [], [], [], []
            tPhi, tBad = 0.0, 0
            for tI in range(len(Th)):
                tT = Th[tI].copy()
                tT[:3, :3] = tT[:3, :3] @ (RZ180 if tFlip else np.eye(3))
                tC, tS = np.cos(tPhi), np.sin(tPhi)
                tT[:3, :3] = tT[:3, :3] @ np.array([[tC, -tS, 0], [tS, tC, 0], [0, 0, 1.0]])
                tReseeded = False
                if tMode == "indep":
                    tQNow, tPeNow = self._multiStart(tQNow, tT, tRng, tA.seeds)
                else:
                    tQTry, tPeNow, _ = (self.mF0 if tI == 0 else self.mFree).ik(tQNow, tT)
                    tBad = tBad + 1 if tPeNow > tA.reseed_mm / MM_PER_M else 0
                    if tMode == "reseed" and tBad >= tA.reseed_n:
                        tQ2, tPe2 = self._multiStart(tQNow, tT, tRng, tA.seeds)
                        if tPe2 < tPeNow:
                            tQTry, tPeNow, tReseeded = tQ2, tPe2, True
                        tBad = 0
                    tQNow = tQTry
                tOver = abs(tQNow[WRIST_ROLL_JOINT]) - tJ6Hi
                if tMode != "indep":
                    if tOver > 0:
                        tPhi -= np.sign(tQNow[WRIST_ROLL_JOINT]) * tOver
                    elif tPhi != 0:
                        tPhi -= np.sign(tPhi) * min(abs(tPhi), np.radians(tA.roll_relax), max(0.0, -tOver))
                tQ.append(tQNow)
                tPe.append(tPeNow)
                tRs.append(tReseeded)
                tPhiAll.append(tPhi)
            tQ, tPe = np.array(tQ), np.array(tPe)
            tBridge, tLog = np.zeros(len(tQ), bool), []
            if tMode in ("bridge", "bridge2"):
                tTargets = []
                # same targets as the forward pass (no roll yield in the bridge)
                for tI in range(len(Th)):
                    tT = Th[tI].copy()
                    tT[:3, :3] = tT[:3, :3] @ (RZ180 if tFlip else np.eye(3))
                    tTargets.append(tT)
                tQ, tPe, tBridge, tLog = (self._bridge if tMode == "bridge" else self._detourBridge)(tTargets, tQ, tPe)
            tScore = (float(np.mean(tPe > FLIP_BAD_M)), float(tPe.max()))
            if tBest is None or tScore < tBest[0]:
                tBest = (tScore, tQ, tPe, tFlip, np.array(tRs), tBridge, tLog, np.degrees(np.array(tPhiAll)))
        _, tQ, tPe, tFlip, self.mLastReseed, self.mLastBridge, self.mLastLog, self.mLastPhi = tBest
        return tQ, tPe, tFlip

    def links(self, q6):
        """origins (base frame, m) of the link set of the inter-arm gap: link3-6, gripper_link, fingers, tcp -> (8, 3)"""
        self.mPin.framesForwardKinematics(self.mPk.model, self.mPk.data, self.mPk._q(q6))
        return np.array([self.mPk.data.oMf[tI].translation for tI in self.mLinkFrameIds])
