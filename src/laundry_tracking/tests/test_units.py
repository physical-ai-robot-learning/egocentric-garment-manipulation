"""Characterization of the pure laundry_tracking functions on the in-memory synthetic arm (synthetic_world.syntheticArm): pose
smoother stages and baselines, SLAM<->IMU self-consistency helpers, IMU motion / AHRS, sync lag, rigid fit, gate / fusion / quality
helpers, calibration-tool math and argparse defaults.  Outputs are compared with the golden record of the pre-refactor code."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from laundry_tracking.tests import golden_record as G
from laundry_tracking.constants import ORBBEC_FRAME_S
from laundry_tracking.intervals import runDurationsS, trueRuns
from laundry_tracking.tests import synthetic_world as W

pytest.importorskip("cv2")
pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[3]
S = 1_000_000_000


def _arm(arm="left"):
    return W.syntheticArm(arm)


def _coreSegs(D):
    """fgo segment dicts -> the raw-segment dicts of core.load_segs_raw"""
    tOut = []
    for tI, tS in enumerate(D["segs"]):
        tN = len(tS["t"])
        tOut.append(dict(t=tS["t"].copy(), p=tS["p"].copy(), R=tS["R"], sim3=1.0 + 0.01 * np.sin(np.arange(tN) / 7.0),
                         kf=np.arange(tN) % 5 == 0, lost_end=tI == 1, gi0=tS["start"], win=0))
    return tOut


def _windows():
    from laundry_tracking.fgo.arm_data import DemoWindow
    return [DemoWindow(mDemo=tI, mA0=W.T_START_NS + int(tT0 * S), mW0=W.T_START_NS + int((tT0 + 1.0) * S),
                       mW1=W.T_START_NS + int((tT0 + tDur - 1.0) * S), mKeptS=tDur - 2.0) for tI, (tT0,
                                                                                                   tDur) in enumerate(W.DEMOS)]


# ------------------------------------------------------------------------------------------------ pose smoother (fgo_core)
def test_pose_smoother():
    from laundry_tracking.fgo import pose_smoother as FC
    D = _arm("left")
    D["slam_ok"][::4] = True
    FC.prepGyro(D)
    tRes = dict(b0=D["b0"], Q=D["Q"].as_quat()[::50])
    tT = W.T_START_NS + np.arange(3 * S, 9 * S, 37_000_000, dtype=np.int64)
    tN = W.T_START_NS + np.arange(3 * S, 9 * S, 50_000_000, dtype=np.int64)
    tRes["Qat"] = FC.gyroAt(D, tT).as_quat()
    tRes["node_of"] = FC.nodeOf(tN, tT)
    tRes["rot_at"] = FC.rotationAt(D, tN, R.from_rotvec(np.zeros((len(tN), 3))), tT).as_quat()
    tRes["slam_in"] = [dict(k=tS["k"], t=tS["t"], p=tS["p"], R=tS["R"].as_quat(), t0=tS["t0"]) for tS in FC.slamInWindow(D,
            tN[0], tN[-1])]
    tRes["hide_mask"] = [FC.hideMask(tT, None), FC.hideMask(tT, (tT[10], tT[40]))]
    tRes["skew"] = FC._skew(np.array([[1.0, 2.0, 3.0], [-0.5, 0.1, 0.7]]))
    tW0, tW1 = W.T_START_NS + 3 * S, W.T_START_NS + 9 * S
    tHide = (W.T_START_NS + 5 * S, W.T_START_NS + 6 * S)
    tSm = FC.smooth(D, tW0, tW1, hide=tHide)
    tRes["smooth"] = dict(tn=tSm["tn"], Rn=tSm["Rn"].as_quat(), Pn=tSm["Pn"], std=tSm["std"], info=tSm["info"])
    tSm2 = FC.smooth(D, tW0, tW1, o=dict(slam_rot=False, slam_pos=False, gn_iters=6), wantCov=False)
    tRes["smooth_tag_only"] = dict(Rn=tSm2["Rn"].as_quat(), Pn=tSm2["Pn"], std=tSm2["std"], info=tSm2["info"])
    tP, tRr, tSd = FC.evalAt(D, tSm, tT)
    tRes["eval_at"] = [tP, tRr.as_quat(), tSd]
    tEval = D["t"][(D["t"] >= tHide[0]) & (D["t"] <= tHide[1])]
    tP, tRr = FC.methodA(D, tEval, tHide)
    tRes["method_a"] = [tP, tRr.as_quat()]
    tRf = R.from_matrix(D["R_fused"])
    tRes["gyro_bridge"] = FC.gyroBridge(D, tHide[0], tHide[1], tRf[10], tRf[200], tEval).as_quat()
    tP, tRr, tBest = FC.methodB(D, tEval, tHide)
    tRes["method_b"] = [tP, tRr.as_quat(), tBest]
    tP, tRr, tBest = FC.methodB(D, tEval, tHide, ctxS=0.5, maxRes=0.5)
    tRes["method_b_loose"] = [tP, tRr.as_quat(), tBest]
    tP, tRr, tOk = FC.methodAClean(D, tEval, tHide)
    tRes["method_a_clean"] = [tP, tRr.as_quat(), tOk]
    tP, tRr, tOk = FC.methodAClean(D, tEval, tHide, p90Mm=500.0)
    tRes["method_a_clean_loose"] = [tP, tRr.as_quat(), tOk]
    tRes["oracle_slam"] = FC.oracleSlam(D, tEval)
    G.checkGolden("unit_pose_smoother", G.summarize(tRes))


# ------------------------------------------------------------------------------------------------ SLAM<->IMU consistency (core)
def test_slam_consistency():
    from laundry_tracking import slam_consistency as C
    from laundry_tracking.fgo import pose_smoother
    D = _arm("right")
    tSegs = _coreSegs(D)
    tRes = dict(slam_tag=[C.slamTag(W.TAKE, "wrist"), C.slamTag(W.TAKE, "wrist_left")])
    tA = R.from_rotvec([[0.1, 0.2, -0.3], [0.5, -0.1, 0.0], [0.0, 0.3, 0.2], [-0.2, 0.1, 0.4]]).as_rotvec()
    tB = R.from_euler("xyz", [20, -10, 5], degrees=True).inv().apply(tA) + 0.001
    tRes["kabsch_rot"] = [C.kabschRotation(tA, tB).as_quat(), C.kabschRotation(tA, tB, np.array([1.0, 2.0, 0.5, 1.0])).as_quat()]
    pose_smoother.prepGyro(D)
    tT = W.T_START_NS + np.arange(3 * S, 9 * S, 100_000_000, dtype=np.int64)
    tRes["gyro_rel"] = C.gyroRelative(D, tT[:-5], tT[5:]).as_quat()
    tRes["omega_mag_corr"] = C.omegaMagCorr(D, tSegs)
    tRX, tN, tMed, tPairs = C.handEyeRotation(D, tSegs)
    tRes["handeye_rot"] = [tRX.as_quat(), tN, tMed, list(tPairs)]
    tRes["refine_bias"] = C.refineBias(D, tRX, tPairs)
    tD2 = dict(D)
    C.integrateGyro(tD2, D["b0"] + 0.001)
    tRes["integrate_q"] = [tD2["Q"].as_quat()[::40], tD2["bias_total"]]
    tCal = C.calibrate(D, tSegs)
    tRes["calibrate"] = tCal
    tRes["calibrate_segs_t"] = [tS["t"] for tS in tSegs]
    tD3 = _arm("right")
    tSegs3 = _coreSegs(tD3)
    tRes["calibrate_fixed"] = C.calibrate(tD3, tSegs3, W.RX)
    tRX = tCal["RX_est"]
    D["tX_val"] = np.array([0.0, 0.04, -0.07])
    for tS in tSegs:
        C.prepSegment(D, tS, tRX)
    tRes["prep_seg"] = [{tKey: (tVal.as_quat() if hasattr(tVal,
                                                          "as_quat") else tVal) for tKey, tVal in tS.items() if tKey not in ("Rs",)} for tS in tSegs]
    tCache = {}
    tRes["overlap_k"] = [C.overlapScale(tSegs[0], tSegs[1], tRX, tCache, (0, 1)),
                         C.overlapScale(tSegs[1], tSegs[2], tRX, tCache, (1, 2)),
                         C.overlapScale(tSegs[0], tSegs[2], tRX, tCache, (0, 2)), sorted(tCache)]
    tRes["pos"] = C.segmentPosition(tSegs[0], tT[:20])
    tRes["disp_cam"] = C.cameraDisplacement(tSegs[0], tT[:20], tT[5:25])
    tRes["win_in_seg"] = [C.windowInSegment(tSegs[0], tT[0], 0.4), C.windowInSegment(tSegs[0], tSegs[0]["t0"], 1.6),
                          C.windowInSegment(tSegs[1], tT[3], 100.0)]
    tRes["rot_res"] = [C.rotationResidual(D, tSegs[0], tT[2], tT[10], tRX), C.rotationResidual(D, tSegs[0], tT[2], tT[10], tRX,
                                                                                               True)]
    tRes["tag_frame"] = [C.tagFrame(D, int(tTt)) for tTt in tT[::7]]
    tX, tL, tM = C.chunkXly(D, tSegs[0], tRX, tT[1], tT[17])
    tRes["chunk_xly"] = [tX, tL, tM.as_quat()]
    C.validationPrep(D, tSegs, tRX)
    tRes["val_prep"] = [[list(tP) for tP in tS["vp"]] for tS in tSegs]
    tRows = []
    for tTt in tT[::4]:
        tRow = {}
        C.validationChunk(D, tSegs, 0, int(tTt), tRX, tRow)
        tRows.append(tRow)
    tRes["val_chunk"] = tRows
    G.checkGolden("unit_slam_consistency", G.summarize(tRes))


# ------------------------------------------------------------------------------------------------ IMU motion / AHRS / sync lag
def test_imu_motion():
    from laundry_tracking.fgo import pose_smoother
    from laundry_tracking import imu_motion as IM
    D = _arm("left")
    pose_smoother.prepGyro(D)
    tRes = dict(grid=IM.motionGrid(W.T_START_NS + 3 * S, W.T_START_NS + 4 * S))
    tG = IM.motionGrid(W.T_START_NS + 3 * S, W.T_START_NS + 12 * S)
    tAnchorT = W.T_START_NS + 3 * S
    tAnchorR = R.from_matrix(D["R_tag"][50])
    tRes["imu_lin_acc"] = IM.imuLinearAcc(D, tG, tAnchorT, tAnchorR)
    tC = W.bodyCentre("left", (D["t"] - W.T_START_NS) / 1e9)
    tC[100:104] = np.nan
    tC[200:220] = np.nan
    tAo = IM.trackAcc(D["t"], tC, tG)
    tRes["track_acc"] = [tAo, IM.trackAcc(D["t"], tC, tG, maxGapS=0.1), IM.trackAcc(D["t"][:4], tC[:4], tG)]
    tAi = IM.imuLinearAcc(D, tG, tAnchorT, tAnchorR)
    tRes["window_ncc"] = [IM.windowNcc(tAo, tAi), IM.windowNcc(tAo, tAi, winS=1.0, eMin=0.1)]
    tRes["mahony"] = IM.mahony(D["imu_t"][:800], D["gyro"][:800], D["acc"][:800], D["b0"])[::20]
    tRes["rest_bias"] = IM.restBias(D, _windows())
    D["b_rest"] = tRes["rest_bias"]
    tRes["ahrs_base"] = IM.ahrsBase(D, tG, tAnchorT, tAnchorR).as_quat()
    tRes["rest_bias_none"] = IM.restBias(D, _windows()[:1])
    tK = np.flatnonzero((D["t"] >= W.T_START_NS + 3 * S) & (D["t"] <= W.T_START_NS + 13 * S))
    tRes["sync_lag"] = IM.syncLag(D, D["t"], tC, tK, int(D["t"][tK[0]]), tAnchorR)
    tRes["sync_lag_short"] = IM.syncLag(D, D["t"], tC, tK[:20], int(D["t"][tK[0]]), tAnchorR)
    tRes["runs"] = trueRuns(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool))
    G.checkGolden("unit_imu_motion", G.summarize(tRes))


# ------------------------------------------------------------------------------------------------ rigid fit, gate, fusion, quality helpers
def test_tracker_helpers():
    from laundry_tracking.fgo import pose_smoother
    from laundry_tracking import imu_motion as IM
    from laundry_tracking import arm_rotation as V
    from laundry_tracking.cotrack import episode_quality as EQ
    from laundry_tracking.cotrack import gate as GT
    from laundry_tracking.fusion import fusion_metrics as FM
    from laundry_tracking.fusion import trajectory_fusion as TF
    from laundry_tracking import whole_demo_stitch as wdo
    tRng = np.random.default_rng(5)
    tModel = tRng.normal(size=(300, 3))
    tModel = tModel / np.linalg.norm(tModel, axis=1, keepdims=True) * W.BODY_R_M
    tRm = R.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    tTrue = np.array([0.4, 0.1, 0.1])
    tPts = tModel[:200] @ tRm.T + tTrue + tRng.normal(0, 0.001, (200, 3))
    tRes = dict(rigid_fit=[V.rigidFit(tPts, tRm, tModel, tTrue + 0.02), V.rigidFit(tPts[:30], tRm, tModel, tTrue),
                           V.rigidFit(tPts, tRm, tModel, tTrue + np.array([0.3, 0.0, 0.0]))])
    D = _arm("right")
    pose_smoother.prepGyro(D)
    D["b_rest"] = IM.restBias(D, _windows())
    tWin = _windows()[0]
    tK = np.flatnonzero((D["t"] >= tWin.mW0) & (D["t"] <= tWin.mW1))
    tC = W.bodyCentre("right", (D["t"][tK] - W.T_START_NS) / 1e9)
    tC[40:50] = np.nan
    tHid = np.zeros(len(D["t"]), bool)
    tNcc, tDec, tRk = GT.imuNcc(D, D["t"][tK], tC, tWin, tHid)
    tRes["imu_ncc"] = [tNcc, tDec, tRk]
    tRes["gate_runs"] = trueRuns(np.array([1, 1, 0, 1], bool))
    tTn = W.T_START_NS + np.arange(0, 2 * S, 50_000_000, dtype=np.int64)
    tRes["lin"] = TF.lin(tTn, tTn[::3] + 12_345_678)
    tRows = [([0, 1], [0.5, 0.5], np.array([1.0, 2.0, 3.0]), 0.1, True), ([1, 2], [1.0, -1.0], np.zeros(3), 0.05, False),
             ([2], [1.0], np.array([0.5, 0.0, -1.0]), 0.2, True), ([0], [1.0], np.array([9.0, 9.0, 9.0]), 0.1, True)]
    tRes["irls"] = TF.irls(tRows, 3)
    tRes["quality_runs_s"] = runDurationsS(np.array([0, 1, 1, 0, 1], bool), tTn[:5], 0.05)
    tNear = np.zeros(10, bool)
    tNear[6] = True
    tRes["grasp_ok"] = [EQ.graspOk(np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], bool), tNear, tTn[:10] * 10, 0.05),
                        EQ.graspOk(np.zeros(10, bool), tNear, tTn[:10], 0.05)]
    tRes["metrics_runs_s"] = runDurationsS(np.array([1, 1, 0, 1], bool), tTn[:4], ORBBEC_FRAME_S)
    tRes["metrics_interp"] = FM.interp(tTn[:5], np.arange(15.0).reshape(5, 3), tTn[:5] + 7_000_000)
    tRes["hidden_mask"] = [wdo.hiddenMask(D["t"], -1), wdo.hiddenMask(D["t"], 0), wdo.hiddenMask(D["t"], 2, blockS=2.0, mod=4)]
    tD2 = wdo.maskTags(D, wdo.hiddenMask(D["t"], 1))
    tRes["mask_tags"] = [tD2["ntags"], tD2["n_tagrot"], tD2["R_fused"]]
    G.checkGolden("unit_tracker_helpers", G.summarize(tRes))


# ------------------------------------------------------------------------------------------------ calibration tools
def test_calibration_math(capsys):
    from laundry_tracking import camera_calibration as camcal
    from laundry_tracking.calib import robot_cam_solve as RCS
    from laundry_tracking.calib import tag_rig_pivot_joint as PJ
    from laundry_tracking.calib import tag_rig_solve as TRS
    from laundry_tracking.calib import umi_tcp_traj as UT
    tRes = dict(camcal=[camcal.calibPath(tTake).name for tTake in ("20261020_100000", "20261026_111328", "20261026_115441", "20261026_120052",
                                                              "20261026_120504", "20261103_120000")])
    tT1 = TRS.poseFromPnp(np.array([0.1, -0.2, 0.3]), np.array([0.5, 0.0, 1.0]))
    tT2 = TRS.poseFromPnp(np.array([0.12, -0.18, 0.31]), np.array([0.51, 0.01, 0.98]))
    tRes["T_from"] = tT1
    tRes["mean_T"] = TRS.meanPose([tT1, tT2, TRS.poseFromPnp(np.array([0.09, -0.21, 0.29]), np.array([0.49, -0.01, 1.01]))])
    tRes["dT"] = TRS.poseDifference(tT1, tT2)
    tRng = np.random.default_rng(3)
    tPRig = np.array([0.02, -0.01, 0.12])
    tBlocks = []
    for tB in range(3):
        tRs = R.from_rotvec(tRng.normal(0, 0.4, (40, 3))).as_matrix()
        tPC = np.array([0.1 * tB, 0.2, 0.9])
        tTs = tPC - np.einsum("nij,j->ni", tRs, tPRig) + tRng.normal(0, 0.0015, (40, 3))
        tTs[::13] += 0.03
        tBlocks.append((tRs, tTs))
    tRes["pivot_joint_solve"] = PJ.solve(tBlocks)
    tRes["rcs_T_of"] = RCS.poseFromRotvec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
    tRes["rcs_inv"] = RCS.invertPose(tT1)
    tK = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    tTct = RCS.poseFromRotvec(np.array([3.0, 0.1, -0.05]), np.array([0.02, -0.01, 0.8]))
    tPx = RCS.project(tTct, tK, 0.05)
    tRes["rcs_project"] = tPx
    tRes["rcs_pnp"] = RCS.pnp(tPx, tK, 0.05)
    tRes["rcs_pnp2"] = RCS.pnpBoth(tPx, tK, 0.05)
    tRes["rcs_p3"] = [RCS.tagCentre(dict(px=tPx, depth_m=0.81), tK, 0.05), RCS.tagCentre(dict(px=tPx, depth_m=None), tK, 0.05)]
    tP = tRng.normal(size=(10, 3))
    tRes["rcs_kabsch"] = RCS.kabsch(tP, tP @ R.from_rotvec([0.1, 0.2, 0.3]).as_matrix().T + 1.0)
    tOk = RCS.synthetic(False)
    tOk2 = RCS.synthetic(True, noDepth=True)
    tRes["rcs_synthetic"] = [tOk, tOk2, capsys.readouterr().out]
    tMeta = dict(t_start_monotonic_ns=W.T_START_NS, protocol=[
        dict(text="오른손, X축, 1회, 멈추세요", t0=0.0, dur=2.0), dict(text="오른손 앞으로 5센티 천천히", t0=2.0, dur=2.0),
        dict(text="멈추세요", t0=4.0, dur=1.0), dict(text="돌아오세요", t0=5.0, dur=2.0), dict(text="멈추세요", t0=7.0, dur=1.0),
        dict(text="왼손, 회전, 2회, 멈추세요", t0=8.0, dur=2.0), dict(text="왼손 왼쪽돌리기 30도 천천히", t0=10.0, dur=2.0),
        dict(text="멈추세요", t0=12.0, dur=1.0), dict(text="돌아오세요", t0=13.0, dur=2.0),
        dict(text="오른손, 자유동작, 3회", t0=15.0, dur=1.0), dict(text="자유: 접기", t0=16.0, dur=3.0), dict(text="자유: 펴기", t0=19.0,
                                                                                                  dur=3.0),
        dict(text="멈추세요", t0=22.0, dur=1.0)])
    tRes["umi_tcp_segments"] = UT.segments(tMeta)
    G.checkGolden("unit_calibration_math", G.summarize(tRes))


class _Captured(Exception):
    pass


_PARSE_ARGS = argparse.ArgumentParser.parse_args


def _parsed(monkeypatch, run, argv):
    tOriginal = _PARSE_ARGS
    tBox = {}

    def _capture(self, args=None, namespace=None):
        tBox["ns"] = tOriginal(self, args, namespace)
        raise _Captured()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    monkeypatch.setattr(sys, "argv", ["prog"] + list(argv))
    with pytest.raises(_Captured):
        run(argv)
    return {tKey: (str(tVal).replace(str(REPO), "<REPO>") if isinstance(tVal,
                                                                        str) else tVal) for tKey, tVal in vars(tBox["ns"]).items()}


def test_calibration_cli_defaults(monkeypatch):
    from laundry_tracking.calib import (robot_cam_from_anchors, robot_cam_solve, tag_rig_pivot, tag_rig_pivot_joint,
                                        tag_rig_solve, tag_rig_touch_tcp, umi_side_check, umi_tcp_traj)
    tRes = dict(
        pivot=_parsed(monkeypatch, lambda a: tag_rig_pivot.main(a), ["take", "--rig", "r.json"]),
        pivot_joint=_parsed(monkeypatch, lambda a: tag_rig_pivot_joint.main(a), ["t1", "t2"]),
        rig_solve=_parsed(monkeypatch, lambda a: tag_rig_solve.main(a), ["take"]),
        touch_tcp=_parsed(monkeypatch, lambda a: tag_rig_touch_tcp.main(a), ["take"]),
        tcp_traj=_parsed(monkeypatch, lambda a: umi_tcp_traj.main(a), ["take"]),
        from_anchors=_parsed(monkeypatch, lambda a: robot_cam_from_anchors.main(a), ["take", "--frames", "0:10:2"]),
        robot_cam_solve=_parsed(monkeypatch, lambda a: robot_cam_solve.main(a), ["take", "run"]),
        side_check=_parsed(monkeypatch, lambda a: umi_side_check.main(a), []),
    )
    G.checkGolden("unit_calibration_cli", G.summarize(tRes))
