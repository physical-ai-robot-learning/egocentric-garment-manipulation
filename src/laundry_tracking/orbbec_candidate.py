"""orbbec_candidate_v0: continuous Orbbec + AHRS TCP trajectory per take / arm -> a SEPARATE derived root that the unchanged production
builder (laundry_dataset tag_rig_unsplit -> robotize_ik) turns into a candidate LeRobot dataset (was orbbec/v2b/build_candidate_v0.py,
10-23).

Per Orbbec frame (the umi_traj grid), per arm:
  position  ORBBEC  tracker output (FUSION_WORK/orbbec/v2b/out/<track>_<take>.npz: v1 dark detector, tag-free start-marker seed) ->
                    TCP = c - R o (o leave-one-demo-out, tags = calibration only), kept EXACTLY (no scale / world refit, no smoothing)
            gaps    shape = the production tag/IMU trajectory p_B (umi_traj, tag-anchored where tags are seen), shifted by a delta that
                    blends linearly from (orbbec - p_B) at the gap start to the gap end -> continuous at both ends (smooth rejoin, no
                    snap); every fill frame is INTERP (tag_in_gap flags gaps that contain >= 2-tag frames); TAG = tag measurement used
                    directly as the position (none in v0).  SLAM fill: not in v0.
            ends    before the first / after the last Orbbec sample of a demo: p_B + constant delta of the nearest sample (INTERP,
                    unresolved)
  rotation  AHRS (rest-period gyro bias + accel tilt), one yaw anchor per demo window at its first >= 2-tag RAW PnP rotation (R_tag,
            calibration; the stored R_fused carries the old fgo/tag_rig_fuse gyro bias and is not used), IMU sampled at t + lag where
            lag = per-demo Orbbec<->IMU clock offset (imu_motion.syncLag; ncc < LAG_MIN_NCC -> 0 + flag)
  offset o  body centre -> TCP in the TCP frame, leave-one-demo-out on >= 2-tag frames with R_tag; per demo n / cond / norm / deviation
            from the pooled median are stored in cand_info
  anchor    if a demo has no >= 2-tag rotation, the nearest one of another demo is used (AHRS across the rest; flagged in cand_info)
  demos     no anchor at all or no Orbbec sample -> demo_valid False, conf LOW everywhere, production pose NOT used (pose = NaN-free copy
            kept only so the file loads; the demo must be excluded from the candidate build -- see candidate_demos.json)
  flags     src 0 ORBBEC | 1 TAG | 2 INTERP;  conf 2 HIGH (ORBBEC, npts >= HIGH_PTS) | 1 MID (other ORBBEC, INTERP <= 0.5 s) | 0 LOW
            fused_valid = ORBBEC only (the builder's 'measured');  gap_len_s = length of the gap containing the frame (0 on ORBBEC)
  conf file <conf_dir>/<take>/<arm>.npz {t_ns, conf} with HIGH 1.0 / MID 0.5 / LOW 0.0 -> builder --pos-conf-dir (variant imuconf)
Writes ONLY under --derived-root / --conf-dir.  Production derived dirs are read (assignment.json, segments.json, umi_traj) and copied.
    python -m laundry_tracking.orbbec_candidate --takes 20261023_123747 20261022_143329 20261022_160308 \\
        --derived-root ~/laundry_dit_training/data/tag_rig_derived_orbbec_v0 --conf-dir ~/laundry_dit_training/data/orbbec_conf_v0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

from . import imu_motion
from . import workdirs
from .constants import MM_PER_M, NS_PER_MS, NS_PER_S
from .fgo import arm_data
from .fgo import pose_smoother
from .intervals import trueRuns

HIGH_PTS = 120                     # tracker points for a HIGH confidence frame
SHORT_S = 0.5                      # interior gaps up to this long stay MID
UNRES_S = 1.0                      # longer gaps are unresolved
LAG_MIN_NCC = imu_motion.LAG_MIN_NCC
SRC = {"ORBBEC": 0, "TAG": 1, "INTERP": 2}
CONF_LOW, CONF_MID, CONF_HIGH = 0, 1, 2
CONF_FILE_VALUES = np.array([0.0, 0.5, 1.0])   # conf file value of LOW / MID / HIGH
DEMO_TAIL_S = 2.0                  # a demo owns the frames from its cue start to its window end + 2 s
END_GAP_LEN_S = 9.9                # gap_len_s marker of frames before the first / after the last Orbbec sample
POSE_SRC = "orbbec_candidate_v0"


class CandidateError(ValueError):
    """The tracker output, the production trajectory and the arm record do not describe the same frames / rigs."""


def buildArm(take: str, arm: str, z, prodTraj: dict):
    """-> (umi_traj dict for the candidate root, conf per frame, cand_info rows) (was build_candidate_v0.build_arm)"""
    D = arm_data.loadArm(take, arm, withSlam=False)
    pose_smoother.prepGyro(D)
    tWindows = arm_data.loadTrimWindows()[take]
    D["b_rest"] = imu_motion.restBias(D, tWindows)
    tT = z["t_ns"]
    tN = len(tT)
    if not (np.array_equal(tT, D["t"]) and np.array_equal(tT, prodTraj["t_ns"])):
        raise CandidateError(f"{take} {arm}: tracker, arm record and production trajectory are on different frames")
    tC = z["c_" + arm]
    tNpts = z["npts_" + arm]
    tValid = np.isfinite(tC[:, 0])
    tRb = D["R_tag"]
    tTag2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(tRb[:, 0, 0])
    tDemoOf = np.full(tN, -1)
    for tWin in tWindows:
        tDemoOf[(tT >= tWin.mA0) & (tT <= tWin.mW1 + DEMO_TAIL_S * NS_PER_S)] = tWin.mDemo
    tP = prodTraj["p_B"].copy()
    tRt = R.from_matrix(prodTraj["R_B_tcp"])
    tSrc = np.full(tN, SRC["INTERP"])
    tConf = np.zeros(tN, int)
    tGapLen = np.zeros(tN)
    tUnres = np.zeros(tN, bool)
    tTagInGap = np.zeros(tN, bool)
    tDemoValid = np.zeros(tN, bool)
    tInfo = []
    tAllOffsets = []
    for tWin in tWindows:
        tI = tWin.mDemo
        tK = np.flatnonzero(tDemoOf == tI)
        tTrain = tTag2 & tValid & (tDemoOf >= 0) & (tDemoOf != tI)
        tA = tRb[tTrain].reshape(-1, 3)
        tO = np.linalg.lstsq(tA, (tC[tTrain] - D["p_tag"][tTrain]).reshape(-1), rcond=None)[0]
        tAllOffsets.append(tO)
        tOInfo = dict(n=int(tTrain.sum()), cond=round(float(np.linalg.cond(tA)), 2),
                      norm_mm=round(float(np.linalg.norm(tO)) * MM_PER_M, 1))
        tAnc = np.flatnonzero((tDemoOf == tI) & tTag2)
        tAncOther = False
        if not len(tAnc) and tTag2.any():                         # no anchor in this demo: nearest >= 2-tag rotation of another demo,
            tJ = np.flatnonzero(tTag2)                            # AHRS propagates across the rest
            tAnc = tJ[[np.argmin(np.abs(tT[tJ] - tT[tK[0]]))]]
            tAncOther = True
        if not len(tK) or not len(tAnc) or not tValid[tK].any():
            tConf[tK] = 0
            tInfo.append(dict(demo=tI, demo_valid=False, why="no >=2-tag anchor" if not len(tAnc) else "no orbbec sample",
                              **tOInfo))
            continue
        tLag, tNcc = imu_motion.syncLag(D, tT, tC, tK, tT[tAnc[0]], R.from_matrix(tRb[tAnc[0]]))
        tRk = imu_motion.ahrsBase(D, tT[tK] + tLag, tT[tAnc[0]] + tLag, R.from_matrix(tRb[tAnc[0]]))
        tRtM = tRt.as_matrix()
        tRtM[tK] = tRk.as_matrix()
        tRt = R.from_matrix(tRtM)
        tVk = tValid[tK]
        tPo = np.full((len(tK), 3), np.nan)
        tPo[tVk] = tC[tK][tVk] - tRk[tVk].apply(tO)
        tPk = tP[tK].copy()
        tSk = np.full(len(tK), SRC["INTERP"])
        tCk = np.zeros(len(tK), int)
        tGk = np.zeros(len(tK))
        tUk = np.zeros(len(tK), bool)
        tPk[tVk] = tPo[tVk]
        tSk[tVk] = SRC["ORBBEC"]
        tCk[tVk] = np.where(tNpts[tK][tVk] >= HIGH_PTS, CONF_HIGH, CONF_MID)
        tIv = np.flatnonzero(tVk)
        if len(tIv) == 0:
            tInfo.append(dict(demo=tI, skipped="no orbbec"))
            continue
        for tS, tE in trueRuns(~tVk):
            tL = (tT[tK][min(tE, len(tK) - 1)] - tT[tK][max(tS - 1, 0)]) / NS_PER_S
            tGk[tS:tE] = tL
            tBase = tP[tK][tS:tE]
            if tS > 0 and tE < len(tK):                           # interior gap: linear delta blend, continuous at both ends
                tD0 = tPo[tS - 1] - tP[tK][tS - 1]
                tD1 = tPo[tE] - tP[tK][tE]
                tW = ((tT[tK][tS:tE] - tT[tK][tS - 1]) / max(tT[tK][tE] - tT[tK][tS - 1], 1))[:, None]
                tPk[tS:tE] = tBase + (1 - tW) * tD0 + tW * tD1
                tSk[tS:tE] = SRC["INTERP"]
                tTagInGap[tK[tS:tE]] = tTag2[tK][tS:tE].any()
                tCk[tS:tE] = CONF_MID if tL <= SHORT_S else CONF_LOW
                tUk[tS:tE] = tL > UNRES_S
            else:                                                 # before the first / after the last Orbbec sample
                tJ = tIv[0] if tS == 0 else tIv[-1]
                tPk[tS:tE] = tBase + (tPo[tJ] - tP[tK][tJ])
                tCk[tS:tE] = CONF_LOW
                tUk[tS:tE] = True
                tGk[tS:tE] = END_GAP_LEN_S
        tP[tK] = tPk
        tSrc[tK] = tSk
        tConf[tK] = tCk
        tGapLen[tK] = tGk
        tUnres[tK] = tUk
        tDemoValid[tK] = True
        tInfo.append(dict(demo=tI, demo_valid=True, offset_mm=(tO * MM_PER_M).round(1).tolist(), lag_ms=round(tLag / NS_PER_MS),
                          lag_ncc=round(tNcc, 2), lag_flag=tNcc < LAG_MIN_NCC, anchor_from_other_demo=tAncOther,
                          anchor_dt_s=round(float(tT[tK[0]] - tT[tAnc[0]]) / NS_PER_S, 1), **tOInfo))
    tMedian = np.median(np.array(tAllOffsets), 0)
    for tRow, tO in zip(tInfo, tAllOffsets):
        tRow["offset_dev_from_pooled_mm"] = round(float(np.linalg.norm(tO - tMedian)) * MM_PER_M, 1)
    tFusedValid = (tSrc == SRC["ORBBEC"]) & tDemoValid
    tOut = dict(prodTraj)
    tOut.update(p_B=tP, R_B_tcp=tRt.as_matrix(), fused_valid=tFusedValid, gap_len_s=tGapLen, src=tSrc, conf=tConf,
                unresolved=tUnres, tag_in_gap=tTagInGap, demo_valid=tDemoValid, p_orbbec_body=tC, npts=tNpts,
                p_B_prod=prodTraj["p_B"], R_B_tcp_prod=prodTraj["R_B_tcp"], pose_src=np.array(POSE_SRC),
                cand_info=np.array(json.dumps(tInfo)))
    return tOut, tConf, tInfo


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--takes", nargs="+", required=True)
    tAp.add_argument("--derived-root", required=True)
    tAp.add_argument("--conf-dir", required=True)
    tAp.add_argument("--track", default="trackv2b")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tInfos = {}
    for tTake in tArgs.takes:
        tSrcDir = paths.DERIVED / f"{tTake}_fused" if (paths.DERIVED / f"{tTake}_fused").exists() else paths.DERIVED / tTake
        tDst = Path(tArgs.derived_root).expanduser() / f"{tTake}_fused"
        tDst.mkdir(parents=True, exist_ok=True)
        tAsg = json.loads((tSrcDir / "assignment.json").read_text())
        tAsg["pose_src"] = POSE_SRC
        tAsg["candidate_from"] = str(tSrcDir)
        (tDst / "assignment.json").write_text(json.dumps(tAsg, indent=1, ensure_ascii=False))
        shutil.copy(tSrcDir / "segments.json", tDst / "segments.json")
        tTrack = np.load(workdirs.trackPath(tArgs.track, tTake))
        for tArm in ("left", "right"):
            tProd = dict(np.load(tSrcDir / f"umi_traj_{tArm}.npz", allow_pickle=True))
            if str(tProd["source_rig_label"]) != tAsg["rig_of_arm"][tArm]:
                raise CandidateError(f"{tTake} {tArm}: production rig {tProd['source_rig_label']} != assignment {tAsg['rig_of_arm'][tArm]}")
            tOut, tConf, tInfo = buildArm(tTake, tArm, tTrack, tProd)
            np.savez(tDst / f"umi_traj_{tArm}.npz", **tOut)
            tInfos.setdefault(tTake, {})[tArm] = tInfo
            tConfDir = Path(tArgs.conf_dir).expanduser() / tTake
            tConfDir.mkdir(parents=True, exist_ok=True)
            np.savez(tConfDir / f"{tArm}.npz", t_ns=tOut["t_ns"], conf=CONF_FILE_VALUES[tConf])
            print(tTake, tArm, "src ORBBEC/TAG/INTERP %", [round(float((tOut["src"] == tS).mean()) * 100, 1) for tS in (0, 1, 2)],
                  "invalid demos", [tR["demo"] for tR in tInfo if not tR["demo_valid"]], "lags",
                  [tR.get("lag_ms") for tR in tInfo], flush=True)
    tRoot = Path(tArgs.derived_root).expanduser()
    tFile = tRoot / "candidate_info.json"
    tOld = json.loads(tFile.read_text()) if tFile.exists() else {}
    tOld.update(tInfos)
    tFile.write_text(json.dumps(tOld, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
