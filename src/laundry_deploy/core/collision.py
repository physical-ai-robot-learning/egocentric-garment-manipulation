"""Capsule link-collision model of the two B601 arms (no link meshes in the repo): arm-arm clearance, link capsules vs the
table, finger height (deploy safety checks, preview, start pose, READY move).

10-23: the URDF references meshes_b601_gripper/link1-6.STL + gripper STLs, but only the three gripper STLs exist on the Mac
(bh-egocentric/hra/hra_a100/urdf_gripper) and no code path loads them -> link-link collision uses CAPSULES between consecutive
frame origins of the corrected pinocchio model (PinkIkSolver.full: base + joint offsets). Radii are conservative half-widths of
the B601 links (tape / photos, 10-23), fingers = segment finger-frame origin -> finger tip at the TCP plane.

Input: joints {side: (6,) bridge rad}. Output: capsules per side, clearance dict.
"""

from __future__ import annotations

import numpy as np

from .contract import SIDES

CAPSULES = [
    ("link1", "link2", 0.045),
    ("link2", "link3", 0.040),
    ("link3", "link4", 0.035),
    ("link4", "link5", 0.032),
    ("link5", "link6", 0.032),
    ("link6", "gbody", 0.032),
]
# 10-23: "gbody" = midpoint of the two finger-joint frames (jaw base).  NOT gripper_link: with the tcp override (-0.019 m) the
# URDF
# gripper_link origin sits 19 mm BEYOND the real jaw tips, so a link6 -> gripper_link capsule dipped into the table on 48 % of the
# training grasp frames.  Table check = link capsules only; the fingers are covered by robotize_ik cornerLift (finger-corner
# floor).
FINGER_RADIUS_M = 0.012  # was FINGER_R
# the wrist capsule (link6-gbody) against the table: its 32 mm arm-arm radius put 5.4 % of the training grasp frames "under" the
# table -- the real jaw housing is narrower toward the fingers
WRIST_TABLE_RADIUS_M = 0.012  # was TABLE_R_WRIST
FINGER_BACKOFF_M = 0.02  # a finger capsule starts this far behind the finger frame along the tool axis
FINGER_FRAMES = ("gripper_left", "gripper_right")
WRIST_CAPSULE = "link6-gbody"
FRAME_NAMES = ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_link", "gripper_left", "gripper_right", "tcp")
DEFAULT_TABLE_Z_M = 0.003  # robotize_ik table plane
DEFAULT_ARM_CLEAR_M = 0.010  # deploy --arm-clear default
SEG_EPS = 1e-12  # degenerate segment / parallel test of segDist


class Collision:
    def __init__(self, arms: dict) -> None:
        import pinocchio as pin

        tPk = arms["right"].mPk
        self.mPin = pin
        self.mModel = tPk.full
        self.mData = self.mModel.createData()
        self.mQ0 = tPk.q_full_pin.copy()
        self.mQidx = tPk.qidx
        self.mJoff = tPk.joff
        self.mFrameIds = {
            f"{tSide}_{tName}": self.mModel.getFrameId(f"{tSide}_{tName}") for tSide in SIDES for tName in FRAME_NAMES
        }

    def capsules(self, q: dict) -> dict:
        """q {side: (6,) bridge rad} -> {side: [(a, b, r, name)]}"""
        tQ = self.mQ0.copy()
        for tSide in SIDES:
            tQ[self.mQidx[tSide]] = np.asarray(q[tSide], np.float64) + self.mJoff[tSide]
        self.mPin.framesForwardKinematics(self.mModel, self.mData, tQ)
        tP = {tKey: self.mData.oMf[tId].translation.copy() for tKey, tId in self.mFrameIds.items()}
        for tSide in SIDES:
            tP[f"{tSide}_gbody"] = (tP[f"{tSide}_gripper_left"] + tP[f"{tSide}_gripper_right"]) / 2
        tOut = {}
        for tSide in SIDES:
            tCaps = [(tP[f"{tSide}_{tA}"], tP[f"{tSide}_{tB}"], tR, f"{tA}-{tB}") for tA, tB, tR in CAPSULES]
            tTcp = self.mData.oMf[self.mFrameIds[f"{tSide}_tcp"]]
            tAxis = tTcp.rotation[:, 2]
            tTcpPos = tTcp.translation
            for tFinger in FINGER_FRAMES:
                tPf = tP[f"{tSide}_{tFinger}"]
                tLateral = (tPf - tTcpPos) - tAxis * float(tAxis @ (tPf - tTcpPos))  # finger offset across the tool axis
                tCaps.append((tPf - FINGER_BACKOFF_M * tAxis, tTcpPos + tLateral, FINGER_RADIUS_M, tFinger))
            tOut[tSide] = tCaps
        return tOut

    def check(
        self, q: dict, tableZ: float = DEFAULT_TABLE_Z_M, armClear: float = DEFAULT_ARM_CLEAR_M, tableClear: float = 0.0
    ) -> dict:
        """-> dict(arm_clear_m, pair, table_clear_m (non-finger capsules), finger_low_m, ok_arm, ok_table)"""
        tCaps = self.capsules(q)
        tBest = (np.inf, None)
        for tA0, tB0, tR0, tN0 in tCaps["left"]:
            for tA1, tB1, tR1, tN1 in tCaps["right"]:
                tD = segDist(tA0, tB0, tA1, tB1) - tR0 - tR1
                if tD < tBest[0]:
                    tBest = (tD, f"L.{tN0} / R.{tN1}")
        # table: link capsules (the wrist capsule with WRIST_TABLE_RADIUS_M); fingers -> cornerLift limiter
        tTable = min(
            min(tA[2], tB[2]) - (WRIST_TABLE_RADIUS_M if tName == WRIST_CAPSULE else tR) - tableZ
            for tSide in SIDES
            for tA, tB, tR, tName in tCaps[tSide]
            if not tName.startswith("gripper_")
        )
        tFingerLow = min(
            min(tA[2], tB[2]) - tR - tableZ
            for tSide in SIDES
            for tA, tB, tR, tName in tCaps[tSide]
            if tName.startswith("gripper_")
        )
        return dict(
            arm_clear_m=float(tBest[0]),
            pair=tBest[1],
            table_clear_m=float(tTable),
            finger_low_m=float(tFingerLow),
            ok_arm=bool(tBest[0] >= armClear),
            ok_table=bool(tTable >= tableClear),
        )


def segDist(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    tD1, tD2, tR = q1 - p1, q2 - p2, p1 - p2
    tA, tE, tF = tD1 @ tD1, tD2 @ tD2, tD2 @ tR
    if tA <= SEG_EPS and tE <= SEG_EPS:
        return float(np.linalg.norm(tR))
    if tA <= SEG_EPS:
        tS, tT = 0.0, np.clip(tF / tE, 0, 1)
    else:
        tC = tD1 @ tR
        if tE <= SEG_EPS:
            tT, tS = 0.0, np.clip(-tC / tA, 0, 1)
        else:
            tB = tD1 @ tD2
            tDen = tA * tE - tB * tB
            tS = np.clip((tB * tF - tC * tE) / tDen, 0, 1) if tDen > SEG_EPS else 0.0
            tT = (tB * tS + tF) / tE
            if tT < 0:
                tT, tS = 0.0, np.clip(-tC / tA, 0, 1)
            elif tT > 1:
                tT, tS = 1.0, np.clip((tB - tC) / tA, 0, 1)
    return float(np.linalg.norm((p1 + tD1 * tS) - (p2 + tD2 * tT)))
