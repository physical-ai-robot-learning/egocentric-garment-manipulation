"""[2026-11-01] Robot <-> training-frame height calibration: the stored file and its evaluation (deploy stage).

user "off를 임의로 주기보다 로봇에 맞추어서" + "왼쪽 에이프릴 태그로 면을 맞추자": the training labels live in the B frame defined by the
table tags 400/401 (configs/calibration/start_anchors_v003_70mm.json: centres at z = 0.001 = table plane, measured with a
PROVISIONAL camera). The robot measures them (tools.height_calib records the touches): per arm dz = TCP z - anchor z (tags: also
dx, dy) and a table plane z = a x + b y + c from all touches of that arm. The deploy's "calib" height mode adds dz(x, y) of the arm,
evaluated at the arm's TCP, instead of a hand-typed z offset. Not a limit -- a frame correction.

Files: CALIB_FILE (height_calib.json, older kept as .prev) and PENDING_FILE (one-touch recordings) in TAGRIG_STATE_DIR.
"""

from __future__ import annotations
import json

import numpy as np

from laundry_common import paths as egopaths

from .contract import SIDES

CALIB_FILE = egopaths.TAGRIG_STATE_DIR / "height_calib.json"  # was OUT
PENDING_FILE = egopaths.TAGRIG_STATE_DIR / "height_calib_pending.json"  # was PENDING
ANCHORS_FILE = egopaths.DATA_CALIB / "start_anchors_v003_70mm.json"  # was ANCHORS
TAG_TABLE_Z_M = 0.001  # anchors file: tags lie on the table -> z = 0.001 m in B (was TABLE_Z)
TABLE_TARGET = "table"
MIN_PLANE_POINTS = 3  # fewer touches (or collinear ones) -> a flat plane at the mean height
PLANE_RANK = 2
MM_PER_M = 1e3


def fitCalibration(points: list) -> dict:
    """Per arm: tag offsets (median of touches on that tag) + table plane (least squares; < 3 points or collinear -> flat at the
    mean). Was fit()."""
    tOut = {}
    for tSide in SIDES:
        tPts = [tP for tP in points if tP["arm"] == tSide]
        if not tPts:
            continue
        tTags = {}
        for tTagId in sorted({tP["target"] for tP in tPts if tP["target"] != TABLE_TARGET}):
            tD = np.array([np.subtract(tP["p_robot"], tP["p_expected"]) for tP in tPts if tP["target"] == tTagId])
            tTags[tTagId] = dict(n=len(tD), d_mm=np.round(np.median(tD, 0) * MM_PER_M, 1).tolist())
        tX = np.array([tP["p_robot"] for tP in tPts])
        tDz = tX[:, 2] - TAG_TABLE_Z_M
        if len(tX) >= MIN_PLANE_POINTS and np.linalg.matrix_rank(np.c_[tX[:, :2] - tX[:, :2].mean(0)]) == PLANE_RANK:
            tA = np.c_[tX[:, :2], np.ones(len(tX))]
            tCoef, *_ = np.linalg.lstsq(tA, tDz, rcond=None)
            tRes = tDz - tA @ tCoef
            tTilt = float(np.degrees(np.arctan(np.hypot(*tCoef[:2]))))
        else:
            tCoef = np.array([0.0, 0.0, float(np.mean(tDz))])
            tRes = tDz - tCoef[2]
            tTilt = None
        tOut[tSide] = dict(
            n=len(tPts),
            tags=tTags,
            plane_dz=dict(a=float(tCoef[0]), b=float(tCoef[1]), c=float(tCoef[2])),
            dz_mean_mm=round(float(np.mean(tDz)) * MM_PER_M, 1),
            resid_mm=round(float(np.abs(tRes).max()) * MM_PER_M, 1),
            tilt_deg=None if tTilt is None else round(tTilt, 2),
        )
    return tOut


def dzAt(cal: dict | None, side: str, xy) -> float:
    """Height error of the training frame at TCP xy (m): robot table = data table + dz -> the deploy adds +dz to the commanded z."""
    tFit = (cal or {}).get("fit", {}).get(side)
    if not tFit:
        return 0.0
    tPlane = tFit["plane_dz"]
    return tPlane["a"] * float(xy[0]) + tPlane["b"] * float(xy[1]) + tPlane["c"]


def toBaseFrame(cal: dict | None, side: str, points) -> np.ndarray:
    """[11-01] robot-FK points (N, 3) -> training B frame: + per-arm xy shift (camera) and - table-plane dz(x, y) (UI overlay).
    Was to_B()."""
    tP = np.array(points, np.float64, ndmin=2)
    tFit = (cal or {}).get("fit", {}).get(side)
    if not tFit:
        return tP
    tOut = tP.copy()
    tOut[:, :2] += np.asarray(tFit.get("xy_cam_minus_robot", [0.0, 0.0]))  # overlay: the CURRENT camera (it moved since 10-26)
    tOut[:, 2] -= np.array([dzAt(cal, side, tX[:2]) for tX in tP])
    return tOut


def loadCalibration() -> dict | None:
    """height_calib.json, or None when the robot was never measured. Was load()."""
    return json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else None
