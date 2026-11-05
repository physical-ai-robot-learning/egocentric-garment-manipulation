"""State and chunk algebra of the deploy contract (core.contract): the 30D proprio of the current frame, relative chunk ->
absolute targets, Real-Time Chunking soft-mask weights, and retiming a relative chunk along its own path.

Inputs: 4x4 TCP poses in the reBot base frame, (32, 20) relative chunks. Outputs: state vectors, {side: (32, 4, 4)} targets,
(32,) weights, retimed (32, 20) chunks.
"""

from __future__ import annotations

import numpy as np

from laundry_shared.contracts.action_contract import T_to_pose9, make_T, pose9_to_T

from .contract import CHUNK_ROWS, GRIP_INDEX, POSE_SLICE, ROW_DT_S, SIDES


def buildState(T_L: np.ndarray, T_R: np.ndarray, anchorL: np.ndarray, anchorR: np.ndarray, gL: float, gR: float) -> np.ndarray:
    """Contract proprio of the CURRENT frame with the episode anchor -- umi_to_lerobot.proprio_rows_lr on [anchor, current]."""
    from laundry_shared.umi_to_lerobot import proprio_rows_lr

    tTL = np.stack([anchorL, T_L])
    tTR = np.stack([anchorR, T_R])
    return proprio_rows_lr(tTL, tTR, np.array([gL, gL], float), np.array([gR, gR], float))[1]


def decodeChunk(chunk: np.ndarray, T_L: np.ndarray, T_R: np.ndarray) -> dict:
    """(32, 20) relative chunk + measured T(t) -> absolute targets {side: (32, 4, 4)} = T(t) @ dT_k (contracts.decode_action_chunk)."""
    from laundry_shared.contracts.action_contract import decode_action_chunk

    tDecoded = decode_action_chunk(chunk, T_L, T_R)
    return {"left": tDecoded.T_L, "right": tDecoded.T_R}


def rtcWeights(horizon: int, delay: int, nPrev: int) -> np.ndarray:
    """[10-29] Real-Time Chunking (Black, Galliker, Levine 2025, eq. 5) soft mask over `horizon` rows of the NEW chunk: rows < delay
    (executed while this chunk is being computed) = 1, rows delay .. nPrev-1 (still covered by the previous plan) fade c (e^c - 1) /
    (e - 1), c = (nPrev - i) / (nPrev - delay + 1), rows >= nPrev (beyond the previous plan) = 0."""
    tW = np.zeros(horizon)
    nPrev = int(np.clip(nPrev, 0, horizon))
    delay = int(np.clip(delay, 0, nPrev))
    tW[:delay] = 1.0
    if nPrev > delay:
        tC = (nPrev - np.arange(delay, nPrev)) / (nPrev - delay + 1)
        tW[delay:nPrev] = tC * np.expm1(tC) / np.expm1(1.0)
    return tW


def retime(chunk: np.ndarray, rowDtS: float, timeScale: float) -> np.ndarray:
    """[10-29] relative chunk (32, 20), row j = pose at (j + 1) * rowDtS (row 0 = identity at t = 0) -> rows at k * ROW_DT_S *
    timeScale, k = 1..32 (position lerp, rotation slerp, jaw lerp): slow motion ALONG the predicted path (X-VLA 15 Hz rows, DiT 20 Hz
    rows)."""
    from scipy.spatial.transform import Rotation, Slerp

    tTs = np.r_[0.0, np.arange(1, CHUNK_ROWS + 1) * rowDtS]
    tTq = np.clip(np.arange(1, CHUNK_ROWS + 1) * ROW_DT_S * timeScale, 0.0, tTs[-1])
    tOut = np.array(chunk, np.float64)
    for tSide in SIDES:
        tT = np.stack([np.eye(4)] + [pose9_to_T(tP) for tP in tOut[:, POSE_SLICE[tSide]]])
        tPos = np.stack([np.interp(tTq, tTs, tT[:, 0, 3]), np.interp(tTq, tTs, tT[:, 1, 3]), np.interp(tTq, tTs, tT[:, 2, 3])], 1)
        tRot = Slerp(tTs, Rotation.from_matrix(tT[:, :3, :3]))(tTq).as_matrix()
        tOut[:, POSE_SLICE[tSide]] = np.stack([T_to_pose9(make_T(tRot[tK], tPos[tK])) for tK in range(CHUNK_ROWS)])
        tG = np.r_[chunk[0, GRIP_INDEX[tSide]], np.asarray(chunk)[:, GRIP_INDEX[tSide]]]
        tOut[:, GRIP_INDEX[tSide]] = np.interp(tTq, tTs, tG)
    return tOut
