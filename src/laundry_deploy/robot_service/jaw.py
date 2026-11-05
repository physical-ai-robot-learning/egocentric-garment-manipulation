"""Jaw and joint frame mapping of the MIT bridge: the robot_service 14-vector (LEADER frame, jaw command 0..45) -> per-side raw
motor degrees and the plugin's normalised jaw, and the normalised jaw read back -> physical jaw degrees.

Jaw: raw = cmd * -6 deg in [-270, 0] -> the colleague's normalised gripper (0 = open = -270 deg, 1 = closed = 0 deg).
Deploy stage: robot bridge.
"""

from __future__ import annotations

import numpy as np

from . import config


def rawJawToNorm(rawDeg: float, side: str | None = None) -> float:
    """Command raw jaw deg (<= 0) -> the plugin's norm. 11-02: with a close extra CE the plugin's span is open..+CE; open targets
    keep their physical angle and only the last GRIP_SQUEEZE_RAMP_DEG ramp into the squeeze (0 deg -> +CE). Before, the whole
    range was stretched, so an open command opened ~20 deg less (user: "이젠 그리퍼를 열지를 못해")."""
    tCe = config.GRIP_CLOSE_EXTRA.get(side, 0.0) if side else 0.0
    tRaw = min(config.GRIP_CLOSE_DEG, float(rawDeg))
    tPhys = tRaw + tCe * max(0.0, 1.0 + tRaw / config.GRIP_SQUEEZE_RAMP_DEG)
    return float(np.clip((tPhys - config.GRIP_OPEN_DEG) / (config.GRIP_CLOSE_DEG + tCe - config.GRIP_OPEN_DEG), 0.0, 1.0))


def normToRawJaw(norm: float, side: str | None = None) -> float:
    """11-02: with MIT_GRIP_CLOSE_DEG_<SIDE> the plugin's norm spans open..+close_deg -> report the physical jaw deg, capped at
    0 (closed): a squeezed-shut jaw read back as ~-20 deg before, and a stop / READY re-sent that reading -> the jaw opened."""
    tCe = config.GRIP_CLOSE_EXTRA.get(side, 0.0) if side else 0.0
    tPhys = config.GRIP_OPEN_DEG + float(norm) * (config.GRIP_CLOSE_DEG + tCe - config.GRIP_OPEN_DEG)
    if tCe > 0 and tPhys > -config.GRIP_SQUEEZE_RAMP_DEG:  # inverse of rawJawToNorm's squeeze ramp
        tPhys = (tPhys - tCe) / (1.0 + tCe / config.GRIP_SQUEEZE_RAMP_DEG)
    return min(config.GRIP_CLOSE_DEG, tPhys)


def leaderVecToGoals(action) -> dict:
    """robot_service /execute_step vector -> {side: (raw arm deg (6,), jaw norm)} for every connected side."""
    tOut = {}
    for tSide in config.SIDES:
        tOffset = config.ALL_SIDES.index(tSide) * config.JOINTS_PER_SIDE
        tArmLeaderDeg = np.degrees(np.asarray(action[tOffset : tOffset + config.ARM_DOF], float))
        tRaw = np.clip(tArmLeaderDeg * config.SEEED_DIR, config.SEEED_LIM_DEG[:, 0], config.SEEED_LIM_DEG[:, 1])
        tFrac = config.GRIP_OPEN_FRAC[tSide]
        tCmd = float(action[tOffset + config.ARM_DOF])
        tJawRaw = float(np.clip(tCmd * config.GRIP_CMD_SCALE * tFrac, config.GRIP_OPEN_DEG * tFrac, config.GRIP_CLOSE_DEG))
        tOut[tSide] = (tRaw, rawJawToNorm(tJawRaw, tSide))
    return tOut
