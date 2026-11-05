"""Commanded-pose profiles of the MIT bridge control tick: the synchronised, acceleration-limited move toward the goal and the
segment-tracking variant used by goal interpolation and PLAN playback. Both advance `arm.mCmd` by one tick and return the
per-joint velocity (deg/s) that is also sent to the drive as the MIT velocity setpoint. Deploy stage: robot bridge.
"""

from __future__ import annotations
import math

import numpy as np

from .arm_state import ArmState

ARM_DOF = 6
# Below this distance (deg) and with a velocity smaller than one tick of acceleration the command snaps onto the goal.
SETTLE_DEG = 1e-3


def _clampPastGoal(step: np.ndarray, vel: np.ndarray, error: np.ndarray) -> None:
    """Never step past the goal on a joint: past = same direction AND longer. 10-02 fix: the old test
    np.sign(e) * (|step| - |e|) > 0 was TRUE on every tick for e < 0, so any negative move snapped straight to the goal."""
    tOver = (step * error > 0) & (np.abs(step) > np.abs(error))
    step[tOver] = error[tOver]
    vel[tOver] = 0.0


def profileStep(arm: ArmState, vMax: float, aMax: float, dt: float) -> np.ndarray:
    """10-01 user "부드럽고 자연스럽게": all joints move along the straight joint-space line to the goal (start and finish
    together); the speed along it is capped by vMax and by the stopping distance sqrt(2 aMax d), and each joint's velocity
    changes by at most aMax dt per tick. A new goal mid-move keeps the present velocity and turns toward it."""
    tV = arm.mVel
    tV = np.zeros(ARM_DOF) if tV is None else tV
    tE = arm.mGoal - arm.mCmd
    tD = float(np.abs(tE).max())
    if tD < SETTLE_DEG and float(np.abs(tV).max()) < aMax * dt:
        arm.mCmd = arm.mGoal.copy()
        arm.mVel = np.zeros(ARM_DOF)
        return arm.mVel
    tSpeed = min(vMax, math.sqrt(2.0 * aMax * tD)) if tD > 0 else 0.0
    tVDes = tE / tD * tSpeed if tD > 0 else np.zeros(ARM_DOF)
    tV = tV + np.clip(tVDes - tV, -aMax * dt, aMax * dt)
    tStep = tV * dt
    _clampPastGoal(tStep, tV, tE)
    arm.mCmd = arm.mCmd + tStep
    arm.mVel = tV
    return tV


def profileStepInterp(arm: ArmState, vMax: float, aMax: float, dt: float, interpKp: float, now: float) -> np.ndarray:
    """10-02 segment tracking: a reference moves from mSegStart to mSegEnd over mSegT; the command follows it with velocity
    feed-forward + interpKp (1/s) on the error. The stopping-distance cap is taken to the segment END (no early braking)."""
    tV = arm.mVel
    tV = np.zeros(ARM_DOF) if tV is None else tV
    tG0, tG1, tT = arm.mSegStart, arm.mSegEnd, arm.mSegT
    tS = min(1.0, (now - arm.mTSeg) / tT) if tT > 0 else 1.0
    tGi = tG0 + (tG1 - tG0) * tS
    tVSeg = (tG1 - tG0) / tT if (tT > 0 and tS < 1.0) else np.zeros(ARM_DOF)
    tEEnd = tG1 - arm.mCmd
    tDEnd = float(np.abs(tEEnd).max())
    if tS >= 1.0 and tDEnd < SETTLE_DEG and float(np.abs(tV).max()) < aMax * dt:
        arm.mCmd = tG1.copy()
        arm.mVel = np.zeros(ARM_DOF)
        return arm.mVel
    tVDes = tVSeg + interpKp * (tGi - arm.mCmd)
    tCap = min(vMax, math.sqrt(2.0 * aMax * tDEnd)) if tDEnd > 0 else 0.0
    tM = float(np.abs(tVDes).max())
    if tM > tCap > 0:
        tVDes = tVDes * (tCap / tM)
    elif tCap == 0:
        tVDes = np.zeros(ARM_DOF)
    tV = tV + np.clip(tVDes - tV, -aMax * dt, aMax * dt)
    tStep = tV * dt
    _clampPastGoal(tStep, tV, tEEnd)
    arm.mCmd = arm.mCmd + tStep
    arm.mVel = tV
    return tV
