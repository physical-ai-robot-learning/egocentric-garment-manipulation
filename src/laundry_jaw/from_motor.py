"""HandUMI jaw label from the Feetech jaw servo, the jaw source of every demo since 2026-10-29 (was scripts/pose/jaw_from_motor.py).

Pipeline stage: jaw state (stage 2). Input: a take recorded by tag_rig_record with gripper_motor_{left,right}.npz (100 Hz, torque
off). Output: <take>/<wrist dir>/umi_jaw.npz in the format the dataset builders read (umi_jaw infer + jaw_aperture): t_ns, g_cont
(0 closed .. 1 open, continuous), p_open (= g_cont), state (1 open / 0 closed, hysteresis + dwell), valid, ev_t / ev_kind,
g_is_aperture = 1, the continuous motor position as motor_pos, and the calibration used as cal (json).

Sides: wrist = RIGHT hand (camera 0x03110, motor 5B3D0480821), wrist_left = LEFT hand (0x03140, 5B3D0497471), from the 10-29 15:57
physical test. Fixed per-side calibration from the 7 takes of 10-29 evening (pooled p2 / p98 of the motor position):
  left  closed -850 (jaws together; closed ON the cloth reads -700..-800) .. open +200 (rises with the jaw opening)
  right closed 1830 .. open 1340 (mirrored: falls with the jaw opening; the right jaw only reached ~1340-1450, stiff linkage)
The left/right direction comes from that calibration: (pos - closed) / (open - closed) is positive towards open on both sides.

    python -m laundry_jaw.from_motor <take_dir> [...] [--hi 0.55 --lo 0.35 --dwell 0.1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# (closed, open) motor ticks per hand
SERVO_CALIBRATION = {"left": (-850.0, 200.0), "right": (1830.0, 1340.0)}
WRIST_DIRS = {"right": "wrist", "left": "wrist_left"}
# A re-plug / power cycle moves the whole reading (10-30: +2150 ticks); normal closed drift is < 50 ticks.
SHIFT_TOLERANCE_TICKS = 300.0
TAIL_PERCENTILES = (50, 1, 99)     # median (rest = closed) and the two tails; the open side is the long tail
NS_PER_S = 1e9
MOTOR_MODEL = "feetech_sts3215_motor"
DEFAULT_HI = 0.55
DEFAULT_LO = 0.35
DEFAULT_DWELL_S = 0.1


def hysteresis(tNs: np.ndarray, g: np.ndarray, hi: float, lo: float, dwellS: float) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """(state per sample: 1 open / 0 closed, events [(t_ns, 'open'|'close')]): a change needs g past `hi` / `lo` for `dwellS`;
    the state is back-filled from the first sample of the change."""
    tState = np.zeros(len(g), np.int64)
    tCurrent = int(g[0] > (hi + lo) / 2)
    tCandidate = None
    tEvents = []
    for tI, tValue in enumerate(g):
        tWant = 1 if tValue > hi else 0 if tValue < lo else tCurrent
        if tWant != tCurrent:
            if tCandidate is None:
                tCandidate = tI
            if (tNs[tI] - tNs[tCandidate]) / NS_PER_S >= dwellS:
                tCurrent = tWant
                tState[tCandidate:tI] = tCurrent
                tEvents.append((int(tNs[tCandidate]), "open" if tCurrent else "close"))
                tCandidate = None
        else:
            tCandidate = None
        tState[tI] = tCurrent
    return tState, tEvents


def labelHand(take: Path, side: str, hi: float, lo: float, dwellS: float) -> None:
    """Write <take>/<wrist dir of `side`>/umi_jaw.npz from gripper_motor_<side>.npz."""
    with np.load(take / f"gripper_motor_{side}.npz") as tMotor:
        tT = tMotor["host_ns"].astype(np.int64)
        tPos = tMotor["pos"].astype(np.float64)
    tClosed, tOpen = SERVO_CALIBRATION[side]
    # 10-30 take 172433: the right servo was re-plugged and its whole reading moved by ~+2150 ticks (closed 1830 -> ~3985, same
    # ~450-tick stroke and direction). The jaw rests CLOSED most of a take, so the take median is the closed reading: if it is
    # more than SHIFT_TOLERANCE_TICKS away from the calibrated closed value, the calibration is shifted by that offset for this take.
    tShift = float(np.median(tPos)) - tClosed
    if abs(tShift) > SHIFT_TOLERANCE_TICKS:
        tClosed, tOpen = tClosed + tShift, tOpen + tShift
        print(f"{take.name} {side}: servo reading shifted by {tShift:+.0f} ticks -> cal closed {tClosed:.0f} open {tOpen:.0f}")
    else:
        tShift = 0.0
    # 11-04 12:xx: the LEFT jaw was re-assembled between takes 105052 and 123741 -> its reading now FALLS when opening (was rising)
    # with a ~600-tick stroke. The jaw rests closed (median), so the open side is the side with the long tail: if that side is
    # opposite to the calibrated direction, take open = the 1st / 99th percentile on the observed side for this take.
    tMed, tP1, tP99 = np.percentile(tPos, TAIL_PERCENTILES)
    tObserved = 1.0 if (tP99 - tMed) > (tMed - tP1) else -1.0
    if tObserved != np.sign(tOpen - tClosed):
        tOpen = float(tP99 if tObserved > 0 else tP1)
        print(f"{take.name} {side}: jaw direction FLIPPED vs calibration -> cal closed {tClosed:.0f} open {tOpen:.0f} "
              f"(stroke {abs(tOpen - tClosed):.0f})")
    tG = np.clip((tPos - tClosed) / (tOpen - tClosed), 0.0, 1.0).astype(np.float32)
    tState, tEvents = hysteresis(tT, tG, hi, lo, dwellS)
    tOut = take / WRIST_DIRS[side]
    tOut.mkdir(exist_ok=True)
    np.savez(tOut / "umi_jaw.npz", t_ns=tT, g_cont=tG, p_open=tG, state=tState, valid=np.ones(len(tT), np.int8), hand=side,
             model=MOTOR_MODEL, g_is_aperture=np.int8(1), motor_pos=tPos.astype(np.float32),
             ev_t=np.array([tEvent[0] for tEvent in tEvents], np.int64), ev_kind=np.array([tEvent[1] for tEvent in tEvents]),
             cal=json.dumps(dict(closed=tClosed, open=tOpen, shift=tShift, hi=hi, lo=lo, dwell_s=dwellS)))
    tOpens = sum(tEvent[1] == "open" for tEvent in tEvents)
    tCloses = sum(tEvent[1] == "close" for tEvent in tEvents)
    print(f"{take.name} {side}: {len(tT)} samples, open frac {tState.mean():.2f}, events {len(tEvents)} "
          f"({tOpens} open / {tCloses} close)", flush=True)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("takes", nargs="+")
    tParser.add_argument("--hi", type=float, default=DEFAULT_HI)
    tParser.add_argument("--lo", type=float, default=DEFAULT_LO)
    tParser.add_argument("--dwell", type=float, default=DEFAULT_DWELL_S)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    for tTake in map(Path, tArgs.takes):
        for tSide in WRIST_DIRS:
            labelHand(tTake, tSide, tArgs.hi, tArgs.lo, tArgs.dwell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
