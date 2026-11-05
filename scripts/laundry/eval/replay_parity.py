"""UMI -> robot replay parity (2026-10-20, GPT plan step 6; was scripts/evaluation/replay_parity.py).

Pipeline stage: D7 comparison of the replay track. Per outward stroke (hold before -> hold after, displacements relative to the stroke
start, robot base frame): A = UMI measured (umi_tcp_traj p_B), B = robot commanded (FK of the replayed q), C = robot executed (FK of
the observed q). Compared on NORMALIZED PROGRESS s in [0,1] (projection on A's chord, each series normalised by its own length,
monotone), so lag / speed differences do not count as error.
Per stroke: main-axis ratio C/A (and B/A), cross-axis leakage |perp(C(s)-A(s))| / |A main|, endpoint |dC-dA|, RMSE / p95 |C(s)-A(s)|,
B-A (transform + IK) and C-B (robot tracking) split, A's own straightness and angle to the robot axis (human, informational).
Per direction: 3D endpoint repeat std of A and C over the repeats.
Gates: ratio 0.9-1.1, leakage < 10% (better < 5%), endpoint < 20 mm, RMSE < 20 mm, p95 < 30 mm, repeat std C < 10 mm.
Inputs: <take>/umi_traj_<side>[<traj>].npz + segments.json (or <take>_derived/), <run>/meta.json + seg_NN.npz; output: printed
tables (Korean), free_NN.png plots for free-motion runs; exit status 1 when the --synthetic self-test fails.
    python scripts/laundry/eval/replay_parity.py datasets/tag_rig/<take> datasets/replay/<run>        # A/B/C
    python scripts/laundry/eval/replay_parity.py datasets/tag_rig/<take> --umi-only [--hand 오른손]     # A only (before the robot)
    python scripts/laundry/eval/replay_parity.py --synthetic                                            # self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
# 10-20 --proto rot: rotation about the robot-base axis through the jaw tip
AXR = {"YAW": AX["Z"], "PITCH": AX["Y"], "ROLL": AX["X"]}
# 10-20 GPT gates for rotation: ratio 0.9-1.1, geodesic median < 5 deg, p95 < 10 deg, unwanted axes < 10 %, fake TCP translation
# < 10 mm, repeat < 3 deg. ("mm" fields hold DEGREES for rotation strokes.)
GATE_ROT = dict(ratio=(0.9, 1.1), leak=0.10, end_mm=5.0, med_deg=5.0, p95_mm=10.0, fake_mm=10.0, rep_deg=3.0, sat_deg_s=40.0)
# gripper-local meaning at READY (R_B_E: x_E ~ base +Z up, y_E ~ base -Y, z_E = approach ~ base +X), user/GPT 10-20: record the
# physical axis
PHYS = {"YAW": "sweep 좌우 (x_E 축)", "PITCH": "bend 상하 (y_E 축)", "ROLL": "twist 집게축 (z_E 축)"}
# normalised progress grid every stroke is resampled on
S = np.linspace(0, 1, 101)
GATE = dict(ratio=(0.9, 1.1), leak=0.10, end_mm=20, rmse_mm=20, p95_mm=30, rep_mm=10)
GATE_FREE = dict(pos_rmse=20.0, pos_p95=30.0, rot_med=5.0, rot_p95=10.0, ratio=(0.9, 1.1), jumps=0)
# GPT 10-20: phase error < 5 % (target 2-3 %), actual time < 100 ms, missed 0, spurious 0
GATE_GRIP = dict(phase=0.05, ms=100.0)

NS_PER_S = 1e9
NS_PER_MS = 1e6
MS_PER_S = 1000
MM_PER_M = 1e3
# rotvecs are scaled deg / ROT_AS_MM so stroke()'s "mm" fields read in DEGREES
ROT_AS_MM = 1000
# a chord / progress length below this is treated as this (no division by zero)
MIN_LENGTH = 1e-6
MIN_CHORD_M = 1e-9
# the commanded stroke gives the lag-search direction when it moved at least this far
MIN_STROKE_DIR_M = 1e-3
MIN_STROKE_SAMPLES = 5
MIN_HOLD_SAMPLES = 3
MIN_PROGRESS_SAMPLES = 3
MIN_LAG_SAMPLES = 5
MIN_FREE_SAMPLES = 20
# lag searches: C(t + lag) against B(t), 20 ms steps up to 1 s (strokes) / 1.5 s (free motion)
LAG_STEP_NS = 2e7
STROKE_MAX_LAG_NS = 1.0e9
FREE_MAX_LAG_NS = 1.5e9
# A stroke counts as rotating (rotation ratio gate) above this human rotation amplitude.
FREE_ROT_GATE_MIN_DEG = 10
# Free-motion discontinuity: C speed above max(0.25 m/s, 3 x p99 of B speed).
JUMP_MIN_SPEED_M_S = 0.25
JUMP_SPEED_FACTOR = 3
SPEED_PERCENTILE = 99
SMOOTH_FRAMES = 9
# joint speed of a rotation stroke: p99 over the replay, dt floored at 1 ms
SAT_PERCENTILE = 99
MIN_DT_S = 1e-3
# gripper: encoder g_obs crosses 0.5 with 0.35 / 0.65 hysteresis; a matched event lies within 0.15 of the UMI phase
GRIP_LEVEL = 0.5
GRIP_HI = 0.65
GRIP_LO = 0.35
GRIP_MATCH_PHASE = 0.15
# tags seen on at least half of a sample's interpolation
TAGGED_MIN = 0.5
P95 = 95
PERCENT = 100
HAND_SIDES = {"오른손": "right", "왼손": "left"}
PLOT_DPI = 90
SYNTHETIC_SEED = 3


def onProgress(points, direction):
    """Displacements `points` (n,3) relative to their start, resampled at S by the monotone projection on `direction`, normalised
    by its own end; None with fewer than MIN_PROGRESS_SAMPLES distinct progress values."""
    points = points - points[0]
    tX = points @ direction
    tLength = tX[-1] if abs(tX[-1]) > MIN_LENGTH else MIN_LENGTH
    tS = np.maximum.accumulate(np.clip(tX / tLength, 0, None))
    tS, tIdx = np.unique(tS, return_index=True)
    points = points[tIdx]
    if len(tS) < MIN_PROGRESS_SAMPLES:
        return None
    return np.stack([np.interp(S, tS, points[:, tJ]) for tJ in range(3)], 1)


def perp(v, e):
    return v - np.outer(v @ e, e)


def stroke(pathA, axis, pathB=None, pathC=None):
    """Metrics of one stroke along the unit `axis`: A's own, and B / C against A on normalised progress (None without progress)."""
    tU = pathA[-1] - pathA[0]
    tU = tU / max(np.linalg.norm(tU), MIN_CHORD_M)
    tA = onProgress(pathA, tU)
    tR = {}
    if tA is None:
        return None
    tMainA = float((pathA[-1] - pathA[0]) @ axis)
    tR.update(main_A_mm=tMainA * MM_PER_M, leak_A=float(np.linalg.norm(perp(tA, axis), axis=1).max() / abs(tMainA)),
              angle_A_deg=float(np.degrees(np.arccos(np.clip(abs(tU @ axis), 0, 1)))), end_A=(pathA[-1] - pathA[0]))
    for tName, tSeries in (("B", pathB), ("C", pathC)):
        if tSeries is None:
            continue
        tX = onProgress(tSeries, tU)
        if tX is None:
            continue
        tD = np.linalg.norm(tX - tA, axis=1) * MM_PER_M
        tR.update({f"ratio_{tName}": float((tSeries[-1] - tSeries[0]) @ axis / tMainA),
                   f"leak_{tName}": float(np.linalg.norm(perp(tX - tA, axis), axis=1).max() / abs(tMainA)),
                   f"end_{tName}_mm": float(np.linalg.norm((tSeries[-1] - tSeries[0]) - (pathA[-1] - pathA[0])) * MM_PER_M),
                   f"rmse_{tName}_mm": float(np.sqrt((tD ** 2).mean())),
                   f"p95_{tName}_mm": float(np.percentile(tD, P95)), f"end_{tName}": tSeries[-1] - tSeries[0]})
        if tName == "C" and "ratio_B" in tR:
            tR["rmse_CB_mm"] = float(np.sqrt((np.linalg.norm(tX - onProgress(pathB, tU), axis=1) ** 2).mean()) * MM_PER_M)
    return tR


def judge(result, rot=False):
    """Names of the failed gates (Korean, as printed)."""
    tFailed = []
    if rot:
        tGate = GATE_ROT
        if not tGate["ratio"][0] <= result["ratio_C"] <= tGate["ratio"][1]:
            tFailed.append("비율")
        if result["leak_C"] >= tGate["leak"]:
            tFailed.append("축혼입")
        if result["end_C_mm"] >= tGate["end_mm"]:
            tFailed.append("끝각도")
        if result["geo_med_deg"] >= tGate["med_deg"] or result["p95_C_mm"] >= tGate["p95_mm"]:
            tFailed.append("경로각도")
        if result.get("fake_mm", 0) >= tGate["fake_mm"]:
            tFailed.append("가짜이동")
        if result.get("sat_deg_s", 0) >= tGate["sat_deg_s"]:
            tFailed.append("포화")
        return tFailed
    tGate = GATE
    if not tGate["ratio"][0] <= result["ratio_C"] <= tGate["ratio"][1]:
        tFailed.append("비율")
    if result["leak_C"] >= tGate["leak"]:
        tFailed.append("새어나감")
    if result["end_C_mm"] >= tGate["end_mm"]:
        tFailed.append("끝점")
    if result["rmse_C_mm"] >= tGate["rmse_mm"] or result["p95_C_mm"] >= tGate["p95_mm"]:
        tFailed.append("경로")
    return tFailed


def _isRotation(key: str) -> bool:
    return key.split()[0] in AXR


def summarize(rows, mode):
    """Print the stroke table and the per-direction repeatability; with B/C (mode != "A") the verdict, returned as bool."""
    if any(_isRotation(tKey) for tKey, _ in rows):
        print("\n회전 축 의미 (READY 그리퍼 기준): " + ", ".join(f"{tKey} = {tValue}" for tKey, tValue in PHYS.items())
              + "\n회전 행: A 이동 = 사람 회전각, 끝점/RMSE/p95/C-B = 도(°), 새어남 = 다른 축 회전 비율")
    tHeader = f"\n{'구간':14s} {'A 이동':>7s} {'A 각도':>6s} {'A 휨':>5s}"
    if mode != "A":
        tHeader += f" {'C/A':>5s} {'B/A':>5s} {'새어남':>6s} {'끝점':>6s} {'RMSE':>6s} {'p95':>6s} {'C-B':>6s}  판정"
    print(tHeader)
    tFails = 0
    for tKey, tR in rows:
        tLine = f"{tKey:14s} {tR['main_A_mm']:6.0f}mm {tR['angle_A_deg']:5.1f}° {tR['leak_A'] * PERCENT:4.0f}%"
        tRot = _isRotation(tKey)
        tUnit = "°" if tRot else "mm"
        if tRot:
            tLine = f"{tKey:14s} {tR['main_A_mm']:6.1f}° {tR['angle_A_deg']:5.1f}° {tR['leak_A'] * PERCENT:4.0f}%"
        if mode != "A" and "ratio_C" in tR:
            tFailed = judge(tR, tRot)
            tFails += bool(tFailed)
            tLine += (f" {tR['ratio_C']:5.2f} {tR.get('ratio_B', np.nan):5.2f} {tR['leak_C'] * PERCENT:5.1f}% {tR['end_C_mm']:5.1f}{tUnit} "
                      f"{tR['rmse_C_mm']:5.1f}{tUnit} {tR['p95_C_mm']:5.1f}{tUnit} "
                      f"{tR.get('rmse_CB_mm', np.nan):5.1f}{tUnit}  " + ("PASS" if not tFailed else "FAIL(" + ",".join(tFailed) + ")"))
            if tRot:
                tLine += (f" | 각오차 중앙 {tR['geo_med_deg']:.1f}° p95 {tR['p95_C_mm']:.1f}° | 가짜이동 최대 {tR['fake_mm']:.0f} mm "
                          f"(중앙 {tR['fake_med_mm']:.0f}) | 사람 끝 흔들림 {tR['tip_A_mm']:.0f} mm | 관절속도 p99 {tR['sat_deg_s']:.0f}°/s")
        print(tLine)
    print("\n방향별 반복 (끝점 3D 표준편차; A = 사람 동작 편차, 참고):")
    tGroups = {}
    for tKey, tR in rows:
        tGroups.setdefault(tKey.rsplit(" ", 1)[0], []).append(tR)
    tRepeatBad = 0
    for tGroup, tRows in tGroups.items():
        if len(tRows) < 2:
            continue
        tSpreadA = float(np.sqrt(np.trace(np.cov(np.array([tX["end_A"] for tX in tRows]).T))) * MM_PER_M)
        tUnit = "°" if _isRotation(tGroup) else "mm"
        tMessage = f"  {tGroup:12s} 반복 {len(tRows)}회: A {tSpreadA:5.1f} {tUnit}"
        if mode != "A" and all("end_C" in tX for tX in tRows):
            # 10-20: each repeat replays a DIFFERENT human stroke, so C's own spread just follows A's; the robot's repeatability is
            # the spread of the endpoint ERROR (C - A) over the repeats
            tSpreadC = float(np.sqrt(np.trace(np.cov(np.array([tX["end_C"] - tX["end_A"] for tX in tRows]).T))) * MM_PER_M)
            tRepeatBad += tSpreadC >= GATE["rep_mm"]
            tLimit = GATE_ROT["rep_deg"] if tUnit == "°" else GATE["rep_mm"]
            tRepeatBad += (tSpreadC >= tLimit) - (tSpreadC >= GATE["rep_mm"])
            tMessage += f", 로봇 끝점오차(C-A) {tSpreadC:5.1f} {tUnit}" + ("" if tSpreadC < tLimit else f"  <- {tLimit:.0f} {tUnit} 이상")
        print(tMessage)
    if mode != "A":
        tAll = [tR for _, tR in rows if "ratio_C" in tR]
        tRmse = np.array([tR["rmse_C_mm"] for tR in tAll])
        print(f"\n전체: 구간 {len(tAll)}개, 통과 {len(tAll) - tFails}, RMSE 중앙 {np.median(tRmse):.1f} mm, 최대 {tRmse.max():.1f} mm, "
              f"반복 기준 넘은 방향 {tRepeatBad}")
        print("최종:", "PASS" if tFails == 0 and tRepeatBad == 0 else "FAIL",
              "| B-A 가 크면 좌표변환/IK 문제, C-B 가 크면 로봇 추종 문제, A 각도가 크면 사람이 로봇 축과 다르게 움직인 것(오류 아님)")
        return tFails == 0 and tRepeatBad == 0
    return True


def windows(segment):
    """(middle of hold_pre, middle of hold_mid) of a segment."""
    tA = (segment["hold_pre"][0] + segment["hold_pre"][1]) // 2
    tB = (segment["hold_mid"][0] + segment["hold_mid"][1]) // 2
    return tA, tB


def selectWindow(times, points, start, end, hold=None):
    """`points` where start <= times <= end (None with fewer than MIN_STROKE_SAMPLES); the first row replaced by the median over
    the `hold` window when given."""
    tMask = (times >= start) & (times <= end)
    if tMask.sum() < MIN_STROKE_SAMPLES:
        return None
    tX = points[tMask].copy()
    if hold is not None:                          # start = median of the hold before the stroke (operator jitter)
        tHold = (times >= hold[0]) & (times <= hold[1])
        if tHold.sum() >= MIN_HOLD_SAMPLES:
            tX[0] = np.median(points[tHold], 0)
    return tX


def _observedOnSourceClock(replay, speed):
    """Observation times of a replay segment mapped onto the UMI source clock (the replay ran at speed `speed`)."""
    tSrc = replay["src_t_ns"].astype(np.int64)
    return tSrc, tSrc[0] + ((replay["t_obs_ns"] - replay["t_cmd_ns"][0]) * speed).astype(np.int64)


def rotStroke(umi, replay, segment, speed, start, end):
    """10-20 rotation stroke: spatial relative rotation (robot base) of A (UMI), B (commanded FK), C (executed FK) on normalised
    progress. Rotvecs are scaled deg/1000 so stroke()'s "mm" fields read in DEGREES. Also the jaw-tip drift during the rotation
    (A vs C)."""
    from scipy.spatial.transform import Rotation
    if "R_B_tcp" not in umi or "rot_cmd" not in replay:
        print(f"  {segment['axis']} {segment['dir']} {segment['rep']}: 회전 데이터 없음 (--orient 재생 + R_B_tcp 필요)")
        return None
    tA = umi["t_ns"].astype(np.int64)
    tSrc, tC = _observedOnSourceClock(replay, speed)
    tHold = (tA >= segment["hold_pre"][0]) & (tA <= segment["hold_pre"][1])
    tMaskA = (tA >= start) & (tA <= end)
    if tMaskA.sum() < MIN_STROKE_SAMPLES:
        return None
    if tHold.sum() >= MIN_HOLD_SAMPLES:
        tR0 = Rotation.from_matrix(umi["R_B_tcp"][tHold]).mean()
    else:
        tR0 = Rotation.from_matrix(umi["R_B_tcp"][tMaskA][0])
    tPathA = (Rotation.from_matrix(umi["R_B_tcp"][tMaskA]) * tR0.inv()).as_rotvec()
    tPathA = np.vstack([np.zeros(3), tPathA])

    def _relative(times, rotvecs, startNs, endNs):
        tMask = (times >= startNs) & (times <= endNs)
        if tMask.sum() < MIN_STROKE_SAMPLES:
            return None, tMask
        tRs = Rotation.from_rotvec(rotvecs[tMask])
        return (tRs * tRs[0].inv()).as_rotvec(), tMask

    tAxis = segment["sign"] * AXR[segment["axis"]]
    tMaskB = (tSrc >= start) & (tSrc <= end)
    tXB = replay["rot_cmd"][tMaskB] @ tAxis
    tXC = replay["rot_obs"] @ tAxis
    if tMaskB.sum() > MIN_LAG_SAMPLES:   # both vs READY
        tLag = min(np.arange(0, STROKE_MAX_LAG_NS, LAG_STEP_NS),
                   key=lambda lagNs: np.mean((np.interp(tSrc[tMaskB] + lagNs, tC, tXC) - tXB) ** 2))
    else:
        tLag = 0
    tPathB, _ = _relative(tSrc, replay["rot_cmd"], start, end)
    tPathC, tMaskC = _relative(tC, replay["rot_obs"], start + int(tLag), end + int(tLag))
    if tPathB is None or tPathC is None:
        return None

    def _asMm(rotvecs):
        return np.degrees(rotvecs) / ROT_AS_MM

    tResult = stroke(_asMm(tPathA), tAxis, _asMm(tPathB), _asMm(tPathC))
    if tResult is None:
        return None
    tU = tPathA[-1] - tPathA[0]
    tU = tU / max(np.linalg.norm(tU), MIN_CHORD_M)
    tAP, tCP = onProgress(tPathA, tU), onProgress(tPathC, tU)
    if tAP is not None and tCP is not None:
        tGeo = np.degrees((Rotation.from_rotvec(tCP) * Rotation.from_rotvec(tAP).inv()).magnitude())
    else:
        tGeo = np.array([np.nan])
    # fake TCP translation: executed tip displacement minus the UMI's own tip displacement, matched in (lag-shifted) time
    tAm = tA[tMaskA]
    tPA = umi["p_B"][tMaskA] - umi["p_B"][tMaskA][0]
    tCm = tC[tMaskC] - int(tLag)
    tPC = replay["p_obs"][tMaskC] - replay["p_obs"][tMaskC][0]
    tPCi = np.stack([np.interp(tAm, tCm, tPC[:, tK]) for tK in range(3)], 1)
    tFake = np.linalg.norm(tPCi - tPA, axis=1) * MM_PER_M
    tQDeg = np.degrees(replay["q_cmd"])
    tDt = np.diff(replay["t_cmd_ns"]) / NS_PER_S
    tSpeed = np.abs(np.diff(tQDeg, axis=0)) / np.maximum(tDt, MIN_DT_S)[:, None]
    tResult.update(geo_med_deg=float(np.nanmedian(tGeo)), p95_C_mm=float(np.nanpercentile(tGeo, P95)), fake_mm=float(tFake.max()),
             fake_med_mm=float(np.median(tFake)), tip_A_mm=float(np.linalg.norm(tPA, axis=1).max() * MM_PER_M),
             sat_deg_s=float(np.percentile(tSpeed.max(1), SAT_PERCENTILE)))
    return tResult


def _interpRows(times, sourceTimes, values):
    return np.stack([np.interp(times, sourceTimes, values[:, tK]) for tK in range(values.shape[1])], 1)


def _plotFree(pathA, pathB, pathC, rotA, rotB, rotC, grid, segment, result, plotPath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})
    tT = (grid - grid[0]) / NS_PER_S
    tFig, tAx = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for tK, tName in enumerate("XYZ"):
        for tX, tLabel, tStyle in ((pathA, "A UMI", "-"), (pathB, "B 명령", "--"), (pathC, "C 실행", ":")):
            tAx[0, tK].plot(tT, tX[:, tK] * MM_PER_M, tStyle, label=tLabel)
        tAx[0, tK].set_title(f"위치 {tName} (mm)")
        for tX, tLabel, tStyle in ((rotA, "A", "-"), (rotB, "B", "--"), (rotC, "C", ":")):
            tAx[1, tK].plot(tT, np.degrees(tX[:, tK]), tStyle, label=tLabel)
        tAx[1, tK].set_title(f"회전 {['ROLL(X)', 'PITCH(Y)', 'YAW(Z)'][tK]} (deg)")
        tAx[1, tK].set_xlabel("s")
    tAx[0, 0].legend()
    tFig.suptitle(f"{segment['hand']} 자유동작 {segment['rep']}회: 위치 RMSE {result['pos_rmse']:.1f} mm, 회전 중앙 {result['rot_med']:.1f}°")
    tFig.tight_layout()
    tFig.savefig(plotPath, dpi=PLOT_DPI)
    plt.close(tFig)


def freeStroke(umi, replay, segment, speed, plotPath=None):
    """10-20 combined free motion (GPT): whole hold->motion->hold window, A = UMI, B = commanded FK, C = executed FK, C time-shifted
    by its fitted lag (3D position), so lag is not error. Position RMSE/p95, orientation geodesic median/p95, translation and
    rotation amplitude ratios (RMS distance from the start pose), path-length ratio, start/end pose error, C-B residual,
    discontinuities, overlay plot."""
    from scipy.spatial.transform import Rotation
    tA = umi["t_ns"].astype(np.int64)
    tSrc, tC0 = _observedOnSourceClock(replay, speed)
    tA0 = (segment["hold_pre"][0] + segment["hold_pre"][1]) // 2
    tA1 = (segment["hold_post"][0] + segment["hold_post"][1]) // 2
    tGrid = tSrc[(tSrc >= tA0) & (tSrc <= tA1)]
    if len(tGrid) < MIN_FREE_SAMPLES:
        return None
    tHold = (tA >= segment["hold_pre"][0]) & (tA <= segment["hold_pre"][1])
    if tHold.sum() >= MIN_HOLD_SAMPLES:
        tPA0 = np.median(umi["p_B"][tHold], 0)
        tRA0 = Rotation.from_matrix(umi["R_B_tcp"][tHold]).mean()
    else:
        tPA0 = umi["p_B"][np.searchsorted(tA, tA0)]
        tRA0 = Rotation.from_matrix(umi["R_B_tcp"][np.searchsorted(tA, tA0)])
    tPathA = _interpRows(tGrid, tA, umi["p_B"]) - tPA0
    tRotA = _interpRows(tGrid, tA, (Rotation.from_matrix(umi["R_B_tcp"]) * tRA0.inv()).as_rotvec())
    tPathB = _interpRows(tGrid, tSrc, replay["p_cmd"])
    tPathB -= tPathB[0]
    tRotB = _interpRows(tGrid, tSrc, replay["rot_cmd"]) if "rot_cmd" in replay else np.zeros_like(tPathA)
    tLag = min(np.arange(0, FREE_MAX_LAG_NS, LAG_STEP_NS),
               key=lambda lagNs: np.mean(np.sum((_interpRows(tGrid + lagNs, tC0, replay["p_obs"]) - _interpRows(tGrid, tSrc, replay["p_cmd"])) ** 2, 1)))
    tPathC = _interpRows(tGrid + tLag, tC0, replay["p_obs"])
    tPathC -= tPathC[0]
    tRotC = _interpRows(tGrid + tLag, tC0, replay["rot_obs"]) if "rot_obs" in replay else np.zeros_like(tPathA)

    def _rel(rotvecs):
        return (Rotation.from_rotvec(rotvecs) * Rotation.from_rotvec(rotvecs[0]).inv()).as_rotvec()

    tRotA, tRotB, tRotC = _rel(tRotA), _rel(tRotB), _rel(tRotC)
    tDp = np.linalg.norm(tPathC - tPathA, axis=1) * MM_PER_M
    tDr = np.degrees((Rotation.from_rotvec(tRotC) * Rotation.from_rotvec(tRotA).inv()).magnitude())

    def _rms(values):
        return float(np.sqrt(np.mean(np.sum(values ** 2, 1))))

    def _pathLength(values):
        return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())

    def _smooth(values, window=SMOOTH_FRAMES):
        return np.stack([np.convolve(np.pad(values[:, tJ], window // 2, mode="edge"), np.ones(window) / window, "valid") for tJ in range(3)], 1)

    tSpeedC = np.linalg.norm(np.diff(tPathC, axis=0), axis=1) / np.diff(tGrid / NS_PER_S)
    tSpeedB = np.linalg.norm(np.diff(tPathB, axis=0), axis=1) / np.diff(tGrid / NS_PER_S)
    tJumps = int(np.sum(tSpeedC > max(JUMP_MIN_SPEED_M_S, JUMP_SPEED_FACTOR * np.percentile(tSpeedB, SPEED_PERCENTILE))))
    if "ntags" in umi:
        tTagged = float(np.mean(np.interp(tGrid, tA, (umi["ntags"] > 0).astype(float)) > TAGGED_MIN))
    else:
        tTagged = 1.0
    tResult = dict(pos_rmse=float(np.sqrt(np.mean(tDp ** 2))), pos_p95=float(np.percentile(tDp, P95)), rot_med=float(np.median(tDr)),
             rot_p95=float(np.percentile(tDr, P95)), tr_ratio=_rms(tPathC) / max(_rms(tPathA), MIN_LENGTH),
             rot_ratio=_rms(tRotC) / max(_rms(tRotA), MIN_LENGTH),
             path_ratio=_pathLength(_smooth(tPathC)) / max(_pathLength(_smooth(tPathA)), MIN_LENGTH),
             end_mm=float(np.linalg.norm(tPathC[-1] - tPathA[-1]) * MM_PER_M), end_deg=float(tDr[-1]),
             cb_rmse=float(np.sqrt(np.mean(np.sum((tPathC - tPathB) ** 2, 1))) * MM_PER_M),
             ba_rmse=float(np.sqrt(np.mean(np.sum((tPathB - tPathA) ** 2, 1))) * MM_PER_M), jumps=tJumps, lag_ms=float(tLag / NS_PER_MS),
             amp_A_mm=float(np.linalg.norm(tPathA, axis=1).max() * MM_PER_M), amp_A_deg=float(np.degrees(np.linalg.norm(tRotA, axis=1).max())),
             tagged=tTagged)
    if plotPath is not None:
        _plotFree(tPathA, tPathB, tPathC, tRotA, tRotB, tRotC, tGrid, segment, tResult, plotPath)
    return tResult


def gripEvents(replay, segment, speed, jawDir):
    """10-20 gripper timing on normalised phase s in [0,1] of the hold->motion->hold window (UMI source time):
    UMI events (umi_jaw infer: hysteresis on the wrist-fisheye classifier) vs robot COMMAND (g_cmd steps) vs robot ACTUAL (encoder
    g_obs crossing 0.5 with 0.35/0.65 hysteresis). Phase is primary (arm lag ~0.2 s is not a gripper error); times are secondary:
    UMI->command in UMI seconds, command->actual in real robot ms. Returns (rows, spurious commands, spurious actual moves)."""
    tJaw = np.load(Path(jawDir) / "umi_jaw.npz")
    tA0 = (segment["hold_pre"][0] + segment["hold_pre"][1]) // 2
    tA1 = (segment["hold_post"][0] + segment["hold_post"][1]) // 2

    def _phase(times):
        return (np.asarray(times, float) - tA0) / (tA1 - tA0)

    tUmi = [(int(tT), tKind) for tT, tKind in zip(tJaw["ev_t"], tJaw["ev_kind"]) if tA0 <= tT <= tA1]
    tSrc = replay["src_t_ns"].astype(np.int64)
    tGCmd = replay["g_cmd"]
    tTCmd = replay["t_cmd_ns"].astype(np.int64)
    tCommands = [(int(tSrc[tI + 1]), int(tTCmd[tI + 1]), "open" if tGCmd[tI + 1] > tGCmd[tI] else "close")
                 for tI in np.where(np.diff(tGCmd) != 0)[0]]
    tGObs = replay["g_obs"]
    tTObs = replay["t_obs_ns"].astype(np.int64)[:len(tGObs)]
    tState = int(tGObs[0] > GRIP_LEVEL)
    tActual = []

    def _cross(index, level):          # linear interpolation of the 0.5 crossing between samples index-1 and index (sample spacing up to 100 ms)
        tF = (level - tGObs[index - 1]) / (tGObs[index] - tGObs[index - 1]) if tGObs[index] != tGObs[index - 1] else 1.0
        return int(tTObs[index - 1] + np.clip(tF, 0, 1) * (tTObs[index] - tTObs[index - 1]))

    for tI in range(1, len(tGObs)):
        if tState == 0 and tGObs[tI] > GRIP_HI:
            tState = 1
            tCrossT = _cross(tI, GRIP_LEVEL)
            tActual.append((int(tSrc[0] + (tCrossT - tTCmd[0]) * speed), tCrossT, "open"))
        elif tState == 1 and tGObs[tI] < GRIP_LO:
            tState = 0
            tCrossT = _cross(tI, GRIP_LEVEL)
            tActual.append((int(tSrc[0] + (tCrossT - tTCmd[0]) * speed), tCrossT, "close"))
    tRows, tUsedC, tUsedA = [], set(), set()
    for tUmiT, tKind in tUmi:
        tIc = min((tI for tI in range(len(tCommands)) if tCommands[tI][2] == tKind and tI not in tUsedC),
                  key=lambda tI: abs(tCommands[tI][0] - tUmiT), default=None)
        tIa = min((tI for tI in range(len(tActual)) if tActual[tI][2] == tKind and tI not in tUsedA),
                  key=lambda tI: abs(tActual[tI][0] - tUmiT), default=None)
        if tIc is not None and abs(_phase(tCommands[tIc][0]) - _phase(tUmiT)) > GRIP_MATCH_PHASE:
            tIc = None
        if tIa is not None and abs(_phase(tActual[tIa][0]) - _phase(tUmiT)) > GRIP_MATCH_PHASE:
            tIa = None
        if tIc is not None:
            tUsedC.add(tIc)
        if tIa is not None:
            tUsedA.add(tIa)
        tRows.append(dict(kind=tKind, s_umi=float(_phase(tUmiT)), s_cmd=float(_phase(tCommands[tIc][0])) if tIc is not None else None,
                          s_act=float(_phase(tActual[tIa][0])) if tIa is not None else None,
                          umi_cmd_s=(tCommands[tIc][0] - tUmiT) / NS_PER_S if tIc is not None else None,
                          cmd_act_ms=(tActual[tIa][1] - tCommands[tIc][1]) / NS_PER_MS if (tIc is not None and tIa is not None) else None))
    return tRows, len(tCommands) - len(tUsedC), len(tActual) - len(tUsedA)


def judgeFree(result):
    """Names of the failed free-motion gates (Korean, as printed)."""
    tGate = GATE_FREE
    tFailed = []
    if result["pos_rmse"] >= tGate["pos_rmse"] or result["pos_p95"] >= tGate["pos_p95"]:
        tFailed.append("위치")
    if result["rot_med"] >= tGate["rot_med"] or result["rot_p95"] >= tGate["rot_p95"]:
        tFailed.append("회전")
    if not (tGate["ratio"][0] <= result["tr_ratio"] <= tGate["ratio"][1]):
        tFailed.append("이동비율")
    if result["amp_A_deg"] > FREE_ROT_GATE_MIN_DEG and not (tGate["ratio"][0] <= result["rot_ratio"] <= tGate["ratio"][1]):
        tFailed.append("회전비율")
    if result["jumps"] > tGate["jumps"]:
        tFailed.append("끊김")
    return tFailed


def _dataAxes(segments, times, points):
    """No robot calibration: axes from the data (sign-corrected mean chords), printed with their mutual angles."""
    tRaw = {}
    for tSeg in segments:
        tX = selectWindow(times, points, *windows(tSeg))
        if tX is not None:
            tRaw.setdefault(tSeg["axis"], []).append(tSeg["sign"] * (tX[-1] - tX[0]) / np.linalg.norm(tX[-1] - tX[0]))
    tRaw = {tKey: np.mean(tValue, 0) / np.linalg.norm(np.mean(tValue, 0)) for tKey, tValue in tRaw.items()}
    if len(tRaw) == 3:
        def _angle(first, second):
            return np.degrees(np.arccos(np.clip(tRaw[first] @ tRaw[second], -1, 1)))
        print(f"데이터에서 잡은 축 사이 각도 (90° 가 이상적): X-Y {_angle('X', 'Y'):.1f}°, X-Z {_angle('X', 'Z'):.1f}°, "
              f"Y-Z {_angle('Y', 'Z'):.1f}° (카메라 좌표, 로봇 캘리브 전)")
    return tRaw


def _umiOnly(umi, segments, hand):
    tA = umi["t_ns"].astype(np.int64)
    tPoints = umi["p_B"] if "p_B" in umi else umi["p_C"]
    segments = [tSeg for tSeg in segments if tSeg["hand"] == hand]
    tAxes = dict(AX) if "p_B" in umi else _dataAxes(segments, tA, tPoints)
    tRows = []
    for tSeg in segments:
        tX = selectWindow(tA, tPoints, *windows(tSeg), tSeg["hold_pre"])
        if tX is None or tSeg["axis"] not in tAxes:
            continue
        tR = stroke(tX, tSeg["sign"] * tAxes[tSeg["axis"]])
        if tR:
            tRows.append((f"{tSeg['axis']} {tSeg['dir']} {tSeg['rep']}", tR))
    return summarize(tRows, "A")


def _gripperReport(run, meta, speed):
    print(f"\n집게 타이밍 (진행률 s, UMI = 손목 어안 판정, 명령 = 로봇에 보낸 g, 실제 = 엔코더) | 기준: 진행률 오차 < "
          f"{GATE_GRIP['phase'] * PERCENT:.0f}%, 명령->실제 < {GATE_GRIP['ms']:.0f} ms, 누락 0, 헛동작 0")
    tGripFails = 0
    tCmdToAct = []
    for tJ, tSeg in enumerate(meta["segments"]):
        tFile = run / f"seg_{tJ:02d}.npz"
        if not tFile.exists() or "g_cmd" not in np.load(tFile):
            continue
        tRows, tSpuriousC, tSpuriousA = gripEvents(np.load(tFile), tSeg, speed, meta["gripper"])
        for tEvent in tRows:
            tMissed = tEvent["s_cmd"] is None or tEvent["s_act"] is None
            tUmiToCmd = None if tEvent["s_cmd"] is None else tEvent["s_cmd"] - tEvent["s_umi"]
            tUmiToAct = None if tEvent["s_act"] is None else tEvent["s_act"] - tEvent["s_umi"]
            tSlow = tEvent["cmd_act_ms"] is not None and tEvent["cmd_act_ms"] >= GATE_GRIP["ms"]
            tBad = tMissed or abs(tUmiToAct) >= GATE_GRIP["phase"] or tSlow
            tGripFails += tBad
            if tEvent["cmd_act_ms"] is not None:
                tCmdToAct.append(tEvent["cmd_act_ms"])

            def _fmt(x):
                return "  -  " if x is None else f"{x:.3f}"

            tUc = "" if tUmiToCmd is None else f"{tUmiToCmd * PERCENT:+.1f}%"
            tUcS = "" if tEvent["umi_cmd_s"] is None else f"{tEvent['umi_cmd_s'] * MS_PER_S:+.0f} ms"
            tUa = "누락" if tUmiToAct is None else f"{tUmiToAct * PERCENT:+.1f}%"
            tCa = "" if tEvent["cmd_act_ms"] is None else f"{tEvent['cmd_act_ms']:.0f} ms"
            print(f"  자유 {tSeg['rep']}회 {'열림' if tEvent['kind'] == 'open' else '닫힘'}: UMI s={tEvent['s_umi']:.3f} | "
                  f"명령 s={_fmt(tEvent['s_cmd'])} | 실제 s={_fmt(tEvent['s_act'])} | "
                  f"UMI->명령 {tUc} ({tUcS}), UMI->실제 {tUa}, 명령->실제 {tCa}  {'FAIL' if tBad else 'PASS'}")
        if tSpuriousC or tSpuriousA:
            print(f"  자유 {tSeg['rep']}회: 헛동작 명령 {tSpuriousC}개, 실제 {tSpuriousA}개  FAIL")
            tGripFails += tSpuriousC + tSpuriousA
    print(f"집게 판정: {'PASS' if tGripFails == 0 else 'FAIL'}"
          + (f" | 명령->실제 지연 중앙 {np.median(tCmdToAct):.0f} ms (로봇 집게 구동 지연)" if tCmdToAct else ""))


def _freeRun(umi, run, meta, speed):
    print(f"\n{'구간':10s} {'사람 범위':>14s} {'위치RMSE':>8s} {'p95':>6s} {'회전중앙':>7s} {'p95':>6s} {'이동비':>5s} {'회전비':>5s} "
          f"{'경로비':>5s} {'끝점':>12s} {'C-B':>6s} {'B-A':>6s} {'끊김':>4s}  판정")
    tFails = 0
    tResults = []
    for tJ, tSeg in enumerate(meta["segments"]):
        tFile = run / f"seg_{tJ:02d}.npz"
        if not tFile.exists() or meta.get("grip_only"):      # 10-20 --grip-only: the arm stayed at READY, only the gripper is compared
            continue
        tR = freeStroke(umi, np.load(tFile), tSeg, speed, plotPath=run / f"free_{tJ:02d}.png")
        if tR is None:
            print(f"  자유 {tSeg['rep']}: 데이터 부족")
            continue
        tFailed = judgeFree(tR)
        tFails += bool(tFailed)
        tResults.append(tR)
        print(f"자유 {tSeg['rep']}회     {tR['amp_A_mm']:5.0f}mm/{tR['amp_A_deg']:3.0f}° {tR['pos_rmse']:6.1f}mm {tR['pos_p95']:5.1f} "
              f"{tR['rot_med']:6.1f}° {tR['rot_p95']:5.1f} {tR['tr_ratio']:5.2f} {tR['rot_ratio']:5.2f} "
              f"{tR['path_ratio']:5.2f} {tR['end_mm']:4.0f}mm/{tR['end_deg']:3.1f}° {tR['cb_rmse']:5.1f} {tR['ba_rmse']:5.1f} "
              f"{tR['jumps']:4d}  "
              + ("PASS" if not tFailed else "FAIL(" + ",".join(tFailed) + ")")
              + f"  (지연 {tR['lag_ms']:.0f} ms, 태그 보임 {tR['tagged'] * PERCENT:.0f}%)")
    print(f"\n전체: {len(tResults)}개 중 통과 {len(tResults) - tFails} | 기준: 위치 RMSE<20 p95<30 mm, 회전 중앙<5° p95<10°, "
          f"이동/회전 비율 0.9-1.1, 끊김 0 | 그림: {run}/free_XX.png")
    if meta.get("gripper"):
        _gripperReport(run, meta, speed)
    return tFails == 0


def _strokeRun(umi, run, meta, speed):
    tA = umi["t_ns"].astype(np.int64)
    tRows = []
    for tJ, tSeg in enumerate(meta["segments"]):
        tFile = run / f"seg_{tJ:02d}.npz"
        if not tFile.exists():
            continue
        tReplay = np.load(tFile)
        tW0, tW1 = windows(tSeg)
        if tSeg["axis"] in AXR:
            tR = rotStroke(umi, tReplay, tSeg, speed, tW0, tW1)
            if tR:
                tRows.append((f"{tSeg['axis']} {tSeg['dir']} {tSeg['rep']}", tR))
            continue
        tSrc, tC = _observedOnSourceClock(tReplay, speed)
        # 10-20: C window = B window shifted by C's measured lag (was a fixed +0.5 s: the operator often kept moving in the 'hold',
        # so the extra 0.5 s of C showed up as a fake 40 mm endpoint error). lag = best match of C(t + lag) to B(t) along the stroke
        # direction
        tU = tReplay["p_cmd"][-1] - tReplay["p_cmd"][0]
        tBf = selectWindow(tSrc, tReplay["p_cmd"], tW0, tW1)
        if tBf is not None and np.linalg.norm(tBf[-1] - tBf[0]) > MIN_STROKE_DIR_M:
            tU = (tBf[-1] - tBf[0]) / np.linalg.norm(tBf[-1] - tBf[0])
        tMaskB = (tSrc >= tW0) & (tSrc <= tW1)
        tXB = tReplay["p_cmd"][tMaskB] @ tU
        tXC = tReplay["p_obs"] @ tU
        if tMaskB.sum() > MIN_LAG_SAMPLES:
            tLag = min(np.arange(0, STROKE_MAX_LAG_NS, LAG_STEP_NS), key=lambda lagNs: np.mean((np.interp(tSrc[tMaskB] + lagNs, tC, tXC) - tXB) ** 2))
        else:
            tLag = 0
        tPathA = selectWindow(tA, umi["p_B"], tW0, tW1, tSeg["hold_pre"])
        tPathB = selectWindow(tSrc, tReplay["p_cmd"], tW0, tW1)
        tPathC = selectWindow(tC, tReplay["p_obs"], tW0 + int(tLag), tW1 + int(tLag))
        if tPathA is None or tPathB is None or tPathC is None:
            print(f"  {tSeg['axis']} {tSeg['dir']} {tSeg['rep']}: 데이터 부족")
            continue
        tR = stroke(tPathA, tSeg["sign"] * AX[tSeg["axis"]], tPathB, tPathC)
        if tR:
            tRows.append((f"{tSeg['axis']} {tSeg['dir']} {tSeg['rep']}", tR))
    return summarize(tRows, "ABC")


def runFiles(args):
    """Score a recorded take (and replay run) as the command line `args` ask: --umi-only, a free-motion run, or a stroke run."""
    tTake = Path(args.take)
    tSide = HAND_SIDES[args.hand]
    tDir = tTake if (tTake / f"umi_traj_{tSide}.npz").exists() else tTake.parent / (tTake.name + "_derived")
    # 10-20: compare against the trajectory that was replayed
    tTraj = json.loads((Path(args.run) / "meta.json").read_text()).get("traj", "") if args.run else ""
    tUmi = np.load(tDir / f"umi_traj_{tSide}{tTraj}.npz", allow_pickle=True)
    tSegs = json.loads((tDir / "segments.json").read_text())
    if tTraj:
        print(f"A = {tDir.name}/umi_traj_{tSide}{tTraj}.npz")
    if args.umi_only:
        return _umiOnly(tUmi, tSegs, args.hand)
    tRun = Path(args.run)
    tMeta = json.loads((tRun / "meta.json").read_text())
    tSpeed = tMeta["speed"]
    if all(tSeg["axis"] == "FREE" for tSeg in tMeta["segments"]):
        return _freeRun(tUmi, tRun, tMeta, tSpeed)
    return _strokeRun(tUmi, tRun, tMeta, tSpeed)


def synthetic():
    """Self-test: a good robot (C = 0.97 x, 1.5 % leak) must pass, a bad one (15 % leak into Z on X strokes) must fail."""
    tRng = np.random.default_rng(SYNTHETIC_SEED)
    tOkAll = True
    for tCase, tLeak, tScale in (("정상 로봇 (C = 0.97배, 1.5% 새어남)", 0.015, 0.97), ("불량 로봇 (X 이동 때 Z 로 15% 새어남)", 0.15, 1.0)):
        print(f"\n=== 합성: {tCase}")
        tRows = []
        for tAxis in "XYZ":
            for tSign in (+1, -1):
                for tRep in range(3):
                    tN = 60
                    tT = np.linspace(0, 1, tN)
                    tMinJerk = 10 * tT ** 3 - 15 * tT ** 4 + 6 * tT ** 5
                    tPathA = (np.outer(tMinJerk, AX[tAxis] * tSign * 0.15) + np.outer(np.sin(np.pi * tT), tRng.normal(0, 0.006, 3))
                         + tRng.normal(0, 0.0015, (tN, 3)))
                    tPathB = tPathA.copy()
                    tT2 = np.linspace(0, 1, 2 * tN)
                    tM2 = 10 * tT2 ** 3 - 15 * tT2 ** 4 + 6 * tT2 ** 5
                    tBs = np.stack([np.interp(np.linspace(0, 1, 2 * tN), tT, tPathB[:, tJ]) for tJ in range(3)], 1)
                    tLagged = np.r_[np.zeros(6), tM2[:-6]]                                   # executed: lags, scaled, maybe leaks
                    tPathC = tScale * np.stack([np.interp(tLagged, tM2, tBs[:, tJ]) for tJ in range(3)], 1) + tRng.normal(0, 0.0005, (2 * tN, 3))
                    if tAxis == "X":
                        tPathC[:, 2] += tLeak * 0.15 * tLagged * tSign * (1 if tLeak < 0.1 else 1)
                    tR = stroke(tPathA, tSign * AX[tAxis], tBs, tPathC)
                    tRows.append((f"{tAxis} {'+' if tSign > 0 else '-'} {tRep + 1}", tR))
        tResult = summarize(tRows, "ABC")
        tOkAll &= (tResult if "정상" in tCase else not tResult)
    print("\n합성 시험", "PASS (정상은 통과, 불량은 걸러냄)" if tOkAll else "FAIL")
    return tOkAll


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("take", nargs="?")
    tParser.add_argument("run", nargs="?")
    tParser.add_argument("--hand", default="오른손", choices=["오른손", "왼손"])
    tParser.add_argument("--umi-only", action="store_true")
    tParser.add_argument("--synthetic", action="store_true")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tParser = buildParser()
    tArgs = tParser.parse_args(argv)
    if tArgs.synthetic:
        return 0 if synthetic() else 1
    if not tArgs.take or (not tArgs.run and not tArgs.umi_only):
        tParser.error("take 와 run (또는 --umi-only) 필요")
    runFiles(tArgs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
