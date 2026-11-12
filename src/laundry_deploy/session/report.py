"""Run outputs of the deploy session: log.npz + summary.json, the gripper plot and the Desktop video (Orbbec projection + top-down
+ wrists). Input: the per-cycle log of runSession; outputs in reports/act_deploy/<run>/ and on the Desktop.
"""

from __future__ import annotations
import json
import shutil
import subprocess

import cv2
import numpy as np

from ..core.camera_model import project, robotCam
from ..core.contract import CHUNK_ROWS, SIDES, TRAIN_ROOT
from ..core.plot import cvPlot
from .paths import DESKTOP, ROLLOUT_VIDEO_DIR

ARM_COLOURS_BGR = {"left": (255, 140, 0), "right": (0, 140, 255)}  # was COL; left = blue, right = orange (tagrig_qa convention)
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
TOP_SIZE_PX = 480
TOP_X_RANGE_M = (-0.1, 0.8)  # base x (forward) shown in the top view
TOP_Y_RANGE_M = (-0.65, 0.65)  # base y (left) shown in the top view
ARM_BASE_Y_M = 0.28  # the two arm bases at (0, +-0.28)
BIG_FLAGS = ("flags", "gimg_jpg", "wl", "wr")
GRIP_OPEN = 0.5
GRIP_OUT_LO, GRIP_OUT_HI = -0.05, 1.05  # raw gripper outputs outside [0, 1] beyond this count as out of range
GRIP_PERCENTILES = [1, 10, 50, 90, 99]
GRIP_PLOT_ROWS = (1, 8, 32)
GRIP_PLOT_COLOURS = {1: (0, 0, 0), 8: (200, 120, 0), 32: (0, 120, 230)}
GRIP_PLOT_YLIM = (-0.2, 1.2)
TRAIN_TCP_SAMPLE = TRAIN_ROOT / "deploy/train_tcp_sample.npy"
ORBBEC_WH = (848, 480)
WRIST_WH = (384, 216)
VIDEO_FPS_DIT, VIDEO_FPS = 10.0, 20.0
MIN_PLAN_S = 1e-6
MM_PER_M = 1e3


def topDownView(Tm: dict, Tc: dict, workspace: dict, trainPoints, size: int = TOP_SIZE_PX) -> np.ndarray:
    """Base frame from above: x forward = up, +y (robot left) = left. Was topdown()."""
    tImg = np.full((size, size, 3), 255, np.uint8)
    tX0, tX1 = TOP_X_RANGE_M
    tY0, tY1 = TOP_Y_RANGE_M

    def _px(p):
        return (int((tY1 - p[1]) / (tY1 - tY0) * size), int((tX1 - p[0]) / (tX1 - tX0) * size))

    for tP in trainPoints:
        cv2.circle(tImg, _px(tP), 1, (215, 215, 215), -1)
    for tSide in SIDES:
        tLo, tHi = workspace[tSide]
        cv2.rectangle(tImg, _px((tHi[0], tHi[1])), _px((tLo[0], tLo[1])), ARM_COLOURS_BGR[tSide], 1)
        tPts = [_px(tT[:3, 3]) for tT in Tc[tSide]]
        for tI in range(len(tPts) - 1):
            cv2.line(tImg, tPts[tI], tPts[tI + 1], ARM_COLOURS_BGR[tSide], 2)
        cv2.circle(tImg, _px(Tm[tSide][:3, 3]), 7, BLACK, 2)
        cv2.circle(tImg, tPts[0], 4, ARM_COLOURS_BGR[tSide], -1)
        cv2.circle(tImg, tPts[-1], 6, ARM_COLOURS_BGR[tSide], 2)
    cv2.circle(tImg, _px((0, ARM_BASE_Y_M)), 5, BLACK, -1)
    cv2.circle(tImg, _px((0, -ARM_BASE_Y_M)), 5, BLACK, -1)
    cv2.putText(tImg, "top view: up = +x fwd, left = +y (robot L)", (8, 18), 0, 0.45, BLACK, 1)
    cv2.putText(tImg, "L", _px((0.02, 0.33)), 0, 0.6, ARM_COLOURS_BGR["left"], 2)
    cv2.putText(tImg, "R", _px((0.02, -0.25)), 0, 0.6, ARM_COLOURS_BGR["right"], 2)
    return tImg


def _quantiles(x):
    if not np.isfinite(np.asarray(x, float)).any():
        return None
    return dict(
        p50=round(float(np.nanmedian(x)), 1), p90=round(float(np.nanpercentile(x, 90)), 1), max=round(float(np.nanmax(x)), 1)
    )


def _flagCounts(flagLists: list) -> dict:
    tCounts = {}
    for tFlags in flagLists:
        for tF in tFlags:
            if ":" not in tF:  # 10-30: a colon-less flag crashed the summary
                tCounts[tF] = tCounts.get(tF, 0) + 1
                continue
            tHead, tBody = tF.split(":", 1)
            tKey = tHead + ":" + "_".join(tBody.split("_")[:2])
            tCounts[tKey] = tCounts.get(tKey, 0) + 1
    return tCounts


def summarize(
    log: dict, args, stopReason, violations: int, runDir, workspace: dict, readyOff: dict, camsCfg: dict, preview, ckpt, runner
):
    tN = len(log["t"])
    if tN == 0:
        print("기록 없음")
        return {}
    tArr = {tKey: np.asarray(tValue) for tKey, tValue in log.items() if tKey not in BIG_FLAGS}
    np.savez_compressed(
        runDir / "log.npz",
        **tArr,
        flags=np.array([";".join(tF) for tF in log["flags"]]),
        wl=np.stack(log["wl"]),
        wr=np.stack(log["wr"]),
        gimg_jpg=np.array(log["gimg_jpg"], dtype=object),
        preview=np.array([json.dumps(tR) for tR in preview.mLog]),
    )
    tTm, tTc = tArr["T_meas"], tArr["T_chunk"]
    tD1 = np.linalg.norm(tTc[:, :, 0, :3, 3] - tTm[:, :, :3, 3], axis=2) * MM_PER_M
    tD32 = (tTc[:, :, CHUNK_ROWS - 1, :3, 3] - tTm[:, :, :3, 3]) * MM_PER_M
    tCounts = _flagCounts(log["flags"])
    tG = tArr["grip_raw"]  # (n, 32, 2) continuous gripper outputs before the threshold
    tThirds = np.array_split(np.arange(tN), 3)
    tGrip = {
        tSide: dict(
            all_steps=dict(
                zip(("p01", "p10", "p50", "p90", "p99"), np.round(np.percentile(tG[:, :, tJ], GRIP_PERCENTILES), 3).tolist())
            ),
            k1_by_third=[round(float(np.mean(tG[tIx, 0, tJ])), 3) for tIx in tThirds],
            k32_by_third=[round(float(np.mean(tG[tIx, CHUNK_ROWS - 1, tJ])), 3) for tIx in tThirds],
            open_frac_k1=round(float(np.mean(tG[:, 0, tJ] > GRIP_OPEN)), 3),
            open_frac_any_step=round(float(np.mean((tG[:, :, tJ] > GRIP_OPEN).any(1))), 3),
            out_of_0_1=round(float(np.mean((tG[:, :, tJ] < GRIP_OUT_LO) | (tG[:, :, tJ] > GRIP_OUT_HI))), 3),
        )
        for tJ, tSide in enumerate(SIDES)
    }
    tPv = preview.mLog
    tSummary = dict(
        run=str(runDir),
        policy=args.policy,
        ckpt=ckpt,
        dit_sampler=("DDIM 10 x%d" % args.dit_samples if args.policy == "dit" else None),
        go=args.go,
        commit=args.commit,
        cycles=tN,
        secs=round(float(log["t"][-1]), 1),
        plan_hz=round(tN / max(log["t"][-1], MIN_PLAN_S), 1),
        stop_reason=stopReason,
        violation_cycles=violations,
        assume_ready=args.assume_ready,
        global_src=("REPLAY " + args.replay_global) if args.replay_global else args.global_url,
        ready_offset_deg=readyOff,
        cams={tSide: camsCfg[tSide] for tSide in SIDES},
        wave_test=camsCfg.get("wave_test"),
        latency_ms=dict(
            loop=_quantiles(tArr["lat_total_ms"]),
            obs=_quantiles(tArr["lat_obs_ms"]),
            policy=_quantiles(tArr["lat_pol_ms"]),
            ctrl_ik_safety=_quantiles(tArr["lat_ctrl_ms"]),
            obs_to_commit=_quantiles(tArr["obs_to_commit_ms"]),
            k0_steps_skipped=_quantiles(tArr["k0"]),
            preview=_quantiles([tR["ms"] for tR in tPv]) if tPv else None,
        ),
        dit_history_used_frac=round(float(np.mean(tArr["hist_used"])), 3) if args.policy == "dit" else None,
        k1_disp_mm=dict(left=_quantiles(tD1[:, 0]), right=_quantiles(tD1[:, 1])),
        k32_disp_mm_median=dict(
            left=np.round(np.median(tD32[:, 0], 0)).tolist(), right=np.round(np.median(tD32[:, 1], 0)).tolist()
        ),
        k32_dz_mm=dict(left=_quantiles(tD32[:, 0, 2]), right=_quantiles(tD32[:, 1, 2])),
        k32_norm_mm=dict(
            left=_quantiles(np.linalg.norm(tD32[:, 0], axis=1)), right=_quantiles(np.linalg.norm(tD32[:, 1], axis=1))
        ),
        chunk_y_median_m=dict(
            left=round(float(np.median(tTc[:, 0, :, 1, 3])), 3), right=round(float(np.median(tTc[:, 1, :, 1, 3])), 3)
        ),
        left_on_plus_y=bool(np.median(tTc[:, 0, :, 1, 3]) > 0 and np.median(tTc[:, 1, :, 1, 3]) < 0),
        grip_raw=tGrip,
        chunk_smoothness_mm={tK: _quantiles(tArr[f"dchunk_{tK}_mm"]) for tK in ("k1", "k8", "k16", "k32")},
        preview=dict(
            n=len(tPv),
            ik_ok=np.round(np.nanmean([tR["ik_ok"] for tR in tPv], 0), 3).tolist() if tPv else None,
            ik_max_mm=_quantiles([max(tR["ik_max_mm"]) for tR in tPv]) if tPv else None,
            min_clear_mm=_quantiles([tR["min_clear"] * MM_PER_M for tR in tPv]) if tPv else None,
            min_table_mm=_quantiles([tR["min_table"] * MM_PER_M for tR in tPv]) if tPv else None,
        ),
        ik_commit_mm=_quantiles(tArr["ik_mm"].ravel()),
        ws_chunk_out_frac=np.round(tArr["ws_chunk_out"].mean(0), 3).tolist(),
        arm_clear_mm=_quantiles(tArr["arm_clear"] * MM_PER_M),
        link_gap_mm=_quantiles(tArr["link_gap"] * MM_PER_M),
        flag_counts=tCounts,
        img_age_ms=_quantiles(tArr["img_age_ms"].ravel()),
        workspace={
            tSide: [np.round(workspace[tSide][0], 3).tolist(), np.round(workspace[tSide][1], 3).tolist()] for tSide in SIDES
        },
    )
    (runDir / "summary.json").write_text(json.dumps(tSummary, indent=1, ensure_ascii=False))
    print(json.dumps(tSummary, indent=1, ensure_ascii=False))
    if not args.no_video:
        tSummary["video"] = str(renderVideo(log, tArr, runDir, workspace))
        tSummary["grip_plot"] = str(gripPlot(tArr, runDir))
        (runDir / "summary.json").write_text(json.dumps(tSummary, indent=1, ensure_ascii=False))
    return tSummary


def gripPlot(arr: dict, runDir):
    """Raw gripper outputs over time (pre-threshold, 1 = open): k = 1 / 8 / 32 per arm -> Desktop png."""
    tT = arr["t"]
    tG = arr["grip_raw"]
    tImages = [
        cvPlot(
            [(tT, tG[:, tK - 1, tJ], GRIP_PLOT_COLOURS[tK], f"k={tK}") for tK in GRIP_PLOT_ROWS],
            GRIP_PLOT_YLIM,
            f"{runDir.name} {tSide} gripper raw (threshold 0.5)",
            hline=GRIP_OPEN,
            xLabel="s",
        )
        for tJ, tSide in enumerate(SIDES)
    ]
    ROLLOUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tPath = ROLLOUT_VIDEO_DIR / f"act_{runDir.name}_grip.png"
    cv2.imwrite(str(tPath), np.vstack(tImages))
    return tPath


def renderVideo(log: dict, arr: dict, runDir, workspace: dict):
    tCamFromBase, tK = robotCam()
    tTrain = np.load(TRAIN_TCP_SAMPLE) if TRAIN_TCP_SAMPLE.exists() else np.zeros((0, 3))
    ROLLOUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tPath = ROLLOUT_VIDEO_DIR / f"act_{runDir.name}.mp4"
    tTmp = runDir / "video_raw.mp4"
    tW, tH = ORBBEC_WH[0] + TOP_SIZE_PX, ORBBEC_WH[1] + WRIST_WH[1]
    tFps = VIDEO_FPS_DIT if runDir.name.startswith("dit") else VIDEO_FPS
    tWriter = cv2.VideoWriter(str(tTmp), cv2.VideoWriter_fourcc(*"mp4v"), tFps, (tW, tH))
    for tI in range(len(log["t"])):
        tGlobal = cv2.resize(cv2.imdecode(np.frombuffer(log["gimg_jpg"][tI], np.uint8), cv2.IMREAD_COLOR), ORBBEC_WH)
        tTm = {tSide: arr["T_meas"][tI][tJ] for tJ, tSide in enumerate(SIDES)}
        tTc = {tSide: arr["T_chunk"][tI][tJ] for tJ, tSide in enumerate(SIDES)}
        for tSide in SIDES:
            tUv = project(tCamFromBase, tK, np.array([tT[:3, 3] for tT in tTc[tSide]]))
            for tR in range(CHUNK_ROWS - 1):
                if np.isfinite(tUv[tR]).all() and np.isfinite(tUv[tR + 1]).all():
                    cv2.line(tGlobal, tuple(tUv[tR].astype(int)), tuple(tUv[tR + 1].astype(int)), ARM_COLOURS_BGR[tSide], 2)
            tU0 = project(tCamFromBase, tK, tTm[tSide][None, :3, 3])[0]
            if np.isfinite(tU0).all():
                cv2.circle(tGlobal, tuple(tU0.astype(int)), 8, BLACK, 2)
            if np.isfinite(tUv[-1]).all():
                cv2.circle(tGlobal, tuple(tUv[-1].astype(int)), 6, ARM_COLOURS_BGR[tSide], -1)
        tText = (
            f"t {log['t'][tI]:5.1f}s loop {log['lat_total_ms'][tI]:4.0f}ms pol {log['lat_pol_ms'][tI]:4.0f}ms  gL {arr['g_pred'][tI][0]:.2f} "
            f"gR {arr['g_pred'][tI][1]:.2f}  spd {log['speed'][tI]:.2f}"
        )
        cv2.rectangle(tGlobal, (0, 0), (ORBBEC_WH[0], 30), BLACK, -1)
        cv2.putText(tGlobal, tText, (8, 21), 0, 0.55, (0, 255, 0), 1)
        tShown = [tX for tX in log["flags"][tI] if not tX.startswith("LIM")]
        if tShown:
            cv2.putText(tGlobal, " ".join(tShown)[:90], (8, 470), 0, 0.5, (0, 0, 255), 2)
        tTop = np.hstack([tGlobal, topDownView(tTm, tTc, workspace, tTrain)])
        tWl = cv2.cvtColor(log["wl"][tI], cv2.COLOR_RGB2BGR)
        tWr = cv2.cvtColor(log["wr"][tI], cv2.COLOR_RGB2BGR)
        cv2.putText(tWl, "L wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1)
        cv2.putText(tWr, "R wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1)
        tBottom = np.hstack([tWl, tWr, np.full((WRIST_WH[1], tW - 2 * WRIST_WH[0], 3), 40, np.uint8)])
        tK1 = [np.linalg.norm(tTc[tSide][0][:3, 3] - tTm[tSide][:3, 3]) * MM_PER_M for tSide in SIDES]
        tK32 = [(tTc[tSide][CHUNK_ROWS - 1][:3, 3] - tTm[tSide][:3, 3]) * MM_PER_M for tSide in SIDES]
        tLines = [
            f"k1 step  L {tK1[0]:.1f} mm  R {tK1[1]:.1f} mm",
            f"k32 L dxyz {np.round(tK32[0]).astype(int)} mm",
            f"k32 R dxyz {np.round(tK32[1]).astype(int)} mm",
            f"preview IK ok L/R {arr['pv_ik_ok'][tI][0]:.2f}/{arr['pv_ik_ok'][tI][1]:.2f}",
            f"arm clear {arr['arm_clear'][tI] * MM_PER_M:.0f} mm  gap {arr['link_gap'][tI] * MM_PER_M:.0f} mm",
        ]
        for tRow, tLine in enumerate(tLines):
            cv2.putText(tBottom, tLine, (778, 30 + 36 * tRow), 0, 0.6, WHITE, 1)
        tWriter.write(np.vstack([tTop, tBottom]))
    tWriter.release()
    tResult = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tTmp),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(tPath),
        ],
        capture_output=True,
    )
    if tResult.returncode:
        shutil.copy(tTmp, tPath)
    print("영상", tPath, flush=True)
    return tPath
