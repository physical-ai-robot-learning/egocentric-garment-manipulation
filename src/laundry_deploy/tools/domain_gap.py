"""[2026-10-23] Wrist-camera domain gap, UMI (training) vs robot (deploy), quantified -- UMI-L <-> robot-L, UMI-R <-> robot-R.
UMI side  = raw take frames (wrist/, wrist_left/ 1280x720 JPEG, = what the converter resized) from the first 1.5 s of every training
            demo ('자유:' start = the typical cloth start state), arm-assigned exactly like the converter (source_meta assignments).
Robot side= live robot wrist cameras (act_tagrig_cams.json locations, same 1920x1080 -> 1280x720 path + fixed exposure as deploy).
Per side: resolution / capture path, nominal intrinsics + FOV (same Arducam KB4 model), brightness (grey mean / p1 / p50 / p99, per
channel), jaw mask (dark pixels connected to the bottom edge, centre band) -> area %, bbox, centroid, width at 90 % height, axis angle,
cloth mask (beige/brown HSV) -> area %, centroid; orientation check = jaw at bottom centre, axis ~vertical (no rotation / flip).
A deterministic framing fix (scale about the centre + shift, from the jaw bboxes) is PROPOSED when the jaws differ by > 5 % scale
or > 15 px, applied to the robot frames and re-measured (before/after); it is written to act_tagrig_cams.json as wrist_align_proposed
and used only with `session.cli run --wrist-align` after the user copies it to wrist_align.
Output: Desktop act_wrist_domain_gap_<t>.jpg (paired, masks overlaid), _hist.png, reports/act_deploy/domain_gap_<t>.json
usage: python -m laundry_deploy.tools.domain_gap [--frames 20] [--umi-per-demo 3]
Was laundry_deploy/act_tagrig_domain_gap.py.
"""

from __future__ import annotations
import argparse
import json
import sys
import time

import cv2
import numpy as np

from laundry_common import paths as egopaths

from ..core.contract import SIDES, TRAIN_ROOT
from ..core.errors import DeployError, HardwareError
from ..core.images import applyAlign
from ..core.plot import cvPlot
from ..session.cameras import WristCam, recorderRunning, umiLocationWarning
from ..session.paths import CAMS_FILE, DESKTOP, RUN_OUT_DIR

TAG_RIG = egopaths.TAG_RIG
FISHEYE = {"left": egopaths.DATA_CALIB / "fisheye_left_v002.yaml", "right": egopaths.DATA_CALIB / "fisheye_right_v002.yaml"}
SOURCE_META = "data/lerobot/tagrig_laundry_v1_fused_train/source_meta.json"
FREE_PREFIX = "자유:"  # protocol steps that start a demo from the cloth start state
START_WINDOW_S = 1.5
NS_PER_S = 1e9
FRAME_WH = (1280, 720)
FRAME_CX = 640  # centre column of the 1280-wide frame
FRAME_HALF_H = 360
JAW_DARK_GREY = 70  # jaw pixels are darker than this
JAW_TOP_FRAC = 0.30  # rows above 30 % of the height are never jaw
JAW_BAND = (0.3, 0.7)  # centre column band of the jaw
JAW_EDGE_PX = 3  # a jaw blob touches the bottom edge within this
JAW_MIN_AREA_FRAC = 0.002
JAW_WIDTH_ROW_FRAC = 0.9  # jaw width is measured at 90 % of its height
CLOTH_H, CLOTH_S, CLOTH_V = (5, 25), (35, 170), (50, 220)  # beige / brown HSV ranges
CLOTH_OPEN_PX = 7
MASK_TINT = 127
AGG_KEYS = ["jaw_area_pct", "cloth_area_pct", "contrast_std", "jaw_top_y", "jaw_width_at90", "jaw_axis_deg_from_vertical"]
KB4_ITERATIONS = 30
NO_FIX_SCALE_TOL, NO_FIX_SHIFT_PX = 0.05, 15
FIXABLE_SCALE = (0.8, 1.25)
ORIENTATION_AXIS_TOL_DEG = 15
ORIENTATION_CENTRE_TOL_PX = 200
PICK_JAW_TOP_PX, PICK_GREY = 50, 20  # typical UMI frame: jaw extent near the UMI median, brightness near the robot's
PANEL_WH = (640, 360)
HIST_BINS = 64
HIST_BIN_WIDTH = 4
HIST_PLOT_WH = (640, 260)
WARMUP_S = 1.5
FRAME_PERIOD_S = 0.1
LABEL_YELLOW = (0, 255, 255)


def jawMask(bgr: np.ndarray) -> np.ndarray:
    """Dark (grey < 70) pixels connected to the bottom edge inside the centre 40 % band, lower 70 % of the image (a black start tag
    touching the jaw can leak in -> medians over many frames)."""
    tH, tW = bgr.shape[:2]
    tGrey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tM = (tGrey < JAW_DARK_GREY).astype(np.uint8)
    tM[: int(JAW_TOP_FRAC * tH)] = 0
    tM[:, : int(JAW_BAND[0] * tW)] = 0
    tM[:, int(JAW_BAND[1] * tW) :] = 0
    tN, tLabels, tStats, _ = cv2.connectedComponentsWithStats(tM, 8)
    tKeep = np.zeros_like(tM)
    for tI in range(1, tN):
        _, tY, _, tHh, tArea = tStats[tI]
        if tY + tHh >= tH - JAW_EDGE_PX and tArea > JAW_MIN_AREA_FRAC * tH * tW:
            tKeep[tLabels == tI] = 1
    return tKeep.astype(bool)


def clothMask(bgr: np.ndarray, jaw: np.ndarray) -> np.ndarray:
    tHsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tHue, tSat, tVal = tHsv[..., 0], tHsv[..., 1], tHsv[..., 2]
    tM = (
        (tHue >= CLOTH_H[0])
        & (tHue <= CLOTH_H[1])
        & (tSat >= CLOTH_S[0])
        & (tSat <= CLOTH_S[1])
        & (tVal >= CLOTH_V[0])
        & (tVal <= CLOTH_V[1])
        & ~jaw
    )
    return cv2.morphologyEx(tM.astype(np.uint8), cv2.MORPH_OPEN, np.ones((CLOTH_OPEN_PX, CLOTH_OPEN_PX), np.uint8)).astype(bool)


def measureFrame(bgr: np.ndarray):
    """-> (numbers dict, jaw mask, cloth mask). Was measure()."""
    tH, tW = bgr.shape[:2]
    tGrey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tJaw = jawMask(bgr)
    tCloth = clothMask(bgr, tJaw)
    tOut = dict(
        size=[tW, tH],
        grey=dict(
            mean=float(tGrey.mean()),
            p1=float(np.percentile(tGrey, 1)),
            p50=float(np.median(tGrey)),
            p99=float(np.percentile(tGrey, 99)),
        ),
        bgr_mean=[float(tX) for tX in bgr.reshape(-1, 3).mean(0)],
        contrast_std=float(tGrey.std()),
        jaw_area_pct=float(tJaw.mean() * 100),
        cloth_area_pct=float(tCloth.mean() * 100),
    )
    if tJaw.any():
        tYs, tXs = np.nonzero(tJaw)
        tX0, tX1, tY0, tY1 = tXs.min(), tXs.max(), tYs.min(), tYs.max()
        tRow = int(tY0 + JAW_WIDTH_ROW_FRAC * (tY1 - tY0))
        tRowCols = np.nonzero(tJaw[tRow])[0]
        tCov = np.cov(np.c_[tXs, tYs].T)
        _, tEvec = np.linalg.eigh(tCov)
        tAxis = tEvec[:, -1]
        tAng = float(np.degrees(np.arctan2(tAxis[0], tAxis[1])))
        tAng = (tAng + 90) % 180 - 90
        tOut.update(
            jaw_bbox=[int(tX0), int(tY0), int(tX1), int(tY1)],
            jaw_centroid=[float(tXs.mean()), float(tYs.mean())],
            jaw_top_y=int(tY0),
            jaw_width_at90=int(tRowCols.max() - tRowCols.min()) if len(tRowCols) else 0,
            jaw_axis_deg_from_vertical=tAng,
        )
    if tCloth.any():
        tYs, tXs = np.nonzero(tCloth)
        tOut["cloth_centroid"] = [float(tXs.mean()), float(tYs.mean())]
    return tOut, tJaw, tCloth


def aggregate(measurements: list) -> dict:
    """Medians over frames. Was agg()."""
    tR = {
        tK: round(float(np.median([tM[tK] for tM in measurements if tK in tM])), 2)
        for tK in AGG_KEYS
        if any(tK in tM for tM in measurements)
    }
    tR["grey"] = {tK: round(float(np.median([tM["grey"][tK] for tM in measurements])), 1) for tK in ("mean", "p1", "p50", "p99")}
    tR["bgr_mean"] = np.round(np.median([tM["bgr_mean"] for tM in measurements], 0), 1).tolist()
    for tK in ("jaw_centroid", "cloth_centroid"):
        tV = [tM[tK] for tM in measurements if tK in tM]
        if tV:
            tR[tK] = np.round(np.median(tV, 0), 1).tolist()
    tBoxes = [tM["jaw_bbox"] for tM in measurements if "jaw_bbox" in tM]
    if tBoxes:
        tR["jaw_bbox"] = np.round(np.median(tBoxes, 0)).astype(int).tolist()
    tR["n"] = len(measurements)
    return tR


def overlay(bgr: np.ndarray, jaw: np.ndarray, cloth: np.ndarray, label: str) -> np.ndarray:
    tV = bgr.copy()
    tV[jaw] = (0.5 * tV[jaw] + [0, 0, MASK_TINT]).astype(np.uint8)
    tV[cloth] = (0.5 * tV[cloth] + [MASK_TINT, 0, 0]).astype(np.uint8)
    cv2.rectangle(tV, (0, 0), (tV.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(tV, label, (10, 36), 0, 1.1, LABEL_YELLOW, 2)
    return tV


def umiFrames(nPerDemo: int, startOnly: bool = True) -> dict:
    """UMI wrist frames of every training demo, arm-assigned like the converter -> {side: [jpg paths]}. Was umi_frames()."""
    tMeta = json.loads((TRAIN_ROOT / SOURCE_META).read_text())
    tOut = {"left": [], "right": []}
    for tTake, tAssign in tMeta["assignments"].items():
        tTakeDir = TAG_RIG / tTake
        tTakeMeta = json.loads((tTakeDir / "meta.json").read_text())
        tT0 = int(tTakeMeta["t_start_monotonic_ns"])
        for tSide in ("left", "right"):
            tCamDir = tTakeDir / tAssign["cam_of_arm"][tSide]
            tT = np.load(tCamDir / "t_ns.npy")
            for tStep in [tP for tP in tTakeMeta["protocol"] if tP["text"].startswith(FREE_PREFIX)]:
                tTs = tT0 + int(tStep["t0"] * NS_PER_S)
                tTe = tTs + int(tStep["dur"] * NS_PER_S)
                tSel = np.flatnonzero((tT >= tTs) & (tT <= (tTs + START_WINDOW_S * NS_PER_S if startOnly else tTe)))
                for tI in tSel[:: max(1, len(tSel) // nPerDemo)][:nPerDemo]:
                    tFile = tCamDir / f"wrist_{tI:05d}.jpg"
                    if tFile.exists():
                        tOut[tSide].append(tFile)
    return tOut


def fisheyeFov(side: str) -> dict:
    """Nominal KB4 fisheye intrinsics -> horizontal / vertical FOV. Was fov()."""
    import yaml

    tCalib = yaml.safe_load(FISHEYE[side].read_text())
    tK = np.array(tCalib["K"])
    tD = np.array(tCalib["D"])
    tW, tH = tCalib["image_size"]

    def theta(rd):  # invert KB4 theta_d = th (1 + k1 th^2 + ...)
        tTh = rd
        for _ in range(KB4_ITERATIONS):
            tTh = rd / (1 + tD[0] * tTh**2 + tD[1] * tTh**4 + tD[2] * tTh**6 + tD[3] * tTh**8)
        return tTh

    tHf = np.degrees(theta((tW - tK[0, 2]) / tK[0, 0]) + theta(tK[0, 2] / tK[0, 0]))
    tVf = np.degrees(theta((tH - tK[1, 2]) / tK[1, 1]) + theta(tK[1, 2] / tK[1, 1]))
    return dict(
        file=FISHEYE[side].name, fx_1920=round(float(tK[0, 0]), 1), hfov_deg=round(float(tHf), 1), vfov_deg=round(float(tVf), 1)
    )


def _proposeAlign(umi: dict, robot: dict) -> dict | None:
    """Proposed framing: scale = UMI jaw bbox height / robot jaw bbox height (jaws touch the bottom edge), shift = centroid x."""
    if "jaw_bbox" not in umi or "jaw_bbox" not in robot:
        return None
    tHu, tHr = FRAME_WH[1] - umi["jaw_bbox"][1], FRAME_WH[1] - robot["jaw_bbox"][1]
    tScale = tHu / max(tHr, 1)
    tDx = umi["jaw_centroid"][0] - (FRAME_CX + tScale * (robot["jaw_centroid"][0] - FRAME_CX))
    # scale about the centre moves the bottom edge to 360 + sc*360; shift it back down so the jaw still touches the bottom
    tDy = FRAME_WH[1] - (FRAME_HALF_H + tScale * FRAME_HALF_H)
    tProp = dict(scale=round(float(tScale), 3), dx=round(float(tDx), 1), dy=round(float(tDy), 1), rot180=False, hflip=False)
    if abs(tScale - 1) <= NO_FIX_SCALE_TOL and abs(tDx) <= NO_FIX_SHIFT_PX:
        tProp["needed"] = False
    elif FIXABLE_SCALE[0] <= tScale <= FIXABLE_SCALE[1]:
        tProp["needed"] = True
    else:
        tProp.update(
            needed=False,
            not_fixable=f"jaw scale ratio {tScale:.2f}: a zoom this large would cut the fisheye FOV -> jaw / camera mount geometry differs, not a framing offset",
        )
    return tProp


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("--frames", type=int, default=20)
    tParser.add_argument("--umi-per-demo", type=int, default=3)
    tParser.add_argument("--exposure", type=int, default=2)
    tParser.add_argument(
        "--whole-demo", action="store_true", help="UMI frames from the whole demos (default: first 1.5 s = start state)"
    )
    tParser.add_argument("--tag", default="")
    return tParser


def _greyHist(images: list) -> np.ndarray:
    return np.concatenate(
        [cv2.calcHist([cv2.cvtColor(tX, cv2.COLOR_BGR2GRAY)], [0], None, [HIST_BINS], [0, 256]).ravel()[None] for tX in images]
    ).mean(0)


def domainGap(args) -> None:
    if recorderRunning():
        raise HardwareError(f"녹화 프로세스가 카메라를 쓰는 중: {recorderRunning()}")
    tCfg = json.loads(CAMS_FILE.read_text())
    umiLocationWarning(tCfg)
    tStamp = time.strftime("%H%M%S") + (f"_{args.tag}" if args.tag else "")
    tUmi = umiFrames(args.umi_per_demo, startOnly=not args.whole_demo)
    tReport = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), cams={tSide: tCfg[tSide] for tSide in SIDES}, sides={})
    tCams = {tSide: WristCam(tSide, tCfg[tSide]["location"], tCfg[tSide]["index"], args.exposure, prep=False) for tSide in SIDES}
    for tCam in tCams.values():
        tCam.start()
    time.sleep(WARMUP_S)
    tRobot = {tSide: [] for tSide in SIDES}
    for _ in range(args.frames):
        for tSide in SIDES:
            tRobot[tSide].append(cv2.resize(tCams[tSide].mRaw, FRAME_WH, interpolation=cv2.INTER_AREA))
        time.sleep(FRAME_PERIOD_S)
    for tCam in tCams.values():
        tCam.mStop = True
    tRows, tHists = [], {}
    for tSide in SIDES:
        tUmiImgs = [tX for tX in (cv2.imread(str(tF)) for tF in tUmi[tSide]) if tX is not None]
        tMu = [measureFrame(tX)[0] for tX in tUmiImgs]
        tMr = [measureFrame(tX)[0] for tX in tRobot[tSide]]
        tAu, tAr = aggregate(tMu), aggregate(tMr)
        tProp = _proposeAlign(tAu, tAr)
        tAfter = (
            aggregate([measureFrame(applyAlign(tX, tProp))[0] for tX in tRobot[tSide]]) if tProp and tProp["needed"] else None
        )
        tAxisOk = (
            abs(tAr.get("jaw_axis_deg_from_vertical", 90) - tAu.get("jaw_axis_deg_from_vertical", 0)) < ORIENTATION_AXIS_TOL_DEG
        )
        tCentreOk = abs(tAr.get("jaw_centroid", [0])[0] - FRAME_CX) < ORIENTATION_CENTRE_TOL_PX
        tReport["sides"][tSide] = dict(
            umi=tAu,
            robot=tAr,
            robot_after_align=tAfter,
            proposed_align=tProp,
            umi_files=len(tUmiImgs),
            capture=dict(
                umi="Arducam 0x0c45:0x0261, 1920x1080 -> 1280x720 INTER_AREA, JPEG q92 (umi_jaw record), fixed exposure 2 + IMAGE_CTRL",
                robot="same model / same path (deploy WristCam + core.prep_wrist), fixed exposure %d + IMAGE_CTRL"
                % args.exposure,
            ),
            intrinsics=dict(umi_nominal=fisheyeFov(tSide), robot="not calibrated (same Arducam fisheye model; assume nominal)"),
            orientation=dict(
                umi_jaw_axis_deg=tAu.get("jaw_axis_deg_from_vertical"),
                robot_jaw_axis_deg=tAr.get("jaw_axis_deg_from_vertical"),
                verdict="no rotation / flip needed" if tAxisOk and tCentreOk else "CHECK",
            ),
            left_right=dict(
                robot_cloth_centroid_x=tAr.get("cloth_centroid", [None])[0],
                expect=(
                    "cloth right of centre (x > 640) for the LEFT arm"
                    if tSide == "left"
                    else "cloth left of centre (x < 640) for the RIGHT arm"
                ),
            ),
        )
        # paired image: typical UMI frame (jaw extent near the UMI median, brightness near the robot's; the cloth-area match
        # picked
        # washed-out frames) vs the last robot frame
        tPick = (
            int(
                np.argmin(
                    [
                        abs(tM.get("jaw_top_y", 0) - tAu.get("jaw_top_y", 0)) / PICK_JAW_TOP_PX
                        + abs(tM["grey"]["mean"] - tAr["grey"]["mean"]) / PICK_GREY
                        for tM in tMu
                    ]
                )
            )
            if tMu
            else 0
        )
        _, tJm, tCm = measureFrame(tUmiImgs[tPick])
        tA1 = overlay(
            tUmiImgs[tPick],
            tJm,
            tCm,
            f"UMI {tSide.upper()} (train) {tUmi[tSide][tPick].parent.parent.name}/{tUmi[tSide][tPick].parent.name}",
        )
        _, tJm, tCm = measureFrame(tRobot[tSide][-1])
        tA2 = overlay(tRobot[tSide][-1], tJm, tCm, f"ROBOT {tSide.upper()} live {tCfg[tSide]['location']}")
        tRow = [tA1, tA2]
        if tAfter:
            tX = applyAlign(tRobot[tSide][-1], tProp)
            _, tJm, tCm = measureFrame(tX)
            tRow.append(overlay(tX, tJm, tCm, f"ROBOT {tSide.upper()} + proposed align"))
        tRows.append(
            np.hstack(
                [cv2.resize(tR, PANEL_WH) for tR in tRow]
                + ([np.zeros((PANEL_WH[1], PANEL_WH[0], 3), np.uint8)] if len(tRow) == 2 else [])
            )
        )
        tHists[tSide] = (_greyHist(tUmiImgs), _greyHist(tRobot[tSide]))
    tPaired = DESKTOP / f"act_wrist_domain_gap_{tStamp}.jpg"
    cv2.imwrite(str(tPaired), np.vstack(tRows))
    tPlots = []
    for tSide in SIDES:
        tHu, tHr = tHists[tSide]
        tX = np.arange(HIST_BINS) * HIST_BIN_WIDTH
        tPlots.append(
            cvPlot(
                [
                    (tX, tHu / tHu.sum(), (200, 120, 0), "UMI (train starts)"),
                    (tX, tHr / tHr.sum(), (0, 120, 230), "robot (live)"),
                ],
                (0, float(max((tHu / tHu.sum()).max(), (tHr / tHr.sum()).max()))),
                f"{tSide} wrist grey histogram",
                HIST_PLOT_WH,
                xLabel="grey",
            )
        )
    tHistPath = DESKTOP / f"act_wrist_domain_gap_{tStamp}_hist.png"
    cv2.imwrite(str(tHistPath), np.hstack(tPlots))
    tCfg["wrist_align_proposed"] = {tSide: tReport["sides"][tSide]["proposed_align"] for tSide in SIDES}
    CAMS_FILE.write_text(json.dumps(tCfg, indent=1, ensure_ascii=False))
    tReport.update(paired=str(tPaired), hist=str(tHistPath))
    RUN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tReportPath = RUN_OUT_DIR / f"domain_gap_{tStamp}.json"
    tReportPath.write_text(json.dumps(tReport, indent=1, ensure_ascii=False))
    print(json.dumps(tReport, indent=1, ensure_ascii=False))
    print("저장", tPaired, tHistPath, tReportPath)


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        domainGap(tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
