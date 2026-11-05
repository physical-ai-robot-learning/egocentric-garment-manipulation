"""Jaw aperture from jaw-tip keypoints (2026-10-22, replaces the binary-only jaw label for takes recorded after the camera re-angle;
was scripts/pose/jaw_aperture.py).

Pipeline stage: jaw state (stage 2) for demos before the 10-29 jaw servo (`from_motor` is the jaw source since then).
Input: a wrist dir (t_ns.npy, wrist_NNNNN.jpg, meta.json hand, optional umi_jaw.npz of the v003 classifier).
Per wrist frame: YOLO-pose (configs/laundry/calibration/jaw_kpt_<hand>_v002.pt) -> tips A/B -> distance d (px, in the jaw crop) ->
  * confidence = detection score; frames with conf < --min-conf are treated as missing and d is interpolated from neighbours
  * d median-filtered over ~0.2 s
  * g = clip((d - d_closed) / (d_open - d_closed), 0, 1)            (0 = closed, 1 = open; per-hand calibration json)
  * state: hysteresis on d around the per-hand threshold (+-band px); RIGHT hand: inside the band the ResNet classifier (umi_jaw
    infer, v003) decides, because the right-hand threshold sits close to the closed-on-cloth distances
Validated on take 20261022_105200 (96 hand-labelled frames): left 48/48, right 46/48 (binary v003 classifier: 37/48 and 47/48).
Output, into <wrist dir>/umi_jaw.npz: keeps the classifier output as p_open_cls / state_cls, sets state, g_cont (= aperture g),
g_is_aperture=1, d_px, kpt_conf, valid (0 = interpolated). Needs ultralytics >= 8.4 (run with ~/finger_distill/.venv/bin/python).
    python -m laundry_jaw.aperture datasets/tag_rig/<take>/wrist        (and .../wrist_left)
    python -m laundry_jaw.aperture --rescore <wrist dir>      # calibration changed: recompute g/state from the saved distances
    NOTE v002: label rounds 1/2 used different tip conventions, so --calibrate on the merged labels is wrong; the v002 json
    thresholds were set by hand (see its note). Do not rerun --calibrate on combined_v002.
    python -m laundry_jaw.aperture --calibrate datasets/jaw_kpt/20261022_105200   # (re)build the calibration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common.paths import CONFIGS
from laundry_shared.visual_teleop.tools.umi_jaw import events

from .filters import medianWindowLen, slidingMedian
from .kpt_label import crop  # same crop as the labels

# 10-22: v002 models (white + dark cloth, new camera settings)
CALIBRATION_PATH = CONFIGS / "calibration/jaw_kpt_calib_v002.json"
# The calibration json names its models relative to the recording tree ("configs/calibration/..."); in this repository that
# folder is CONFIGS.
CALIBRATION_MODEL_PREFIX = "configs"
OTSU_BINS = 60
DEFAULT_BAND_PX = 15.0
OPEN_PERCENTILE = 90
CALIBRATION_NOTE = "10-22 Otsu threshold on hand-labelled tip distances; distances in px of the jaw crop (jaw_kpt_label.CROP)"
# d is median-filtered over this long a window.
MEDIAN_WINDOW_S = 0.2
MIN_VALID_FRAMES = 2
# Hysteresis of the pseudo-probability (same rule as the classifier's events).
EVENT_HI = 0.6
EVENT_LO = 0.4
PROGRESS_EVERY_FRAMES = 500
DEFAULT_MIN_CONF = 0.3
DEFAULT_MIN_DWELL_S = 0.2
DEFAULT_DEVICE = "mps"


class ApertureError(RuntimeError):
    """Too few frames had a confident jaw-tip detection to interpolate the aperture."""


def otsu(x: np.ndarray) -> float:
    """Otsu threshold of `x` on a OTSU_BINS-bin histogram (bin edge); the median when no split separates two classes."""
    tHist, tEdges = np.histogram(x, bins=OTSU_BINS)
    tCentres = (tEdges[:-1] + tEdges[1:]) / 2
    tBest = (0.0, float(np.median(x)))
    for tK in range(1, len(tHist)):
        tW0, tW1 = tHist[:tK].sum(), tHist[tK:].sum()
        if tW0 and tW1:
            tVariance = tW0 * tW1 * ((tHist[:tK] * tCentres[:tK]).sum() / tW0 - (tHist[tK:] * tCentres[tK:]).sum() / tW1) ** 2
            if tVariance > tBest[0]:
                tBest = (tVariance, float(tEdges[tK]))
    return tBest[1]


def calibrate(labelDir: Path) -> None:
    """Per-hand threshold / closed / open distances from labelDir/<hand>/labels.json, written to CALIBRATION_PATH."""
    tCal = {}
    for tHand in ("left", "right"):
        tLabels = json.loads((labelDir / tHand / "labels.json").read_text())["labels"]
        tD = np.array([np.linalg.norm(np.subtract(*tLabel["tips"])) for tLabel in tLabels if tLabel["visible"] and tLabel["tips"]])
        tThreshold = otsu(tD)
        tCal[tHand] = dict(threshold_px=tThreshold, d_closed_px=float(np.median(tD[tD < tThreshold])),
                           d_open_px=float(np.percentile(tD[tD >= tThreshold], OPEN_PERCENTILE)), band_px=DEFAULT_BAND_PX,
                           model=f"configs/calibration/jaw_kpt_{tHand}_v002.pt", n_labels=int(len(tD)))
    tCal["source"] = str(labelDir)
    tCal["note"] = CALIBRATION_NOTE
    CALIBRATION_PATH.write_text(json.dumps(tCal, indent=1))
    print(json.dumps(tCal, indent=1))


def modelPath(calibrationModel: str) -> Path:
    """Repository path of a model named in the calibration json ("configs/calibration/x.pt" -> CONFIGS/calibration/x.pt)."""
    return CONFIGS / Path(calibrationModel).relative_to(CALIBRATION_MODEL_PREFIX)


def _detectDistances(wristDir: Path, frameCount: int, modelFile: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(tip distance px, detection score) per frame; NaN / 0 where the frame is missing or nothing was detected."""
    from ultralytics import YOLO
    tNet = YOLO(str(modelFile))
    tD = np.full(frameCount, np.nan)
    tConf = np.zeros(frameCount)
    for tK in range(frameCount):
        tImage = cv2.imread(str(wristDir / f"wrist_{tK:05d}.jpg"))
        if tImage is None:
            continue
        tResult = tNet.predict(crop(tImage), verbose=False, device=device)[0]
        if tResult.keypoints is None or len(tResult.boxes) == 0:
            continue
        tBest = int(np.argmax(tResult.boxes.conf.cpu().numpy()))
        tTips = tResult.keypoints.xy[tBest].cpu().numpy()
        tConf[tK] = float(tResult.boxes.conf[tBest])
        tD[tK] = float(np.linalg.norm(tTips[0] - tTips[1]))
        if tK % PROGRESS_EVERY_FRAMES == 0:
            print(f"[개구] {wristDir.name} {tK}/{frameCount}", flush=True)
    return tD, tConf


def measureAperture(wristDir: Path, minConf: float, minDwellS: float, device: str, rescore: bool) -> None:
    """Write the aperture label into wristDir/umi_jaw.npz (see the module docstring)."""
    tMeta = json.loads((wristDir / "meta.json").read_text()) if (wristDir / "meta.json").exists() else {}
    tHand = tMeta.get("hand", "right")
    tCal = json.loads(CALIBRATION_PATH.read_text())[tHand]
    tT = np.load(wristDir / "t_ns.npy")
    tOld = dict(np.load(wristDir / "umi_jaw.npz")) if (wristDir / "umi_jaw.npz").exists() else {}
    if "p_open_cls" in tOld:      # rerun: the classifier, not our last output
        tOld["p_open"] = tOld["p_open_cls"]
        tOld["state"] = tOld.get("state_cls", tOld.get("state"))
    tModelName = Path(tCal["model"]).name
    tDetected = None
    if rescore and "d_px" in tOld and str(tOld.get("model_kpt_dpx", "")) == tModelName:
        # new calibration only: reuse the saved (already filtered) distances, skip YOLO
        tSmoothed = tOld["d_px"].astype(float)
        tConf = tOld["kpt_conf"]
        tValid = tOld["valid"].astype(bool)
    else:
        tDetected, tConf = _detectDistances(wristDir, len(tT), modelPath(tCal["model"]), device)
        tValid = np.isfinite(tDetected) & (tConf >= minConf)
        if tValid.sum() < MIN_VALID_FRAMES:
            raise ApertureError("[개구] 검출된 프레임이 거의 없음")
        tFilled = np.interp(np.arange(len(tT)), np.where(tValid)[0], tDetected[tValid])
        tSmoothed = slidingMedian(tFilled, medianWindowLen(tT, MEDIAN_WINDOW_S, len(tT)))
    tG = np.clip((tSmoothed - tCal["d_closed_px"]) / (tCal["d_open_px"] - tCal["d_closed_px"]), 0, 1).astype(np.float32)
    tThreshold, tBand = tCal["threshold_px"], tCal["band_px"]
    # pseudo-probability for the existing hysteresis/event code: 1 above thr+band, 0 below thr-band, linear inside
    tP = np.clip((tSmoothed - (tThreshold - tBand)) / (2 * tBand), 0, 1)
    if tHand == "right" and "p_open" in tOld:        # inside the band defer to the classifier (closed-on-cloth sits near the threshold)
        tClassifier = np.interp(tT, tOld["t_ns"], tOld["p_open"])
        tInside = np.abs(tSmoothed - tThreshold) < tBand
        tP = np.where(tInside, tClassifier, tP)
    tState, tEvents = events(tT, tP, EVENT_HI, EVENT_LO, minDwellS)
    tOut = dict(tOld)
    tOut.update(t_ns=tT, state=tState, g_cont=tG, g_is_aperture=np.int8(1), d_px=tSmoothed.astype(np.float32),
                kpt_conf=tConf.astype(np.float32), valid=tValid.astype(np.int8), hand=tHand, model_kpt=tModelName,
                model_kpt_dpx=(tOld["model_kpt_dpx"] if tDetected is None else tModelName),
                ev_t=np.array([tEvent[0] for tEvent in tEvents], np.int64), ev_kind=np.array([tEvent[1] for tEvent in tEvents]))
    if "p_open" in tOld:
        tOut["p_open_cls"] = tOld["p_open"]
        tOut["state_cls"] = tOld.get("state")
    tOut["p_open"] = tP.astype(np.float32)
    np.savez(wristDir / "umi_jaw.npz", **{tKey: tValue for tKey, tValue in tOut.items() if tValue is not None})
    print(f"[개구] {wristDir.name} ({tHand}): 프레임 {len(tT)}, 검출 {tValid.mean() * 100:.1f}%, 열림 비율 {tState.mean():.2f}, "
          f"사건 {len(tEvents)}개, g 중앙값 {np.median(tG):.2f}", flush=True)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("dir")
    tParser.add_argument("--calibrate", action="store_true")
    tParser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    tParser.add_argument("--min-dwell", type=float, default=DEFAULT_MIN_DWELL_S)
    tParser.add_argument("--device", default=DEFAULT_DEVICE)
    tParser.add_argument("--rescore", action="store_true", help="reuse saved d_px (after a calibration change), no YOLO")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    if tArgs.calibrate:
        calibrate(Path(tArgs.dir))
        return 0
    try:
        measureAperture(Path(tArgs.dir), tArgs.min_conf, tArgs.min_dwell, tArgs.device, tArgs.rescore)
    except ApertureError as tError:
        print(tError, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
