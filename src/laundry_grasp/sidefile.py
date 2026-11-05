"""Per-frame p_grasp side file for an aggregated tag-rig LeRobot set (FLAT250-PGRASP), no dataset rebuild (was
grasp_pgrasp_sidefile.py).

Episode i of the set = key i of meta/split.json (aggregate order). VERIFIED per episode: the source time of frame 0 of both wrist
cams (take wrist t_ns[src_index]) must sit at the demo window start (laundry_grasp.demos.demoWindow, keep_start) within 0.25 s, and
every src_index must be inside that take's wrist recording. Frame value = the latest 15 Hz p_grasp tick at or before that frame's
wrist source time (zero-order hold, causal; the policy sees the same wrist frame).
-> <set>/meta/pgrasp_<version>.npy  float32 (total_frames, 2) [L, R] by global index  + pgrasp_<version>.json
(read by laundry_training, env PGRASP=<version>).

    python -m laundry_grasp.sidefile SET_DIR PGRASP_DIR [--check-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from laundry_common import paths as egopaths

from .demos import demoWindow, parseDemoKey
from .pgrasp import HZ

TOLERANCE_S = 0.25                    # frame 0 vs demo window start
WRIST_COLUMNS = (("L", "wrist_left", 1), ("R", "wrist", 2))   # src_index columns: (global, left_wrist, right_wrist)
NS_PER_S = 1e9
PARQUET_COLUMNS = ["index", "episode_index", "frame_index", "src_index"]
HOLD_RULE = "value = latest 15 Hz p_grasp tick <= wrist source time of the frame (src_index)"
MEAN_DECIMALS = 3


class SideFileError(RuntimeError):
    """the set and the p_grasp series do not line up (episode order, frame range, missing values)"""


def holdValues(tickTimes: np.ndarray, values: np.ndarray, frameTimes: np.ndarray) -> np.ndarray:
    """zero-order hold: per frame the value of the latest tick at or before its time; raises when a frame precedes the first tick
    or lies more than one tick period after the last"""
    tIdx = np.searchsorted(tickTimes, frameTimes, side="right") - 1
    if not ((tIdx >= 0).all() and (frameTimes <= tickTimes[-1] + NS_PER_S / HZ).all()):
        raise SideFileError("frames outside the p_grasp series")
    return values[tIdx]


def checkFrameZero(key: str, cam: str, frameZeroNs: int, windowStartNs: int) -> float:
    """offset [s] of frame 0 from the demo window start; raises beyond TOLERANCE_S"""
    tOffset = abs(frameZeroNs - windowStartNs) / NS_PER_S
    if tOffset > TOLERANCE_S:
        raise SideFileError(f"{key} {cam}: frame 0 at {tOffset:.2f} s from the demo window start -> episode/key order mismatch?")
    return tOffset


def readFrames(setDir: Path) -> pd.DataFrame:
    tFiles = sorted(glob.glob(str(setDir / "data/*/*.parquet")))
    return pd.concat([pd.read_parquet(tF, columns=PARQUET_COLUMNS) for tF in tFiles]).sort_values("index")


def buildSideFile(setDir: Path, pgraspDir: Path, checkOnly: bool) -> tuple[np.ndarray, str | None, int, float]:
    """(P (total_frames, 2) float32, version, episodes, worst frame-0 offset s); P stays NaN in check-only mode"""
    tDf = readFrames(setDir)
    tTotal = json.loads((setDir / "meta/info.json").read_text())["total_frames"]
    if not (len(tDf) == tTotal and (tDf["index"].values == np.arange(tTotal)).all()):
        raise SideFileError("global index not contiguous")
    tSplits = json.loads((setDir / "meta/split.json").read_text())["episodes"]
    tKeys = [tK for tV in tSplits.values() for tK in tV]
    tEpisodes = sorted(tDf["episode_index"].unique())
    if len(tEpisodes) != len(tKeys):
        raise SideFileError(f"{len(tEpisodes)} episodes but {len(tKeys)} split keys")
    tP = np.full((tTotal, 2), np.nan, np.float32)
    tVersion, tWorst = None, 0.0
    for tEpisode, tKey in zip(tEpisodes, tKeys):
        tTake, tDemo = parseDemoKey(tKey)
        tT0, _ = demoWindow(tTake, tDemo)
        tSub = tDf[tDf["episode_index"] == tEpisode]
        tSrc = np.stack(tSub["src_index"].values)
        if not (tSub["frame_index"].values == np.arange(len(tSub))).all():
            raise SideFileError(f"{tKey}: frame_index not 0..n-1")
        tSeries = None if checkOnly else np.load(pgraspDir / "demos" / f"{tTake}_{tKey.split('/')[1]}.npz")
        if tSeries is not None:
            tVersion = str(tSeries["version"])
        for tCol, (tHand, tCam, tSrcCol) in enumerate(WRIST_COLUMNS):
            tWrist = np.load(egopaths.TAG_RIG / tTake / tCam / "t_ns.npy")
            if not tSrc[:, tSrcCol].max() < len(tWrist):
                raise SideFileError(f"{tKey} {tCam}: src_index beyond the recording")
            tFrameT = tWrist[tSrc[:, tSrcCol]]
            tWorst = max(tWorst, checkFrameZero(tKey, tCam, tFrameT[0], tT0))
            if tSeries is not None:
                try:
                    tP[tSub["index"].values, tCol] = holdValues(tSeries["t_ns"], tSeries[f"{tHand}_p"], tFrameT)
                except SideFileError as tErr:
                    raise SideFileError(f"{tKey}: {tErr}") from tErr
    return tP, tVersion, len(tEpisodes), tWorst


def writeSideFile(setDir: Path, pgraspDir: Path, values: np.ndarray, version: str) -> None:
    if np.isnan(values).any():
        raise SideFileError("frames without a p_grasp value")
    np.save(setDir / f"meta/pgrasp_{version}.npy", values)
    (setDir / f"meta/pgrasp_{version}.json").write_text(json.dumps(dict(
        version=version, shape=list(values.shape), columns=["L", "R"], source=str(pgraspDir), rule=HOLD_RULE,
        mean=values.mean(0).round(MEAN_DECIMALS).tolist()), indent=1))


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("set_dir", help="aggregated LeRobot set (data/*/*.parquet with src_index, meta/split.json)")
    tParser.add_argument("pgrasp_dir", help="generator output with demos/<take>_demoNN.npz")
    tParser.add_argument("--check-only", action="store_true", help="verify the episode order only, write nothing")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tSet, tPg = Path(tArgs.set_dir).expanduser(), Path(tArgs.pgrasp_dir).expanduser()
    try:
        tP, tVersion, tEpisodes, tWorst = buildSideFile(tSet, tPg, tArgs.check_only)
        print(f"{tSet.name}: {tEpisodes} episodes, {len(tP)} frames, order verified (worst frame-0 offset {tWorst:.3f} s)", flush=True)
        if tArgs.check_only:
            return 0
        writeSideFile(tSet, tPg, tP, tVersion)
    except SideFileError as tErr:
        print(f"ERROR: {tErr}", file=sys.stderr, flush=True)
        return 1
    print(f"  wrote meta/pgrasp_{tVersion}.npy  mean L {tP[:, 0].mean():.3f} R {tP[:, 1].mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
