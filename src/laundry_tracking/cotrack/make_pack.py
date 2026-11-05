"""CoTracker3 pack builder v2 (was orbbec/v2b/cotrack/make_pack2.py, 10-24): per (take, demo) the Orbbec frames of the idle-trim
window + per-arm query points for CoTracker.
  initial seeds  at the first tracked frame of the window: dark above-table pixels whose 3D point lies within SEED_R of the tag-free
                 v31speed tracker centre (start-marker seeded), up to N_SEED per arm, spread by farthest-point sampling (query_kind 0)
  reserve        every RES_S seconds per arm: dark above-table pixels within RES_R of that arm's v31speed centre at that frame (may be
                 wrong -- the point solver's probation decides), up to N_RES per arm per time (query_kind 1)
Inputs: datasets/tag_rig/<take> (color / depth frames, intrinsics), FUSION_WORK/orbbec/v2b/out/trackv31speed_<take>.npz, the demo
windows, the per-take camera calibration.  Output: <pack>/<take>_dNN/{frames/NNNNN.jpg, meta.json (ks, t_ns, seeds, queries
[frame, x, y], arm_of_query, query_kind)}.
    python -m laundry_tracking.cotrack.make_pack <pack_dir> take:demo ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths

from .. import workdirs
from ..camera_calibration import baseFromCamera
from ..constants import MM_PER_M, NS_PER_S
from ..fgo import arm_data

SEED_R = 0.06                      # m, initial seed pixels around the tracker centre
N_SEED = 40
DARK = 40                          # gray level below which a pixel is UMI-dark
H_MIN = 0.02                       # m above the table
RES_S = 1.0                        # reserve queries every second ...
RES_R = 0.08                       # ... within 8 cm of the tracker centre ...
N_RES = 20                         # ... up to 20 per arm per time
MIN_DEPTH_M = 0.2
MAX_RESERVE_TIMES = 1000           # reserve times tried after the first tracked frame (1..999 x RES_S)
TRACK_NAME = "trackv31speed"
ARMS = ("left", "right")
KIND_SEED, KIND_RESERVE = 0, 1


class PackError(RuntimeError):
    """A demo window has no seed pixel for an arm."""


def _darkPoints(takeDir: Path, k: int, intr: dict, tbc: np.ndarray, zTable: float, c: np.ndarray, radius: float) -> np.ndarray:
    """(x, y) pixels of frame k that are dark, above the table and within radius of c in the base frame"""
    tGray = cv2.imread(str(takeDir / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
    tDepth = cv2.imread(str(takeDir / f"depth_{k:05d}.png"), -1).astype(np.float32) / MM_PER_M
    tVv, tUu = np.mgrid[0:tDepth.shape[0], 0:tDepth.shape[1]]
    tX = np.stack([(tUu - intr["cx"]) / intr["fx"] * tDepth, (tVv - intr["cy"]) / intr["fy"] * tDepth, tDepth],
                  -1) @ tbc[:3, :3].T + tbc[:3, 3]
    tM = (tDepth > MIN_DEPTH_M) & (tGray < DARK) & (tX[..., 2] - zTable > H_MIN) & (np.linalg.norm(tX - c, axis=-1) < radius)
    return np.argwhere(tM)[:, ::-1].astype(float)


def farthestPoints(cand: np.ndarray, n: int) -> np.ndarray:
    """the point nearest the centroid, then farthest-point sampling up to n points"""
    tSel = [int(np.argmin(np.linalg.norm(cand - cand.mean(0), axis=1)))]
    while len(tSel) < min(n, len(cand)):
        tDmin = np.min(np.linalg.norm(cand[:, None] - cand[tSel][None], axis=2), axis=1)
        tSel.append(int(np.argmax(tDmin)))
    return cand[tSel]


def buildDemoPack(pack: Path, take: str, demo: int) -> None:
    tTbc = baseFromCamera(take)
    tTakeDir = paths.TAG_RIG / take
    tIntr = json.loads((tTakeDir / "meta.json").read_text())["intrinsics"]
    tWin = arm_data.demoWindow(take, demo)
    tZ = np.load(workdirs.trackPath(TRACK_NAME, take))
    tT = tZ["t_ns"]
    tZTable = float(tZ["z_table"])
    tKs = np.flatnonzero((tT >= tWin.mW0) & (tT <= tWin.mW1))
    tOut = pack / f"{take}_d{demo:02d}"
    (tOut / "frames").mkdir(parents=True, exist_ok=True)
    for tI, tK in enumerate(tKs):
        shutil.copy(tTakeDir / f"color_{tK:05d}.jpg", tOut / "frames" / f"{tI:05d}.jpg")
    tQueries, tKinds, tArmsQ = [], [], []
    tSeeds = {}

    def _reserve(j, arm):
        tK = tKs[j]
        tC = tZ["c_" + arm][tK]
        if not np.isfinite(tC[0]):
            return np.zeros((0, 2))
        tCand = _darkPoints(tTakeDir, tK, tIntr, tTbc, tZTable, tC, RES_R)
        return tCand if not len(tCand) else farthestPoints(tCand, N_RES)
    for tArm in ARMS:
        tJ0 = next(tI for tI, tK in enumerate(tKs) if np.isfinite(tZ["c_" + tArm][tK, 0]))
        tTres = tT[tKs[tJ0]] + np.arange(1, MAX_RESERVE_TIMES) * RES_S * NS_PER_S
        tTres = tTres[tTres < tT[tKs[-1]]]
        for tTr in tTres:
            tJ = int(np.argmin(np.abs(tT[tKs] - tTr)))
            tQ = _reserve(tJ, tArm)
            tQueries += [[float(tJ), float(tX), float(tY)] for tX, tY in tQ]
            tKinds += [KIND_RESERVE] * len(tQ)
            tArmsQ += [tArm] * len(tQ)
    for tArm in ARMS:
        tJ = next(tI for tI, tK in enumerate(tKs) if np.isfinite(tZ["c_" + tArm][tK, 0]))
        tK = tKs[tJ]
        tC = tZ["c_" + tArm][tK]
        tCand = _darkPoints(tTakeDir, tK, tIntr, tTbc, tZTable, tC, SEED_R)        # (x, y)
        if not len(tCand):
            raise PackError(f"{take} d{demo:02d} {tArm}: no seed pixel within {SEED_R} m of the tracker centre at frame {tJ}")
        tQ = farthestPoints(tCand, N_SEED)
        tSeeds[tArm] = dict(frame=int(tJ), k=int(tK), n=len(tQ), centre=tC.tolist())
        tQueries += [[float(tJ), float(tX), float(tY)] for tX, tY in tQ]
        tKinds += [KIND_SEED] * len(tQ)
        tArmsQ += [tArm] * len(tQ)
    (tOut / "meta.json").write_text(json.dumps(dict(take=take, demo=demo, ks=tKs.tolist(), t_ns=tT[tKs].tolist(), seeds=tSeeds,
                                                    queries=tQueries, arm_of_query=tArmsQ, query_kind=tKinds), indent=0))
    print(take, demo, len(tKs), "frames", {tA: tSeeds[tA]["n"] for tA in tSeeds}, "reserve", tKinds.count(KIND_RESERVE))


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="CoTracker3 packs (frames + seed / reserve queries) of demo windows")
    tAp.add_argument("pack")
    tAp.add_argument("demos", nargs="*", help="take:demo ...")
    tArgs = tAp.parse_args(argv)
    tPack = Path(tArgs.pack)
    tPack.mkdir(parents=True, exist_ok=True)
    for tSpec in tArgs.demos:
        tTake, tDemo = tSpec.split(":")
        buildDemoPack(tPack, tTake, int(tDemo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
