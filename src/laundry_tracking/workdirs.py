"""Working-tree locations of the fixed-camera pose stage and the per-rig body offset shipped with the package.

Before the package existed the fusion modules wrote "next to themselves"; the production tree (laundry_common.paths.FUSION_WORK,
`_imu_consistency/` on the recording Mac) keeps that layout, so every sub-path below is the one the old module location implied:

    FUSION_WORK/out/RX_pooled_<cam>.npy              pooled camera->TCP hand-eye rotation (was core.py / stitch/wdo.py: out/)
    FUSION_WORK/stitch/qa/<variant>/...              fusion / stitch runs and their QA tables (was stitch/qa)
    FUSION_WORK/orbbec/v2b/start_seed_v1.json        tag-free start-marker seeds (+ start_seed_<take>.json per take)
    FUSION_WORK/orbbec/v2b/out/                      Orbbec tracker outputs (trackv31speed_<take>.npz, rig_model_<rig>.npz)

Read-only files shipped in the package (body_offset_by_rig_v1.json) resolve next to this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from laundry_common import paths

from .constants import MM_PER_M

POOLED_OUT_DIR = paths.FUSION_WORK / "out"
STITCH_QA_DIR = paths.FUSION_WORK / "stitch/qa"
ORBBEC_DIR = paths.FUSION_WORK / "orbbec/v2b"
TRACK_OUT_DIR = ORBBEC_DIR / "out"
BODY_OFFSET_FILE = Path(__file__).resolve().with_name("body_offset_by_rig_v1.json")


def pooledHandEyePath(cam: str) -> Path:
    return POOLED_OUT_DIR / f"RX_pooled_{cam}.npy"


def startSeedPath(take: str | None = None) -> Path:
    """start_seed_v1.json (take None) or the per-take override start_seed_<take>.json"""
    return ORBBEC_DIR / ("start_seed_v1.json" if take is None else f"start_seed_{take}.json")


def trackPath(name: str, take: str) -> Path:
    """Orbbec tracker output out/<name>_<take>.npz (name e.g. trackv31speed, trackv2b)"""
    return TRACK_OUT_DIR / f"{name}_{take}.npz"


def rigModelPath(rig: str) -> Path:
    return TRACK_OUT_DIR / f"rig_model_{rig}.npz"


def qaRunPath(variant: str, take: str, arm: str, demo: int, hideOffset: int) -> Path:
    """one fusion / stitch run: stitch/qa/<variant>/<take>_<arm>_dNN_h<off>.npz (h-1 = production run)"""
    return STITCH_QA_DIR / variant / f"{take}_{arm}_d{demo:02d}_h{hideOffset}.npz"


def bodyOffsetsMm() -> dict:
    """{rig: [x, y, z] mm} body centre = p_tcp + R o, TCP frame (body_offset_by_rig_v1.json)"""
    return json.loads(BODY_OFFSET_FILE.read_text())["offsets_by_rig"]


def bodyOffsetM(rig: str) -> np.ndarray:
    return np.array(bodyOffsetsMm()[rig]) / MM_PER_M
