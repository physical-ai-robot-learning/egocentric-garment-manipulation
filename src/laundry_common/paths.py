"""File-system roots of the laundry pipeline: the one place a machine-specific location is written down.

Every default is the layout of the recording / training Mac, so nothing needs setting there. On another machine, point the
roots at your copies with these environment variables; no code changes:

    EGO_DATA_ROOT     recording workspace: datasets/tag_rig/<take>, configs/calibration, datasets/m3_runs   (~/ego_collector)
    EGO_FUSION_WORK   working tree of the fusion stage (CoTracker packs, out/, stitch/qa, start seeds)      (EGO_DATA_ROOT/datasets/m3_runs/_imu_consistency)
    EGO_TRAIN_ROOT    training workspace: data/tag_rig_derived, reports, checkpoints_pod, LeRobot sets       (~/laundry_dit_training)
    EGO_CKPT_ROOT     checkpoint + raw-take archive on the external SSD                                   (/Volumes/PortableSSD/laundry_dit)
    EGO_COCKPIT_ROOT  robot-cockpit checkout (bin/uvc-util camera control tool, robot bridge launcher)    (~/robot-cockpit)
    EGO_DESKTOP       where interactive tools drop QA videos / snapshots                                   (~/Desktop)
    EGO_MODELS_ROOT   fingertip / jaw-keypoint student models and test clips                              (~/finger_distill)
    EGO_VIO_ROOT      stereo VIO builds (Basalt, AirSLAM) of the dropped stereo experiment                  (~/vio)
    EGO_HANDUMI_SW    handumi-sw checkout (bimanual reBot URDF)                                            (~/handumi-sw)
    EGO_IK_PYTHON     interpreter with pinocchio / pink used for the IK sub-process of the dataset builder (~/xvla-mac/bin/python)
    TAGRIG_STATE_DIR  working state of the deploy tools: act_tagrig_ui_settings.json, act_tagrig_cams.json (+ .bak_auto),
                      start_pose.json, height_calib.json (+ .prev, height_calib_pending*.json)   (EGO_TRAIN_ROOT/deploy)

FUSION_WORK is the tree the fusion modules used as "the folder next to the module" before they became a package
(`_imu_consistency/` on the recording Mac); every sub-path below it is unchanged. TAGRIG_STATE_DIR defaults to the deploy
folder the robot Mac has always used, so an installed package never writes into its own source tree.

Files shipped in the repository (configs/laundry/, package resources) resolve from REPO_ROOT and need no setting.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
# This file is <repo>/src/laundry_common/paths.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CONFIGS = REPO_ROOT / "configs" / "laundry"


def _rootFromEnv(envName: str, default: Path) -> Path:
    tValue = os.environ.get(envName)
    return Path(tValue).expanduser() if tValue else default


DATA_ROOT = _rootFromEnv("EGO_DATA_ROOT", HOME / "ego_collector")
TRAIN_ROOT = _rootFromEnv("EGO_TRAIN_ROOT", HOME / "laundry_dit_training")
CKPT_ROOT = _rootFromEnv("EGO_CKPT_ROOT", Path("/Volumes/PortableSSD/laundry_dit"))
COCKPIT_ROOT = _rootFromEnv("EGO_COCKPIT_ROOT", HOME / "robot-cockpit")
DESKTOP = _rootFromEnv("EGO_DESKTOP", HOME / "Desktop")
MODELS_ROOT = _rootFromEnv("EGO_MODELS_ROOT", HOME / "finger_distill")
VIO_ROOT = _rootFromEnv("EGO_VIO_ROOT", HOME / "vio")
HANDUMI_SW_ROOT = _rootFromEnv("EGO_HANDUMI_SW", HOME / "handumi-sw")
IK_PYTHON = _rootFromEnv("EGO_IK_PYTHON", HOME / "xvla-mac/bin/python")

# --- recording workspace -------------------------------------------------------------------------------------------
TAG_RIG = DATA_ROOT / "datasets/tag_rig"                  # recorded takes
DATA_CALIB = DATA_ROOT / "configs/calibration"            # calibration files of the recording machine
M3_RUNS = DATA_ROOT / "datasets/m3_runs"                  # fusion / FGO working data
FUSION_WORK = _rootFromEnv("EGO_FUSION_WORK", M3_RUNS / "_imu_consistency")

# --- training workspace --------------------------------------------------------------------------------------------
DERIVED = TRAIN_ROOT / "data/tag_rig_derived"             # arm-assigned + robotized trajectories
REPORTS = TRAIN_ROOT / "reports"                          # idle_trim.csv, QA stats, deploy logs
IDLE_TRIM_CSV = REPORTS / "idle_trim.csv"
TAGRIG_STATE_DIR = _rootFromEnv("TAGRIG_STATE_DIR", TRAIN_ROOT / "deploy")

UVC_UTIL = COCKPIT_ROOT / "bin/uvc-util"                  # UVC camera-control helper (exposure, device list)
