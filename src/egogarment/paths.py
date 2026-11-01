"""One place for the file-system locations the pipeline code reads and writes.

Every default below is exactly the literal the code used before (the layout of the original recording / training
machine), so behaviour there is unchanged. On another machine, point the roots at your copies with environment
variables; nothing else needs editing.

    EGO_DATA_ROOT     recording workspace: datasets/tag_rig/<take>, configs/calibration, datasets/m3_runs   (~/ego_collector)
    EGO_TRAIN_ROOT    training workspace: data/tag_rig_derived, reports, checkpoints_pod, LeRobot sets   (~/laundry_dit_training)
    EGO_CKPT_ROOT     checkpoint + raw-take archive on the external SSD                                   (/Volumes/PortableSSD/laundry_dit)
    EGO_COCKPIT_ROOT  robot-cockpit checkout (bin/uvc-util camera control tool, robot bridge launcher)    (~/robot-cockpit)
    EGO_DESKTOP       where interactive tools drop QA videos / snapshots                                   (~/Desktop)
    EGO_MODELS_ROOT   fingertip / jaw-keypoint student models and test clips                              (~/finger_distill)
    EGO_VIO_ROOT      stereo VIO builds (Basalt, AirSLAM) of the dropped stereo experiment                  (~/vio)
    EGO_HANDUMI_SW    handumi-sw checkout (bimanual reBot URDF)                                            (~/handumi-sw)
    EGO_IK_PYTHON     interpreter with pinocchio / pink used for the IK sub-process of the dataset builder (~/xvla-mac/bin/python)
    TAGRIG_STATE_DIR  working-state folder of the tag-rig deploy tools: act_tagrig_ui_settings.json, act_tagrig_cams.json (+ .bak_auto),
                      start_pose.json, height_calib.json (+ .prev, height_calib_pending*.json)   (src/garment_pipeline/eval_deploy)
                      The robot Mac sets it to its old deploy folder (~/laundry_dit_training/deploy) so the repo code reads and
                      writes the same files the single-file deploy did.  Run outputs are not here: reports/act_deploy under
                      EGO_TRAIN_ROOT, videos on EGO_DESKTOP, as before.

Paths inside the repository (configs/, the package folders) are resolved from REPO_ROOT / the module's own folder and
need no setting.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]          # this file: <repo>/src/egogarment/paths.py
SRC_ROOT = REPO_ROOT / "src"
CONFIGS = REPO_ROOT / "configs"


def _root(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v).expanduser() if v else default


DATA_ROOT = _root("EGO_DATA_ROOT", HOME / "ego_collector")
TRAIN_ROOT = _root("EGO_TRAIN_ROOT", HOME / "laundry_dit_training")
CKPT_ROOT = _root("EGO_CKPT_ROOT", Path("/Volumes/PortableSSD/laundry_dit"))
COCKPIT_ROOT = _root("EGO_COCKPIT_ROOT", HOME / "robot-cockpit")
DESKTOP = _root("EGO_DESKTOP", HOME / "Desktop")
MODELS_ROOT = _root("EGO_MODELS_ROOT", HOME / "finger_distill")
VIO_ROOT = _root("EGO_VIO_ROOT", HOME / "vio")
HANDUMI_SW_ROOT = _root("EGO_HANDUMI_SW", HOME / "handumi-sw")
IK_PYTHON = _root("EGO_IK_PYTHON", HOME / "xvla-mac/bin/python")
TAGRIG_STATE_DIR = _root("TAGRIG_STATE_DIR", Path(__file__).resolve().parents[1] / "garment_pipeline/eval_deploy")   # default: the eval_deploy folder (as before)

# --- derived locations (same literals as before) ---------------------------------------------------------------
TAG_RIG = DATA_ROOT / "datasets/tag_rig"                  # recorded takes
DATA_CALIB = DATA_ROOT / "configs/calibration"            # calibration files of the recording machine
M3_RUNS = DATA_ROOT / "datasets/m3_runs"                  # fusion / FGO working data
DERIVED = TRAIN_ROOT / "data/tag_rig_derived"             # arm-assigned + robotized trajectories
REPORTS = TRAIN_ROOT / "reports"                          # idle_trim.csv, QA stats, deploy logs
IDLE_TRIM_CSV = REPORTS / "idle_trim.csv"
UVC_UTIL = COCKPIT_ROOT / "bin/uvc-util"                  # UVC camera-control helper (exposure, device list)

