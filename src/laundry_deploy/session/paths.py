"""Files of the deploy tools and the log prefix: act_tagrig_cams.json and start_pose.json live in $TAGRIG_STATE_DIR
(laundry_common.paths; default EGO_TRAIN_ROOT/deploy, the robot Mac's deploy folder), run outputs under the training reports.
"""

from __future__ import annotations

from laundry_common import paths as egopaths

from ..core.contract import TRAIN_ROOT

CAMS_FILE = egopaths.TAGRIG_STATE_DIR / "act_tagrig_cams.json"  # was CAMS_CFG
UVC_UTIL = str(egopaths.UVC_UTIL)  # was UVC
DESKTOP = egopaths.DESKTOP
ROLLOUT_VIDEO_DIR = DESKTOP / "rollout_videos"  # 11-06 user: every rollout video / grip plot goes here, not loose on the Desktop
RUN_OUT_DIR = TRAIN_ROOT / "reports/act_deploy"  # was OUT
START_FILE = egopaths.TAGRIG_STATE_DIR / "start_pose.json"
LATENCY_FILE = egopaths.TAGRIG_STATE_DIR / "latency.json"  # 11-06 UMI latency matching: tools.latency_probe (arm / jaw) + tools.cam_latency
HIST_TOL_S = 0.03  # n_obs 2 (DiT): accept a history sample within 50 +- 30 ms (dit_serving HISTORY_TOL_S)
LOG_PREFIX = "[배포]"


def say(text) -> None:
    print(LOG_PREFIX, text, flush=True)
