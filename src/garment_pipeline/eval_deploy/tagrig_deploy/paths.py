"""Files of the deploy tools and the log prefix. HERE is the eval_deploy folder (where act_tagrig_deploy.py lives), so
act_tagrig_cams.json and start_pose.json stay where they were before the split.
"""
from __future__ import annotations
import pathlib

from egogarment import paths as egopaths
from garment_pipeline.eval_deploy import act_tagrig_core as core


HERE = pathlib.Path(__file__).resolve().parent.parent     # eval_deploy/ (the folder of the old single file)
CAMS_CFG = HERE / "act_tagrig_cams.json"
UVC = str(egopaths.UVC_UTIL)
DESKTOP = egopaths.DESKTOP
OUT = core.LDT / "reports/act_deploy"
START_FILE = HERE / "start_pose.json"
HIST_TOL_S = 0.03          # n_obs 2 (DiT): accept a history sample within 50 +- 30 ms (dit_serving HISTORY_TOL_S)


def say(t):
    print("[배포]", t, flush=True)
