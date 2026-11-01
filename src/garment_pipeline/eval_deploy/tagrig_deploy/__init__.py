"""Closed-loop real-robot deploy of the tag-rig laundry policies (split out of act_tagrig_deploy.py on 2026-11-01, behaviour
unchanged). Entry point and CLI help: `python -m garment_pipeline.eval_deploy.act_tagrig_deploy {cams,wave,start,run} --help`.

  paths.py       working files (act_tagrig_cams.json, start_pose.json next to act_tagrig_deploy.py), run output folder, say()
  cameras.py     USB / UVC camera mapping, ring buffer, wrist Arducams, Orbbec poller, `cams` and `wave` sub-commands
  bridge.py      gripper-opening guard, /observe poller, 30 Hz command streamer (indexed timeline), return to READY
  keys.py        terminal keys and the CLI run control
  preview.py     ~5 Hz whole-chunk look-ahead check (IK, workspace, collision, table)
  rig.py         Rig: cameras + Orbbec + poller + arms + collision model + policy cache, shared by every run
  session.py     run_session: the receding-horizon loop (observe -> infer -> re-anchor -> limiter -> IK -> safety -> commit)
  start_pose.py  `start`: READY -> a training start pose
  report.py      log.npz, summary.json, gripper plot, Desktop video
  cli.py         argument parser (the full description) and `run`

The names below are the public names of the old single file; the web UI uses them as `dep.<name>`.
"""
from garment_pipeline.eval_deploy.tagrig_deploy.paths import HERE, CAMS_CFG, UVC, DESKTOP, OUT, START_FILE, HIST_TOL_S, say
from garment_pipeline.eval_deploy.tagrig_deploy.cameras import (
    uvc_list, auto_remap_cams, cam_key, recorder_running, umi_location_warning, Ring, WristCam, FakeWristCam, GlobalCam,
    cams_cmd, wave_cmd)
from garment_pipeline.eval_deploy.tagrig_deploy.bridge import (
    GRIP_OPEN_FRAC_MAX, check_grip_limits, RobotPoller, Streamer, go_ready)
from garment_pipeline.eval_deploy.tagrig_deploy.keys import Keys, KeysCtl
from garment_pipeline.eval_deploy.tagrig_deploy.preview import Preview
from garment_pipeline.eval_deploy.tagrig_deploy.rig import Rig
from garment_pipeline.eval_deploy.tagrig_deploy.session import check_run_args, run_session, ensemble
from garment_pipeline.eval_deploy.tagrig_deploy.start_pose import start_cmd
from garment_pipeline.eval_deploy.tagrig_deploy.report import COL, topdown, summarize, grip_plot, render_video
from garment_pipeline.eval_deploy.tagrig_deploy.cli import run_cmd, main

__all__ = [
    "HERE", "CAMS_CFG", "UVC", "DESKTOP", "OUT", "START_FILE", "HIST_TOL_S", "say", "uvc_list", "auto_remap_cams",
    "cam_key", "GRIP_OPEN_FRAC_MAX", "check_grip_limits", "recorder_running", "umi_location_warning", "Ring",
    "WristCam", "FakeWristCam", "GlobalCam", "RobotPoller", "cams_cmd", "Keys", "Streamer", "Preview", "Rig", "KeysCtl",
    "check_run_args", "run_cmd", "run_session", "start_cmd", "ensemble", "go_ready", "COL", "topdown", "summarize",
    "grip_plot", "render_video", "wave_cmd", "main"
]
