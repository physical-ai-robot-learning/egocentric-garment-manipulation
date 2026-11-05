"""Closed-loop real-robot deploy of the tag-rig laundry policies on the bimanual reBot B601-DM (deploy stage). Entry point and
the full description of the loop: `python -m laundry_deploy.session.cli {cams,wave,start,run} --help`.

  paths.py            working files (act_tagrig_cams.json, start_pose.json in TAGRIG_STATE_DIR), run output folder, say()
  ring.py             timestamped ring buffer on the host monotonic clock
  cameras.py          UVC device list, wrist-camera auto-mapping, recorder / UMI-port checks, wrist Arducams, Orbbec poller
  camera_commands.py  the `cams` and `wave` sub-commands
  bridge.py           gripper-opening guard, /observe poller, explicit return to READY
  streamer.py         30 Hz command streamer on an indexed joint timeline (merge / set)
  keys.py             terminal keys and the CLI run control
  preview.py          ~5 Hz whole-chunk look-ahead check (IK, workspace, collision, table)
  rig.py              Rig: cameras + Orbbec + poller + arms + collision model + policy cache, shared by every run
  run_loop.py         runSession: the receding-horizon loop (observe -> infer -> re-anchor -> limiter -> IK -> safety -> commit)
  start_pose.py       `start`: READY -> a training start pose
  report.py           log.npz, summary.json, gripper plot, Desktop video
  cli.py              argument parser (the full description), `run`, main()

A run is driven by a control object `ctl` with poll() -> stop reason or None, publish(snapshot) and optionally takeStep() (UI step
mode); KeysCtl (CLI), ui.run_control.UiRunControl and tools.run_n_steps.StepControl implement it.
"""
