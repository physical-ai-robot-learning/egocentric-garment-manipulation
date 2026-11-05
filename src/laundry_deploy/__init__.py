"""Deploy stage of the laundry pipeline: receding-horizon policy deploy on the two reBot B601-DM arms, its web UI, the robot bridge
service and the calibration / evaluation tools.

  core/            contract, image / state / chunk algebra, policy runners, kinematics, collision, camera model, height calibration
  session/         the closed-loop run (Rig, runSession, streamer, preview, reports) and its CLI
  ui/              the Korean web UI (Flask) over the same Rig and runSession
  tools/           offline evaluation, height calibration, camera alignment / viewer, domain gap, return-to-start, N-step runs
  robot_service/   the MIT-mode robot bridge (Flask service in the colleague's plugin env) and the gripper zero / read tools
  xvla_server.py   X-VLA inference server (separate env) used by core.xvla_runner

Runnable modules (python -m ...):
  laundry_deploy.session.cli {cams,wave,start,run}     laundry_deploy.ui.server                laundry_deploy.xvla_server
  laundry_deploy.tools.offline_eval                    laundry_deploy.tools.height_calib       laundry_deploy.tools.cam_align
  laundry_deploy.tools.cam_viewer                      laundry_deploy.tools.return_to_start    laundry_deploy.tools.run_n_steps
  laundry_deploy.tools.domain_gap                      laundry_deploy.robot_service.mit_bridge (+ check_grippers, read_left_gripper,
                                                       zero_left_j6, zero_left_gripper)
"""
