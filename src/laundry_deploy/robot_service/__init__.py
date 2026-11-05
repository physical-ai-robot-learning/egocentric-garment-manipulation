"""MIT-mode robot bridge for the bimanual reBot and its read-only / re-zero gripper tools (run in the colleague's
bh_indy7_LeRobot env on the robot Mac).

  mit_bridge.py         HTTP service (robot_service contract, PLAN playback): `python -m laundry_deploy.robot_service.mit_bridge`
  config.py             environment switches + live motion settings
  jaw.py                leader 14-vector -> raw joint deg / jaw norm, and back
  profile.py            commanded-pose profiles of the control tick
  arm_state.py          runtime state of one arm
  hardware.py           plugin arm build, gains, ports, cameras
  check_grippers.py, read_left_gripper.py, zero_left_j6.py, zero_left_gripper.py   gripper / joint-zero tools
"""
