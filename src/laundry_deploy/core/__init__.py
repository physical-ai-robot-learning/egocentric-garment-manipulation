"""Shared core of the tag-rig deploy: everything that decides WHAT the policy sees and HOW its output becomes a robot target,
so the live loop, the UI and the offline tools run the exact same code path.

  contract.py        input / output contract constants (cameras, image size, chunk layout, row period, checkpoints)
  images.py          camera frame -> policy input (Orbbec global, wrist Arducams, framing fix), LeRobot-style items
  chunk.py           state of the current frame, chunk decode, Real-Time Chunking weights, retiming
  checkpoint.py      loads an ACT / Multi-Task DiT checkpoint with its own processors
  policy_runner.py   ACT / Multi-Task DiT in process (incl. the RTC sampler)
  xvla_runner.py     X-VLA through xvla_server.py
  label_runner.py    a recorded robotize_ik label played as an oracle policy
  kinematics.py      robotize_ik arms, FK, link-origin gap, rotation helpers
  collision.py       capsule link-collision model
  camera_model.py    Orbbec calibration files and projection
  height_calib.py    robot-measured table height / xy calibration (file + evaluation)
  plot.py            small cv2 line plot
"""
