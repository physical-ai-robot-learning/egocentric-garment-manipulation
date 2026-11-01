### Update 2026-10-20: servo-less HandUMI, visual gripper, T-shirt flattening data, UMI→robot calibration

After the v5 report the work moved to T-shirt flattening for the laundry project. Summary in
[`docs/UPDATE_2026-10-20.md`](docs/UPDATE_2026-10-20.md) ([한국어](docs/UPDATE_2026-10-20_ko.md)); the report itself is unchanged.

- **Hardware:** jaw servos removed (10-13) — **reinstalled 10-29** as torque-off jaw encoders (laundry project; see
  [egocentric-garment-manipulation](https://github.com/physical-ai-vla/egocentric-garment-manipulation)); red tape on the jaw tips (later removed); AprilTag rigs (200–202 / 210–212) per unit; XIAO
  nRF52840 wrist IMUs replace the Teensy ([`hardware/README.md`](https://github.com/physical-ai-vla/egocentric-cube-stacking/blob/main/hardware/README.md)).
- **Gripper open/close from the wrist video:** red-tape tip distance per frame (`jaw_track.py`) → hysteresis binary
  (closed ≤ 200 px / open ≥ 260 px) in `src/ego_cart20/sources/jaw_visual.py`; 16/16 spot checks, corr 0.93 vs the servo.
  Used only while the servos were off; from 10-29 the jaw label is the servo encoder again.
- **UMIalign pose pipeline** ([`scripts/rebot/laundry/`](scripts/rebot/laundry/)): MASt3R restart_v1, Orbbec canonical
  tracker, smoother with corrected camera↔IMU rotation + fixed jaw-tip lever + AprilTag gate (TCP error p50 25–36 cm →
  38–59 mm), 65-episode corrected dataset. The fine-tune on it was stopped at ~11k steps; no result.
- **UMI → robot** ([`scripts/rebot/parity/`](scripts/rebot/parity/), `src/ego_cart20/robotize/`):
  robotization (IK-ok p50 0.79, amplitude ~0.81), UMI TCP calibration by corner touch (1–4 mm static), robot camera
  calibration and replay-parity tools built but **not yet run on the robot**.
- **UMI as teleop device** (`src/visual_teleop/`): DPVO / mono VIO NO-GO, stereo VIO first robot teleop then
  dropped (10-16).

