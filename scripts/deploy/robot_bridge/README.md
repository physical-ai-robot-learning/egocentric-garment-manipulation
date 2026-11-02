# reBot MIT bridge

`robot_service_mit.py` is an HTTP service (port 8021) for two reBot B601-DM arms.
- It runs on the vendor reBot LeRobot plugin (`bh_indy7_LeRobot`), which is not included here. Start it with that plugin's Python.
- Settings: [configs/robot/rebot_mit_deploy_v001.json](../../../configs/robot/rebot_mit_deploy_v001.json).
- Start/restart it with `python -m visual_teleop.tools.bridge_start --sides left,right --go`.
  - This checks the gripper zeros, kills an old bridge by PID, starts a new one, then calls `/connect`.

## Control

- One 100 Hz thread drives both CAN buses.
- MIT mode with integral feed-forward (no gravity model).
- Motion is synchronised and limited to 45°/s and 180°/s².
- At `/rest` the arms are parked: torque off, read-only. The next command unparks them.
- The plugin's collision guard only logs a would-be trip (in `/status` `auto_events`); `MIT_COLLISION_GUARD=1` restores its latch.

## Environment options (added 2026-10-20/21)

| option | effect |
|---|---|
| `MIT_KP`, `MIT_KI`, `MIT_KI_JOINTS` | per-joint gains; j2 integral 84 cut tracking error to 12–14 mm |
| `MIT_GRIP_OPEN_FRAC_LEFT/RIGHT` | jaw travel fraction (left 0.5) |
| `MIT_MOTOR_REENABLE` (default on) | re-enable a motor that reports "disabled" and hold the measured pose |
| `MIT_USB_RECONNECT` (default on) | on "Broken pipe", rebuild that arm's session (waits up to 30 s for the USB location) |

> **10-27 incident:** a bridge restarted *without* `MIT_GRIP_OPEN_FRAC_LEFT=0.5` let the left jaw open to its full travel
> during a policy run and broke the jaw holder. Always carry every environment option of the running bridge over to a restart
> (read `/status` first: `grip_open_frac` must be `{left: 0.5, right: 1.0}`). `src/garment_pipeline/eval_deploy/act_tagrig_deploy.py` now refuses GO
> otherwise. When the deploy tool opens the wrist cameras itself, start the bridge with `MIT_NO_CAMERAS=1`. A jaw that reads
> non-zero while closed is fixed with `zero_left_gripper_only.py <port> --go` (the zero lives in the motor; a power cycle does not reset it).


## Left-jaw squeeze (added 2026-11-02)

The left jaw developed mechanical slack: its zero is still correct (hand-closed reads about -1°, open about -218°), but a
command to 0° no longer pinches cloth firmly. Two environment options and one endpoint handle it:

| option / endpoint | effect |
|---|---|
| `MIT_GRIP_CLOSE_DEG_LEFT/RIGHT` (default unset = 0) | sets the plugin's jaw close endpoint that many degrees past the zero (the 2026-11-02 runs used 25 on the left) |
| squeeze ramp (fixed, 30°) | the extra close is blended in only over the last 30° before closed, so open commands keep their physical angle; a squeezed jaw is reported back as 0° (closed), so a stop or READY that re-sends the measured jaw no longer opens it |
| `POST /jaw_close_test {"arm": "left", "close_deg": N}` | live tuning: sets the endpoint (allowed range -5 to 45) and closes that jaw; not persisted, the launch option wins after a restart |

Re-cabling the arms can swap which USB location belongs to which arm. On 2026-11-02 this was fixed in the launch configuration
(`LEFT_LOC` / `RIGHT_LOC`, the per-arm USB location IDs), not in code: after any re-cabling, confirm the sides at READY before a policy run.

## Tools (run with the plugin's Python; close the bridge first if it holds the port)

| tool | use |
|---|---|
| `check_grippers.py <ports…>` | prints `GRIP <port> <deg>`; anything beyond ±5° needs a re-zero |
| `zero_left_gripper_only.py <port> --go` | re-zero only the gripper motor (works for either arm) |
| `zero_left_j6_only.py` | re-zero only the left j6 (the plugin's set-zero would re-zero all 7 motors) |
| `read_left_gripper.py <port> [secs]` | live gripper encoder read |
