# laundry_deploy

Receding-horizon deployment of a trained policy on the two reBot B601-DM arms, the robot bridge, and offline tools.

**Status:** current code of the deploy loop. The robot computer still runs the earlier copy of this code (launchers);
moving it to this package is a separate step. Run DRY first; GO only with a person at the robot.

## Layout
| path | what |
|---|---|
| `ui/` (`server`, `app`, `run_control`, `ready_move`, `settings`, `page.html`) | Flask UI on :8770: checkpoint list, DRY / GO, plan settings |
| `session/` (`run_loop`, `streamer`, `cameras`, `ring`, `rig`, `start_pose`, `bridge`, `cli`, …) | the run loop: observe → policy chunk → IK / safety → commit steps |
| `core/` (`policy_runner`, `xvla_runner`, `label_runner`, `chunk`, `contract`, `kinematics`, `collision`, `height_calib`, `camera_model`, `checkpoint`, `images`, `plot`, `errors`) | policy and label runners, chunk / RTC blending, kinematics and table-collision checks, per-arm height calibration |
| `robot_service/` (`mit_bridge`, `config`, `jaw`, `profile`, `arm_state`, `hardware`, gripper tools) | MIT-mode robot bridge (:8021, 100 Hz, auto re-enable, jaw-opening limit) |
| `tools/` (`offline_eval`, `domain_gap`, `cam_align`, `cam_viewer`, `height_calib`, `return_to_start`, `run_n_steps`) | offline evaluation on val / test sets, camera alignment and viewer, utilities |
| `xvla_server` | X-VLA policy server (separate environment) |

## Entry points
```bash
TAGRIG_STATE_DIR=<state dir> python -m laundry_deploy.ui.server
python -m laundry_deploy.session.cli run --policy dit --secs 30
python -m laundry_deploy.tools.offline_eval --ckpt <ckpt> --data <set> --out <dir>
python -m laundry_deploy.robot_service.mit_bridge          # on the robot computer only: starts the bridge (no --help)
```
Tests: `src/laundry_deploy/tests/` (fake robot / cameras, golden routes and outputs) and `tests/laundry/test_ui_routes.py`.

## Provenance
Was `src/garment_pipeline/eval_deploy/` + `scripts/deploy/robot_bridge/` (11-04 OA refactor; module map in the refactor log).
