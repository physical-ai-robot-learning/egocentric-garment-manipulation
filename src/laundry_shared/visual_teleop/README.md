# visual_teleop — reBot bimanual visual teleoperation (Agent C)

> **In this repository** (since 2026-11-05) this package is `laundry_shared.visual_teleop` and holds only the modules the laundry code imports. The text below is the package's original README: read `visual_teleop.…` as `laundry_shared.visual_teleop.…` and `src/visual_teleop/` as `src/laundry_shared/visual_teleop/`; modules it names that are not in `src/laundry_shared/visual_teleop/` are not in this tree. Provenance: [docs/provenance.md](../../../docs/provenance.md).

Robot-side tools and libraries: the take recorder, jaw classifier, reBot IK / limits / safety, bridge client, and the
open-loop replay planner (its original purpose, live visual teleoperation, is parked — see the status note below).

**Status:** current (recorder, jaw classifier, robot tools); live teleoperation parked; stereo / trackcam / fingertip tools superseded

## Inputs → Outputs
Orbbec RGB-D, wrist cameras, IMUs, reBot bridge `/status` → recorded takes (`datasets/tag_rig/<take>`), jaw state
(`umi_jaw.npz`), IK solutions and replayed / commanded robot trajectories (`datasets/replay/<run>`).

## Entry points
Run from the repo root with `src/` importable (`pip install -e .` or `PYTHONPATH=src`):
```bash
python -m laundry_shared.visual_teleop.tools.tag_rig_record --proto task --wrist      # recorder (sudo for the Orbbec on macOS)
python -m laundry_shared.visual_teleop.tools.umi_jaw infer <take>/wrist               # jaw state
python -m laundry_deploy.robot_service.mit_bridge                                     # reBot MIT bridge (was tools.bridge_start + robot_service_mit.py)
python -m laundry_shared.visual_teleop.tools.umi_replay_bi <take> ... --offline|--go  # plan + check / replay
```
Most tools also import `ego_teleop` (`src/ego_teleop/`, shared with the cube-stacking project), e.g. `umi_replay_bi` and the live-teleop pipeline.
`config/` stays inside the package: `config.py` reads `config/*.yaml` next to itself and `tools/calib_right.py` writes
`config/calibration.yaml`. Files outside the package (`configs/`, `datasets/`, `.venv`) are found from the repo root,
`Path(__file__).parents[3]`.

## Layout
- `tools/` — command-line tools (recorder, jaw, calibration, replay, bridge start, probes)
- `robot/` — IK (`ik.py`, `ik_pink.py`), bridge client, dual-arm controller
- `safety/`, `retargeting/`, `fusion/`, `imu/`, `vision/`, `gripper/`, `recording/`, `ui/` — live-teleop pipeline pieces
- `config/` — teleop / calibration yaml; `assets/` — tag image; `reports/` — calibration, latency, validation reports

## Provenance
Moved from top-level `visual_teleop/` to `src/visual_teleop/` in the 11-01 layout refactor; root lookups changed from
`parents[2]` to `parents[3]`. Imports `ego_teleop` / `ego_collector` / `handumi_collector` in places — those packages are in `src/`
too (identical copies shared with the cube-stacking project, [`docs/provenance.md`](../../../docs/provenance.md)).

## Details (original package README)


> **Status 10-27.** Live visual teleoperation (the original purpose of this package, 10-14 → 10-16) is **parked**.
> Still used by the current pipeline: `tools/tag_rig_record` (recorder), `tools/umi_jaw` (jaw classifier),
> `tools/robot_cam_poses`, `tools/bridge_start`, `tools/ready_pose`, `robot/` (IK, limits, safety); by the replay track:
> `tools/umi_replay_bi`, `tools/umi_replay`. ~~Stereo / trackcam / fingertip / teacher tools~~ (`stereo_*`, `trackcam_*`,
> `fingertip_node`, `teacher_record`, `live_teleop`, `calib_tracker`, `mock_tracker`) are not used any more; they stay here
> because other modules in this package still import them.

Fixed RGB-D + wrist RGB + wrist IMU → relative bimanual SE(3) → reBot IK + safety → robot. The robot-executed trajectory
is recorded as `VISUAL_TELEOP` data in the common 20-D action contract.

This package is new (2026-10-14). It reuses the tested pieces of `ego_teleop/` and `handumi_collector/` rather than
copying them:

- Cartesian limiter
- joint limits and feasibility gate
- bimanual 14-vector and leader-frame conversion
- gripper polarity
- RGB-D hand front-end
- Teensy/XIAO IMU driver
- clock fit and camera↔IMU cross-correlation
- UMI contract exporter

## Processes

The required packages live in three interpreters, so the system runs as three processes. They share
`time.monotonic_ns()` as the host clock.

| process | interpreter | does |
|---|---|---|
| `tools/fingertip_node.py` | `~/finger_distill/.venv` (ultralytics 8.4) | wrist cams → student fingertips → pinch metric (UDP 47811) |
| `tools/sensor_hub.py` | `ego_collector/.venv` (`sudo`, Orbbec) | IMUs → AHRS; Orbbec → wrists in W; fusion → hand (UDP 47810) |
| `tools/live_teleop.py` | `~/xvla-mac` (pyroki) | pinch hysteresis, dual-arm controller, bridge, recorder, dashboard :8730 |

```bash
cd ~/ego_collector
~/finger_distill/.venv/bin/python -m visual_teleop.tools.fingertip_node
sudo .venv/bin/python -m visual_teleop.tools.sensor_hub
~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 1            # PREVIEW: real /observe, no POST
~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 1 --live     # grippers move (operator present)
```

**Offline, with no hardware:**
- `.venv/bin/python -m visual_teleop.tools.stage0_replay --synthetic`
- `--ik pyroki` runs the same replay with real IK; use `~/xvla-mac`.
- `live_teleop --mock-robot` runs the control loop without a robot.

## Stages (spec §30) and what gates them

| stage | moves | refuses unless |
|---|---|---|
| 0 | nothing | — |
| 1 | grippers only (`arms_enabled` false) | `--live`, deadman held, INIT, no E-STOP |
| 2/3 | one arm (`--arm`) + its gripper | every calibration in `config.unmeasured()` measured |
| 4 | both arms | same |

## Control law (one tick, `robot/dual_arm_controller.py`)

```
human T_W_H (fusion) -> relative retarget (adapter + scaling) -> filter.propose -> limits.propose (NaN, jump, workspace, v/a)
  -> IK seeded with the last EXECUTED q -> feasibility (residual, bridge limits, joint step) -> inter-arm check (both HOLD)
  -> gripper (hysteresis -> rate limit -> LPF) -> ONE /execute_step vector -> accepted? -> ONLY THEN commit every state
```

Mode precedence: ESTOP > NOT_INITIALISED > DEADMAN_RELEASED > CLUTCH > HOLD (tracking/IK/safety, both arms) > FOLLOW.

**Frames (measured 2026-10-14):** the pyroki IK TCP is defined as follows.
- `z` is the tool/approach axis: it points from the wrist to the fingertips.
- `y` is the jaw opening axis.
- j6 rolls about `z`.

The human virtual TCP `H` is defined with the same axes, so the adapter is the identity. See `config/teleop_rebot.yaml`.

**Code structure:** the unit tests (`tests/vteleop/`, spec §37) are not included in this repository.

**Reports:** see `reports/` for the calibration, latency and validation reports.
