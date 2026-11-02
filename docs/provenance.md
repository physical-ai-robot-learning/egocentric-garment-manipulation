# Provenance of vendored modules

The pipeline code imported a few modules that lived outside this repository on the original machine (the shared
action / observation contract, two dataset helpers, a UMI pose helper and parts of the earlier `ego_collector`
workspace). They were copied into `src/` (refactor v3, 2026-11-01) so that every import resolves from a clean clone.
Import names are unchanged, so the code that uses them is unchanged.

**Shared package (2026-11-01, later the same day).** The copies of `ego_collector`, `ego_teleop`, `handumi_collector` and
`umi` were removed from `src/`. These packages now come from the shared package [`handumi-common`](https://github.com/physical-ai-vla/handumi-common) (v0.1.0;
`pyproject.toml`: `handumi-common @ git+https://github.com/physical-ai-vla/handumi-common@v0.3.0`), which holds their full
versions once for this repository and egocentric-cube-stacking; see [Moved to the shared package](#moved-to-the-shared-package).
Still vendored here: `contracts`, `xvla_core`, `umi_to_lerobot`.

* Copies are byte-exact at the time of copying; the md5 below is that of the **source** file. Edits made afterwards
  in this repository are listed under "Changes after copying".
* `ego_collector`, `ego_teleop` and `handumi_collector` are the HandUMI / teleop packages of the earlier cube-stacking
  project. Only the modules this repository imports at module level, plus the package `__init__` files they need, were
  copied — not the whole packages (now replaced by the shared package, see above).
* `umi/common/pose_util.py` (formerly `src/umi/`, now in the shared package) is our own minimal stand-in (written 2026-10-08) for UMI's `umi/common/pose_util.py`:
  same function names and conventions (pose = xyz + rotation vector; rot6d = first two rows of R; Gram–Schmidt back
  to R) for the five functions we use. It is not a copy of upstream code; UMI's MIT licence and source are recorded in
  [`upstream/README.md`](../upstream/README.md) because the API and conventions follow it.

Sources: `~/laundry_system` (git `5006650`), `~/laundry_dit_training` (git `765008c`), `~/softfold_infer` (no git),
`~/ego_collector` (git `b7930d74` + working tree; files copied as they were on disk).

| file in this repository | source (original machine) | md5 of source |
|---|---|---|
| `src/contracts/__init__.py` | `~/laundry_system/contracts/__init__.py` | `b2f5b2c9b04ffbfadca26ce6b15d34a6` |
| `src/contracts/action_contract.py` | `~/laundry_system/contracts/action_contract.py` | `8e60112b54f2ba3763ce964b623f41a2` |
| `src/contracts/observation_contract.py` | `~/laundry_system/contracts/observation_contract.py` | `a8fa917daa2c8de43f02b13a968b3a0e` |
| `src/contracts/policy_protocol.py` | `~/laundry_system/contracts/policy_protocol.py` | `fa0dba4da9a890cfa38f58eef96040f4` |
| `src/contracts/skill_contract.py` | `~/laundry_system/contracts/skill_contract.py` | `edfcd2cc82c6f41d85dbf2418ebf0e98` |
| `src/xvla_core.py` | `~/laundry_dit_training/datasets/xvla_core.py` | `17f5af4c858a424e71ba26e10421e7d8` |
| `src/umi_to_lerobot.py` | `~/laundry_dit_training/datasets/umi_to_lerobot.py` | `4d39624a53ff90e1fbf2c320be49732e` |

## Changes after copying

| file | change | why |
|---|---|---|
| `src/xvla_core.py` | the two lines that put `~/laundry_system` on `sys.path` removed | `contracts` is now `src/contracts` |
| `src/umi_to_lerobot.py` | of the three `sys.path` inserts (own folder, `~/laundry_system`, `~/ego_cart20`) only `~/ego_cart20` kept | `xvla_core` and `contracts` are now in `src/`; `ego_cart20` (cube project) is only needed by the ego-data branches |

## Imports that are still not resolvable from this repository

| import | used by | why it is not copied |
|---|---|---|
| `rel_relonly_bce` | `garment_pipeline.eval_deploy.xvla_server` | X-VLA training patch from the cube project's X-VLA deploy tree (`~/xvla_deploy/umi_bridge/umi76`, `~/ego_cart20/launch/rel32bin`); the server runs in a separate X-VLA environment set up with `PYTHONPATH` (see the module docstring) |
| `ego_cart20.*` | `umi_to_lerobot` (lazy, ego-data branches only), `ego_teleop.ego_cart20_link` (shared package, lazy), `legacy/scripts/umi_fusion/jaw_redx.py` | cube-project package (`egocentric-cube-stacking`) |
| `lerobot_plugin_rebot.*`, `bh_indy7_lerobot*`, `usb_arm_ports` | `scripts/deploy/robot_bridge/` | reBot driver stack of the robot PC (`~/robot-cockpit`), third-party / company code |
| `stereo_openvins_eval`, `stereo_basalt_eval` | `visual_teleop.vision.stereo_vio_tracker` | stereo VIO experiment (dropped 10-16, see `legacy/`), needs native OpenVINS / Basalt bridges |
| `jaw_redx` | `visual_teleop.vision.fingertip_detector` (red-tape detector, lazy) | lives in `legacy/scripts/umi_fusion/` (tape removed 10-20) and itself needs `ego_cart20` |

The modules `ego_teleop.hand3d.*`, `ego_collector.hands.mediapipe_tracker` and `ego_teleop.tracking.backends.openvins`
(imported lazily by `visual_teleop.vision.rgbd_pose`, `visual_teleop.tools.sensor_hub` and
`visual_teleop.vision.stereo_vio_tracker`) were in this table until the shared package replaced the partial copies; they
now resolve from `handumi-common` (their optional dependencies, e.g. MediaPipe, `pyorbbecsdk` or the OpenVINS bridge, still
apply).

## Moved to the shared package

Record of the 34 files (31 of `handumi_collector` / `ego_collector` / `ego_teleop`, 3 of `umi`) that were copied here (source on the original machine, md5 of the source) and are now provided
by [`handumi-common`](https://github.com/physical-ai-vla/handumi-common). The shared package has these sources byte-identical, with one exception:
`handumi_collector/config.py` sets `REPO_ROOT = parents[2]` (the repository root in the `src/` layout) instead of the
`parents[1]` of the flat source tree; this repository only uses `ImuCfg` from it. `ego_teleop/robot/__init__.py` is the
source file in full, i.e. with the `aero_client` / `coordinator` re-exports that the copy here had removed (the shared
package contains those modules). The four files where this copy was newer than the cube-stacking copy
(`ego_collector/camera/orbbec_capture.py`, `ego_teleop/robot/rebot_client.py`, `ego_teleop/robot/safety.py`,
`ego_teleop/transforms/frames.py`) are the versions below, so the code this repository runs is unchanged; the shared
package's [`docs/provenance.md`](https://github.com/physical-ai-vla/handumi-common/blob/main/docs/provenance.md) lists every difference that was reconciled.

| file (was in this repository) | source (original machine) | md5 of source |
|---|---|---|
| `src/umi/__init__.py` | `~/softfold_infer/rebot/umi/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` |
| `src/umi/common/__init__.py` | `~/softfold_infer/rebot/umi/common/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` |
| `src/umi/common/pose_util.py` | `~/softfold_infer/rebot/umi/common/pose_util.py` | `63731316f6b46e0958326817f9df5748` |
| `src/ego_collector/__init__.py` | `~/ego_collector/ego_collector/__init__.py` | `dc71c01a389ac7cec1ee6e4282b68923` |
| `src/ego_collector/camera/__init__.py` | `~/ego_collector/ego_collector/camera/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` |
| `src/ego_collector/camera/calibration.py` | `~/ego_collector/ego_collector/camera/calibration.py` | `36c56465d2ffa1c4c3a21619897f9360` |
| `src/ego_collector/camera/capture.py` | `~/ego_collector/ego_collector/camera/capture.py` | `6d2d88f004c58854a10fa927bb66b484` |
| `src/ego_collector/camera/intrinsics.py` | `~/ego_collector/ego_collector/camera/intrinsics.py` | `fd0836cb97a74d8e699638023a0dcf98` |
| `src/ego_collector/camera/orbbec_capture.py` | `~/ego_collector/ego_collector/camera/orbbec_capture.py` | `f21c526dbc86ffe84a9a113beeba871b` |
| `src/ego_teleop/__init__.py` | `~/ego_collector/ego_teleop/__init__.py` | `31c29ff13fee322f63a7340d22840a8f` |
| `src/ego_teleop/ego_cart20_link.py` | `~/ego_collector/ego_teleop/ego_cart20_link.py` | `e82e4e48ae592077c86f4bf022fc9d9c` |
| `src/ego_teleop/recorder/__init__.py` | `~/ego_collector/ego_teleop/recorder/__init__.py` | `a62cdfdbf6992196daa4b047380ef9af` |
| `src/ego_teleop/recorder/episode_logger.py` | `~/ego_collector/ego_teleop/recorder/episode_logger.py` | `5a0689e4e7032c4aa942c0d53a5f40b4` |
| `src/ego_teleop/recorder/umi_contract_export.py` | `~/ego_collector/ego_teleop/recorder/umi_contract_export.py` | `f11d5db177003151ca52134f9a30c309` |
| `src/ego_teleop/robot/__init__.py` | `~/ego_collector/ego_teleop/robot/__init__.py` | `e0dcc72cdf055dcb99887b8ed2dcceb5` |
| `src/ego_teleop/robot/bimanual.py` | `~/ego_collector/ego_teleop/robot/bimanual.py` | `32d7c88d2e4bf13224110468edbfacfd` |
| `src/ego_teleop/robot/deadman.py` | `~/ego_collector/ego_teleop/robot/deadman.py` | `da11423402a76d9e5c96d45f49b9b822` |
| `src/ego_teleop/robot/feasibility.py` | `~/ego_collector/ego_teleop/robot/feasibility.py` | `445237a5d9ddae4d5e87bf81b71b626e` |
| `src/ego_teleop/robot/rebot_client.py` | `~/ego_collector/ego_teleop/robot/rebot_client.py` | `1bd869130e83e75bc40767240d7cb02c` |
| `src/ego_teleop/robot/safety.py` | `~/ego_collector/ego_teleop/robot/safety.py` | `2f66914713d351e6abba082cc66b67fa` |
| `src/ego_teleop/transforms/__init__.py` | `~/ego_collector/ego_teleop/transforms/__init__.py` | `ddf8e0f9673a2eb438d8f1dbf3ea0f93` |
| `src/ego_teleop/transforms/frames.py` | `~/ego_collector/ego_teleop/transforms/frames.py` | `6f8d2505b4dcd3b49aa9ef441a249ab8` |
| `src/ego_teleop/transforms/se3.py` | `~/ego_collector/ego_teleop/transforms/se3.py` | `ab2a2875c690fa813af7fb96d50a9d1a` |
| `src/ego_teleop/transforms/umi_robot_frame.py` | `~/ego_collector/ego_teleop/transforms/umi_robot_frame.py` | `6933a5b47e449996950319078579d030` |
| `src/handumi_collector/__init__.py` | `~/ego_collector/handumi_collector/__init__.py` | `41069146f609f29af7530bff9b23bc25` |
| `src/handumi_collector/config.py` | `~/ego_collector/handumi_collector/config.py` | `a6d1616ecc37a85064aaf786174fe6f5` |
| `src/handumi_collector/devices/__init__.py` | `~/ego_collector/handumi_collector/devices/__init__.py` | `dbd3e01b1ecb9cf77275b0254efe746d` |
| `src/handumi_collector/devices/base.py` | `~/ego_collector/handumi_collector/devices/base.py` | `21d2c49232e2d8d21cd96a7b4940cc23` |
| `src/handumi_collector/devices/feetech_bus.py` | `~/ego_collector/handumi_collector/devices/feetech_bus.py` | `e0008810ee07e53ab58a8308ae34f0e5` |
| `src/handumi_collector/devices/teensy_imu.py` | `~/ego_collector/handumi_collector/devices/teensy_imu.py` | `b5c5885a42c176bf39c13473b524ff1c` |
| `src/handumi_collector/pose/__init__.py` | `~/ego_collector/handumi_collector/pose/__init__.py` | `876c2bb5f48f9890e75ee4fbe1ba6f56` |
| `src/handumi_collector/pose/estimator.py` | `~/ego_collector/handumi_collector/pose/estimator.py` | `755dccb672dae4aabae41e86cd9e13db` |
| `src/handumi_collector/pose/se3.py` | `~/ego_collector/handumi_collector/pose/se3.py` | `7039a6f7e06e29535757eb975097c6db` |
| `src/handumi_collector/pose/timing.py` | `~/ego_collector/handumi_collector/pose/timing.py` | `c7cceefabc8db408d33893725dd567fc` |
