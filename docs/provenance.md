# Provenance of vendored modules

**Current state (2026-11-05).** The vendored modules now live in `src/laundry_shared/` under their original names
(`laundry_shared.contracts`, `laundry_shared.xvla_core`, `laundry_shared.umi_to_lerobot`, `laundry_shared.visual_teleop`,
`laundry_shared.rebot_prep`, `laundry_shared.handumi_collector`, `laundry_shared.ego_collector`,
`laundry_shared.ego_teleop`). Only the modules the laundry code imports are kept (90 Python files), in their original
code style; the full shared-package copies of 11-02 and the `umi` stand-in were removed from the tree (they are in git
history before commit `3fbf473` and in egocentric-cube-stacking). Old → new paths:
[refactor_2026-11.md](refactor_2026-11.md). The sections below are the record of 11-01 / 11-02; the file paths in their
tables (`src/contracts/…`, `src/ego_teleop/…`) are those of that date.


The pipeline code imported a few modules that lived outside this repository on the original machine (the shared
action / observation contract, two dataset helpers, a UMI pose helper and parts of the earlier `ego_collector`
workspace). They were copied into `src/` (refactor v3, 2026-11-01) so that every import resolves from a clean clone.
Import names are unchanged, so the code that uses them is unchanged.

**Shared packages.** `ego_collector`, `ego_teleop`, `handumi_collector` and `umi` are in `src/` as the full packages,
identical copies of the ones in egocentric-cube-stacking; see [Shared packages](#shared-packages) (the partial copy of
2026-11-01, a separate shared package for one day, and the copy back on 2026-11-02). Also vendored here: `contracts`,
`xvla_core`, `umi_to_lerobot`.

* Copies are byte-exact at the time of copying; the md5 below is that of the **source** file. Edits made afterwards
  in this repository are listed under "Changes after copying".
* `ego_collector`, `ego_teleop` and `handumi_collector` are the HandUMI / teleop packages of the earlier cube-stacking
  project. On 2026-11-01 only the modules this repository imports at module level, plus the package `__init__` files they
  need, were copied; since 2026-11-02 the full packages are here (see [Shared packages](#shared-packages)).
* `src/umi/common/pose_util.py` is our own minimal stand-in (written 2026-10-08) for UMI's `umi/common/pose_util.py`:
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

Current as of 2026-11-05 (the stereo-VIO modules that imported `stereo_openvins_eval` / `stereo_basalt_eval` are no longer in the tree).

| import | used by | why it is not copied |
|---|---|---|
| `rel_relonly_bce` | `laundry_deploy.xvla_server` | X-VLA training patch from the cube project's X-VLA deploy tree (`~/xvla_deploy/umi_bridge/umi76`, `~/ego_cart20/launch/rel32bin`); the server runs in a separate X-VLA environment set up with `PYTHONPATH` (see the module docstring) |
| `ego_cart20.*` | `laundry_shared.umi_to_lerobot` (lazy, ego-data branches only), `archive/scripts/umi_fusion/jaw_redx.py` | cube-project package (`egocentric-cube-stacking`) |
| `lerobot_plugin_rebot.*`, `bh_indy7_lerobot*`, `usb_arm_ports` | `laundry_deploy.robot_service` | reBot driver stack of the robot PC (`~/robot-cockpit`), third-party / company code |
| `jaw_redx` | `laundry_shared.visual_teleop.vision.fingertip_detector` (red-tape detector, lazy) | lives in `archive/scripts/umi_fusion/` (tape removed 10-20) and itself needs `ego_cart20` |

The modules `ego_teleop.hand3d.*`, `ego_collector.hands.mediapipe_tracker` and `ego_teleop.tracking.backends.openvins`
(imported lazily by `visual_teleop.vision.rgbd_pose`, `visual_teleop.tools.sensor_hub` and
`visual_teleop.vision.stereo_vio_tracker`) were in this table while `src/` held only the partial copies; they
now resolve from the full packages in `src/` (their optional dependencies, e.g. MediaPipe, `pyorbbecsdk` or the OpenVINS bridge, still
apply).

## Shared packages

`handumi_collector`, `ego_collector`, `ego_teleop` and `umi` are the HandUMI / teleop packages of the cube-stacking
project ([egocentric-cube-stacking](https://github.com/physical-ai-robot-learning/egocentric-cube-stacking)). Both repositories hold
them in `src/` as identical copies: every Python file has the same bytes in both (236 files; md5 list in
[`archive/docs/shared_packages_MANIFEST.md`](../archive/docs/shared_packages_MANIFEST.md)). Only the three package `README.md` files differ (each
repository's own links). Import names are unchanged.

History:

1. **2026-11-01 (refactor v3).** The import closure of what this repository uses (34 files, table below) was copied here.
2. **2026-11-01, later.** The copies were removed and the full packages were installed from a separate shared package
   (a git dependency in `pyproject.toml`) that held them once for both repositories.
3. **2026-11-02: copied back from the shared package (v0.3.0) because that repository is private.** A public clone could
   no longer install this repository. The four full packages were copied back from release v0.3.0 into `src/`,
   byte-identical (239 files: 236 `.py`, 2,009,728 bytes, plus 3 `README.md` rewritten for this repository), and the git
   dependency was removed. The full packages were chosen over the 34-file closure: the robot launcher
   (`~/laundry_dit_training/deploy/_garment_repo.py`) puts this `src/` first on `sys.path`, so a partial `ego_teleop`
   here would hide the modules it does not contain; with the full packages every import resolves to the same code as with
   the shared package. `pyproject.toml` gained the packages' extras (`collector`, `experimental`, `apriltag`, `viz`,
   `aero`, from the shared package's `pyproject.toml`; `collector` uses `opencv-python<5` instead of
   `opencv-contrib-python` so one environment keeps one OpenCV wheel).

Differences between the 34 files copied on 2026-11-01 and the files now in `src/`: `handumi_collector/config.py` sets
`REPO_ROOT = parents[2]` (the repository root in the `src/` layout) instead of the `parents[1]` of the flat source tree;
this repository only uses `ImuCfg` from it. `ego_teleop/robot/__init__.py` is the source file in full, i.e. with the
`aero_client` / `coordinator` re-exports that the partial copy had removed (the full package contains those modules).
All other files are byte-identical to their source (md5 below). Where this repository's copy had been newer than the
cube-stacking copy (`ego_collector/camera/orbbec_capture.py`, `ego_teleop/robot/rebot_client.py`,
`ego_teleop/robot/safety.py`, `ego_teleop/transforms/frames.py`) the shared packages carry this repository's version, so
the code this repository runs is unchanged.

Tests (fresh Python 3.12 environment, `pip install -e ".[test]"` and `opencv-python-headless<5`, as in CI):
`python -m pytest -q` gives 348 passed, 103 skipped. Before (same environment, the shared package on the path instead of
`src/`): 186 passed, 37 skipped. The difference is the import smoke test (`tests/test_imports.py`), which now also
walks the 228 modules of the shared packages: 162 import, 66 skip for an optional dependency, and two scripts that run
at import (`ego_collector.__main__`, `handumi_collector.tools.metric_qa`) are listed as not imported. The 34 modules of
the 2026-11-01 copy give the same result as then (31 import, 3 skip for pandas).

Record of the 34 files of the 2026-11-01 partial copy (source on the original machine, md5 of the source):

| file | source (original machine) | md5 of source |
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
