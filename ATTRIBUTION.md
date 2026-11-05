# Attribution

This repository's own code, configurations, documentation and media are licensed for noncommercial use only under the
PolyForm Noncommercial License 1.0.0 (see [LICENSE](LICENSE)). The exception is `hardware/umihand/`, a build copy of
the upstream HandUMI design, which keeps its Apache-2.0 licence ([LICENSE](hardware/umihand/LICENSE),
[NOTICE](hardware/umihand/NOTICE.md)). The project builds on the following work, which keeps its own licence and credit.

## Hardware

* **HandUMI** — hand-worn parallel-jaw gripper for robot-free bimanual data capture. **Designed and published by
  BrikHMP18 (murobotics-ai), Apache-2.0: <https://github.com/murobotics-ai/handumi-hw>. It is not our design.** The
  project uses a build of the upstream design with small modifications (wrist IMU instead of VR tracking, several
  AprilTags per hand, jaw servo removed 13–28 Oct and reinstalled 29 Oct as a torque-off jaw encoder). The build copy
  (upstream print files, BOM, CAD) and our modified parts and IMU firmware are in [`hardware/umihand/`](hardware/umihand/README.md),
  under the upstream Apache-2.0 licence ([LICENSE](hardware/umihand/LICENSE), [NOTICE](hardware/umihand/NOTICE.md)).
  B. Meza et al., "HandUMI: A modular hand-worn parallel-jaw interface for robot-free bimanual data capture," 2026,
  <https://murobotics.ai/research/handumi>.
* **Universal Manipulation Interface (UMI)** — C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake,
  S. Song, "Universal Manipulation Interface: In-the-wild robot teaching without in-the-wild robots," RSS 2024. The
  hand-held-gripper approach that HandUMI and this project follow.
* **reBot B601-DM** arms and **Orbbec Gemini 336** RGB-D camera: commercial products, used as is.

## Methods and software used in the project

| name | use | reference |
|---|---|---|
| AprilTag (tag36h11) | fiducial markers | E. Olson, ICRA 2011; J. Wang and E. Olson, IROS 2016 |
| OpenCV | marker detection, PnP | G. Bradski, Dr. Dobb's Journal 2000 |
| IPPE | single planar-marker pose | T. Collins and A. Bartoli, IJCV 2014 |
| SQPnP | multi-marker pose initialisation | G. Terzakis and M. Lourakis, ECCV 2020 |
| Robot-world/hand-eye calibration | initialisation of camera-to-robot calibration | M. Shah, J. Mechanisms and Robotics 2013 |
| CoTracker3 | point tracking on the fixed camera (current pose source) | N. Karaev et al., "CoTracker3: Simpler and better point tracking by pseudo-labelling real videos," 2024 |
| MASt3R-SLAM (with MASt3R, DUSt3R) | evaluated as pose source and as a gap filler; rejected (now in `archive/`) | R. Murai, E. Dexheimer, A. J. Davison, CVPR 2025; V. Leroy et al., ECCV 2024; S. Wang et al., CVPR 2024 |
| ORB-SLAM3 | evaluated (mono-inertial); rejected | C. Campos et al., IEEE T-RO 2021 |
| Basalt, OpenVINS, AirSLAM | evaluated with an earlier stereo camera | V. Usenko et al., RA-L 2020; P. Geneva et al., ICRA 2020; K. Xu et al., T-RO 2025 |
| Kalibr | camera–IMU calibration in an earlier pipeline | P. Furgale, J. Rehder, R. Siegwart, IROS 2013 |
| Pink / Pinocchio | differential inverse kinematics | S. Caron et al., Pink (Apache-2.0); J. Carpentier et al., SII 2019 |
| ResNet-18 / ImageNet | jaw-state classifier backbone | K. He et al., CVPR 2016; J. Deng et al., CVPR 2009 |
| LeRobot | robot software ecosystem used by the robot driver; LeRobot v3 dataset format and the DiT / X-VLA training code | R. Cadene et al., 2024 |
| Mahony filter | IMU rotation in the pose fusion | R. Mahony, T. Hamel, J.-M. Pflimlin, IEEE TAC 2008 |
| Ultralytics YOLO (pose) | jaw-tip keypoint model (AGPL-3.0) | Ultralytics, https://github.com/ultralytics/ultralytics |
| Diffusion Transformer (DiT) | policy architecture (Multi-Task DiT in LeRobot) | W. Peebles and S. Xie, ICCV 2023 |
| CLIP | text encoder of the DiT policy | A. Radford et al., ICML 2021 |
| X-VLA | VLA base model (`lerobot/xvla-base`) | https://huggingface.co/lerobot/xvla-base |
| diffusers | dependency of the LeRobot policy and training code | Hugging Face, https://github.com/huggingface/diffusers |

## Related work cited

DexUMI (M. Xu et al., CoRL 2025), EgoMimic (S. Kareer et al., ICRA 2025), ALOHA (T. Z. Zhao et al., RSS 2023),
SpeedFolding (Y. Avigal et al., IROS 2022), FlingBot (H. Ha and S. Song, CoRL 2021).

## This repository

The full project code is here: one package per pipeline stage in `src/` (`laundry_tracking`, `laundry_jaw`,
`laundry_dataset`, `laundry_training`, `laundry_deploy`, `laundry_common`), operational scripts in `scripts/laundry/`,
calibration and robot files in `configs/laundry/`, and superseded code in `archive/` (merged on 1 November 2026 from the
project's former working repository, `egocentric-laundry-folding`; restructured on 5 November 2026, see
[docs/refactor_2026-11.md](docs/refactor_2026-11.md)). The code in `src/laundry_reference/` and the generator in
`assets/laundry/examples/` are small standalone re-implementations written for this repository; they depend only on
numpy, scipy and matplotlib. The data in `assets/laundry/examples/` is synthetic.

Modules copied from the project's other working folders so that imports resolve from a clean clone are in
`src/laundry_shared/` (`contracts`, `xvla_core.py`, `umi_to_lerobot.py`, and the modules this code imports from
`visual_teleop`, `rebot_prep`, `ego_collector`, `ego_teleop` and `handumi_collector`); they are our own code, and
sources and md5 are listed in [docs/provenance.md](docs/provenance.md). The action contract
(`laundry_shared.contracts`) follows the rot6d convention of UMI's pose helper (first two rows of R); UMI's MIT licence
is kept in [upstream/umi/LICENSE](upstream/umi/LICENSE) ([upstream/README.md](upstream/README.md)).

Pretraining data named in the training plan: Soft-Fold, iPhUMI (MIT), PrimeBot UMI (CC BY-SA 4.0); not redistributed
here.

* **ORB-SLAM3** (UZ-SLAMLab, GPL v3): `archive/orbslam3/patch*.py` modify ORB-SLAM3's source and are therefore
  distributed under the GPL v3, not under this repository's licence; see
  [archive/orbslam3/LICENSE_NOTICE.md](archive/orbslam3/LICENSE_NOTICE.md). ORB-SLAM3 itself is not included.
