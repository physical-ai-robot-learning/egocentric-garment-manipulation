# UMI hand (HandUMI) hardware

> **This is not our design.** HandUMI was designed and published by **BrikHMP18** (Apache-2.0):
> original hardware [murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw) (first published as
> `robonet-ai/handumi-hw`), original software [murobotics-ai/handumi-sw](https://github.com/murobotics-ai/handumi-sw).
> This repository is our **build copy**: the upstream print files, STEP and BOM from commit `e58de33` as we used them, plus
> our own additions only: three modified parts (`mods/`) and the wrist-IMU firmware (`firmware/`). See `LICENSE` and `NOTICE.md`.

3D-printing files and hardware for building the hand-worn UMI gripper (HandUMI) used to record human demonstrations for the
reBot B601. Software, data pipeline and reports live in the `egocentric` repository.

HandUMI is an open-source, hand-worn variant of UMI for robot arms with parallel-jaw grippers, by BrikHMP18 (Apache-2.0):
thumb + index/middle cradles, a natural pinch drives the jaw, the jaw width is read from a Feetech servo encoder, detachable
robot-specific tips, and upstream wrist SE(3) comes from a PICO 4 Ultra / Quest 3 controller. We use it as the human-side
"skill capture gripper" for the reBot B601 instead of designing our own. The base design is
**[murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw)** (released 2026-08-20 as `robonet-ai/handumi-hw`;
that URL now redirects). Upstream has since moved to an IMX335 camera and a per-pair BOM (`fc7462a`); we did not follow.
How we built and changed it is described in Sec. III of the Part I full report in the `egocentric` repository. Our print package was generated from commit `e58de33` of that repo. The upstream parts in
`print/` are unchanged; **our modified parts (controller-support connector with a +10 mm plate, camera mount, servo cover)
are in `mods/`**. A reBot-specific gripper tip is still to be designed (see below).
License text and attribution: `LICENSE` (Apache-2.0, upstream handumi-hw) and `NOTICE.md`.

## What is here

| path | content |
|---|---|
| `print/README.md` | Plate overview, print order and fit notes for the full package (both hands plus the tip) |
| `print/left/`, `print/right/` | Four Bambu Studio 3MF projects per hand (`A_fitcheck`, `B_main_support`, `C_mounts`, `D_small_parts`), `stl/` (12 parts per hand), and a per-hand README with per-part time and PLA |
| `mods/` | **Our modified parts** (Shapr3D): controller-support connector + plate y +10 mm, camera mount with an extra window, servo cover with a wider cable opening — see `mods/README.md` |
| `cad/STEP/` | Upstream STEP sources (left/right HandUMI, gripper tips) — edit these, not the STLs |
| `bom/` | Upstream bill of materials (README, PDF, DOCX) |
| `tags/` | AprilTag sheets: UMI rigs 200–202 / 210–212, robot gripper 220–222 / 230–232, corner board id 300–311 (added 10-20) |
| `firmware/` | Wrist IMU firmware: `teensy_imu/` (Teensy 4.1 + ICM-42688-P, protocol in `PROTOCOL.md`), `xiao_imu/` variant, `bringup/` probes; `docs/IMU_BRINGUP.md` |
| `print/tips/` | `E_piper_generic` 3MF plus the L/R AgileX-Piper tip STLs. This is a **temporary** generic parallel-jaw tip for mechanism and pinch tests. A reBot tip will replace it. |

The STLs are the upstream `hardware/STL/{left,right}_handumi` and `gripper_tips` meshes. The geometry is unchanged; each
mesh is only translated so all coordinates are positive, because the originals have negative Z and the Bambu CLI
refuses them ("no object fully inside"). Edit the STEP sources in `cad/STEP/`, not these STLs.

## Build summary

- **Print**: PLA, Bambu 0.20 mm Standard, 2 walls, 20 % infill. There are 9 plates and 26 parts in total, about 9 h and
  about 250 g PLA (about 4 h and 110 g per hand). Supports are off everywhere except `*_controller_support` (tree
  support under the cantilevered screw tab).
- **Print `A_fitcheck` first, for both hands.** The thumb and index/middle cradles come from the upstream author's hand
  scan. If they don't fit, edit the cradle STEP files before printing anything else.
- **Fits**: MR63ZZ bearing (6 mm OD), LM4UU linear bearing (8 mm OD), 4 mm shaft. If holes are tight, use XY hole
  compensation of +0.05 to 0.1 mm instead of scaling the part. Print the finger links cradle-up with the LM4UU bores
  vertical, as arranged in the 3MF.
- **Slicing gotchas** (Bambu Studio 2.08):
  - All 3MFs are X1C projects. For the H2D, switch printer in the GUI and re-slice.
  - The H2D cannot be sliced from the CLI. The filament-to-extruder map is a per-project key inside the 3MF, not a
    preset key, so the CLI cannot set it.
  - CLI time and PLA estimates ignore supports.
- **CAD editing**: we edit the STEP sources in Shapr3D. Fusion 360 and FreeCAD were tried and rejected.

## Electronics / BOM

The upstream BOM is `bom/README.md` (also as PDF/DOCX), roughly US$110 per unit:

- Mechanical: PLA, 4x MR63ZZ, 4x LM4UU, 4 mm axle (135 mm rod), M2/M3 hardware, velcro straps.
- Jaw encoder: Feetech **STS3215** servo plus a Waveshare serial bus servo adapter. The adapter is a CH343, 1 Mbaud, with
  servo id 1 on our units. It needs its 12 V supply before the servo answers.
- Wrist camera: **Arducam 160° fisheye USB** (1080p UVC).

Our capture rig deviates from upstream in three ways:

- **No VR tracking.** The upstream design gets wrist SE(3) from a PICO/Quest controller. Our collector gets wrist
  motion from the fisheye camera plus an IMU (software in the `egocentric` repository).
- **Added wrist IMU per unit**: a Teensy 4.1 with an ICM-42688-P on SPI. Firmware is in `firmware/`
  (`teensy_imu/`, plus a `xiao_imu/` variant and the `bringup/` probes).
- **Head camera**: a Logitech C922 at 640x480@30 is the shared human/robot policy view.

The IMU bring-up procedure is `firmware/docs/IMU_BRINGUP.md`; device identities and the collector are documented in the
`egocentric` repository (`collector/README.md`).

## Changes after 2026-10-12 (current state 2026-10-20)

The units in use differ from the build above as follows. Details and numbers: `docs/UPDATE_2026-10-20.md` in the
`egocentric` repository.

- ~~**Jaw servos removed (10-13).** The Feetech STS3215 shafts broke and the servos were heavy. The jaws now move freely
  and the servo cover/adapter are not fitted. Collector hardware profile: `handumi_rgbd_noservo`.~~
- **Jaw servos back (10-29) as encoders.** Each unit again carries its STS3215 (id 1, Waveshare adapter), with **torque
  off** so it never drives the jaw while a person holds it; it is only read (~100 Hz) by the recorder
  (`src/visual_teleop/tools/tag_rig_record.py`) and converted to jaw labels by `scripts/pose/jaw_from_motor.py`.
  The right unit's stiff linkage reaches only ~0.65–0.7 of the left one's normalised opening; re-plugging a servo shifts
  its whole reading (handled by per-take re-centring).
- **Red/orange tape on both jaw tips** (first a small red X on white tape, from 10-13 15:xx solid tape). The gripper
  open/close state is measured from the wrist-camera video (tip-to-tip distance of the tape; closed ≤ 200 px,
  open ≥ 260 px at 1920 width). Keep the tape on the inner jaw faces, fully visible to the wrist camera, and avoid other
  red/orange objects near the jaws. Note: the wrist camera mount flexes relative to the jaws.
- **AprilTag rigs on each unit** (tag36h11, `tags/`): IDs 200–202 on the right unit, 210–212 on the left, seen by a
  fixed global Orbbec. Printed at "50 mm" they measure **48 mm** on our printer; measure the black square and use the
  measured size. Robot gripper tags 220–222 (right) / 230–232 (left); `aprilgrid_3x4_50mm_id300` is the corner-touch
  board for the TCP calibration. The TCP-in-rig calibration (`umi_tag_rig_v001.json`) is only valid for the tag
  placement it was measured with.
- **Wrist IMU = Seeed XIAO nRF52840 Sense Plus** on both units (right 10-16, left 10-17), replacing the Teensy 4.1 +
  ICM-42688-P (lower still noise). Firmware `firmware/xiao_imu/`; build with
  `arduino-cli compile --fqbn Seeeduino:mbed:xiaonRF52840SensePlus --build-property "build.extra_flags=-DTARGET_SEEED_XIAO_NRF52840_SENSE_PLUS"`
  (without the flag WHO_AM_I reads 0). The USB serial changes after flashing. The IMU must be rigidly fixed: a loose
  mount showed up as a drifting camera↔IMU rotation.
- A stereo pair of room-facing fisheye cameras was mounted on the right unit for wrist VIO teleop on 10-15/16. It was
  dropped on 10-16 (bracket slipped, VIO accuracy insufficient); no print files are kept for it.

## Open items

- A reBot B601 gripper tip does not exist yet. The base for it is `cad/STEP/gripper_tips`.
- The finger cradles fit the upstream author's hand. Check the fit for each operator.
