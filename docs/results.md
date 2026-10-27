# Results

All results are from one physical setup (two low-cost 6-DoF arms, one fixed RGB-D camera, two hand-worn grippers) and
one demonstrator, measured in October 2026. They are reported as measured, including failures. No success rate is
reported, because no counted trials were run.

## Metric definitions

The replay is compared at three levels, all in the robot base frame and on the host clock:

| symbol | meaning |
|---|---|
| **A** | human jaw-tip trajectory, from markers + depth |
| **B** | robot command: forward kinematics (FK) of the commanded joints |
| **C** | robot: FK of the measured encoder joints |

For a comparison window:

* A is taken relative to its pose at the start of the window (median over the preceding hold), B and C relative to their
  first sample. **Relative metrics hide constant offsets**; absolute position was checked separately (tape, video).
* C is shifted by the lag L* ∈ [0, 1.5] s that minimises the mean squared distance to B, so a pure delay is not error.
* **Position RMSE / p95**: of ‖C(t + L*) − A(t)‖ over the window.
* **Rotation median / p95**: geodesic angle between the relative rotations of C and A.
* **Amplitude ratio**: RMS distance from the start, C over A (1.0 = same size of motion).
* **C − B**: robot tracking error. **B − A**: what conversion and planning changed (robotization, IK compromises).
* **Parity gates** used for the single-arm tests: RMSE < 20 mm, p95 < 30 mm, rotation median < 5°, p95 < 10°,
  amplitude ratio 0.9–1.1, no jumps. Single-axis strokes are compared on normalised progress along the human chord,
  with gates on main-axis ratio, cross-axis leakage (< 10 %), endpoint (< 20 mm), RMSE and p95.

A generic implementation of the position/rotation/lag part is in [`../scripts/parity_metrics.py`](../scripts/parity_metrics.py).

## 1. Calibration accuracy

| quantity | result | meaning |
|---|---|---|
| Fixed camera ↔ robot base | 0.98 px held-out reprojection, ≈1.3 mm lateral | how well a camera measurement maps into the robot frame (5-fold held-out poses) |
| Marker rig → jaw tip | 1–4 mm static; 1.8–4.8 mm leave-one-out; 5–7 mm RMS on an independent pivot test | how well the gripper's markers locate its jaw tip |
| Corrected robot model, absolute jaw position | within 6–9 mm of a tape measurement at the home pose | whether the robot's own model puts its jaws where they physically are |
| Human-gripper ↔ robot-gripper orientation | spread 2–9° | consistency of the tool-axis mapping |

Two robot-model errors (tool-centre-point location and one arm's mounting pose, both of several centimetres) were found
only by measuring the real robot; they were invisible to FK-based parity, which compares the model with itself.

## 2. Single-arm replay parity

| test | result |
|---|---|
| Six-direction translation strokes | 16/18 pass; median per-axis RMSE 5.5–11.6 mm (max 14.5 mm); endpoint repeatability ≤ 8 mm. Failures: one marginal cross-axis leak, one poor human stroke |
| Rotation strokes | amplitude ratio 0.98–1.04; geodesic median error 0.3–1.6°; repeat ≤ 1.4° |
| Free combined motion (near human speed) | 2/2 pass; RMSE 13.7 / 14.6 mm, p95 26 / 29 mm, rotation median 1.7 / 1.4°, ratio 0.99–1.03, robot lag ≈0.2 s |
| Error decomposition | B − A ≤ 3.4 mm: the residual is robot tracking (C − B), not conversion |
| Jaw timing | robot jaw command within 0.5 % of window progress of the human jaw event |

**Controller finding.** The moving error did not change when the proportional gain was raised. With an integral-only
feed-forward and no gravity model, a quasi-static analysis gives a moving error ≈ (rate of change of gravity torque) /
(integral gain), independent of the proportional gain; a simulation explained most of the measured error on the
shoulder joint (77 %). Earlier gain settings passed 0/6, 3/6 and 5/6 strokes of a test set; raising the integral gain
in two steps reduced the forward-stroke RMSE from 25–37 mm to 12–14 mm, and the final setting gives the 16/18 above.

## 3. First cloth grasp from human data

One arm replayed the grasp window of a demonstration and **picked up and lifted the T-shirt**. The first attempt
scraped the table at a strongly tilted tool, which led to the finger-corner clearance model in the planner.

## 4. Full two-arm replay of one demonstration

Input: one 33 s demonstration, replayed at roughly one third of human speed (94 s).

| input quality | right | left |
|---|---|---|
| frames with at least one visible marker | 79 % | 95 % |
| frames neither marker-measured nor covered by accepted SLAM (interpolated) | 21 % | — |
| robotization displacement, median / p95 / max | 0.1 / 9.3 / 10.5 mm | 1.1 / 20.6 / 21.5 mm |
| frames in the best quality class | 100 % | 92 % |

Plan: IK error ≤ 6.3 mm / 11.5° (right) and ≤ 10.3 mm / 21° (left); the two arms stayed ≥ 27 cm apart.

**Error of the replay (runs with the same plan, whole demo as one window).** Here A is the *robotized* path that was replayed; against the raw human path the RMSE is ≈22 mm (right) and 32–34 mm (left), because the robotization displacement then also counts as error.

| arm | pos. RMSE | pos. p95 | rot. median | rot. p95 | C − B | B − A |
|---|---|---|---|---|---|---|
| right | 17.2–17.4 mm | 35–37 mm | 8.1° | 12.6–12.7° | 12.1–12.3 mm | 8.6 mm |
| left | 29.0–30.3 mm | 50–51 mm | 12.3–12.7° | 17.1–17.2° | 14.5–15.5 mm | 22.6 mm |

* The runs agree within ≈1 mm: the robot is deterministic and the remaining error is systematic.
* The left command already deviates more from the replayed path (B − A 22.6 mm vs 8.6 mm): the left IK gives up more
  orientation. Against the raw human path, the larger left robotization adds to this.

## 5. Jaw height at grasp

| planner setting | jaw-close height above the table |
|---|---|
| conservative finger-corner clearance | 19–61 mm; missed the cloth |
| finger corner allowed to touch the table plane (as human fingers do) | 14–16 mm (right, 4 closes), 5–19 mm (left, 5 closes); xy within 2–8 mm of the command at that moment |
| human demonstration | ≈10 mm |

Rejected alternatives: tilting the tool toward vertical where the corner would touch (the IK could not follow even 15°;
errors up to 100 mm); closing at the lowest nearby point (a heuristic).

**Qualitative outcome.** In two runs the shirt lay 15–20 cm farther away than in the demonstration and the jaws closed on
the empty table. In one run the jaws caught the shirt edge and dragged it; the shirt was moved, but not folded as by the
human.

## 6. Negative results: SLAM as the trajectory source

The markers are hidden exactly when the hand grasps, palm down, so wrist-camera SLAM was tested as the main source.

| method | result |
|---|---|
| MASt3R-SLAM on the wrist fisheye, with automatic restarts after tracking loss | accurate where it tracks (2–14 mm on held-out marker frames), but kept only **19–48 %** of frames and **0–38 %** of the marker-hidden frames around grasps; joins between restarted segments jumped 40–73 mm |
| ORB-SLAM3 monocular-inertial | initialised with good rotation, but **translation scale diverged by 30–1000×**; hand accelerations (median 0.07–0.16 m/s²) were below the accelerometer bias (0.25–0.43 m/s²), so scale was not observable |
| Stereo VIO (earlier stereo wrist camera) | stereo-inertial diverged; stereo VO alone ran but the camera was dropped |

Consequence: SLAM is kept only as an optional filler, accepted per segment where it agrees with the markers.

An earlier label pipeline (SLAM + external body tracking + IMU) passed all of its own quality checks yet was 4–6 cm from
the jaw position seen by the external camera, because of three calibration/anchoring errors that internal residuals
could not see. This motivated the marker rig with 1–4 mm error.

## 7. Training-data pipeline (October 24–27)

**Pose from point tracking + markers + IMU.** Accuracy is measured by hiding markers that were actually visible and
comparing the estimate with them, on takes not used for tuning.

| source | hidden-marker error, median |
|---|---|
| markers + IMU interpolation | ≈ 42 mm |
| point tracking + marker anchors + IMU in gaps | 16 mm (left) / 19 mm (right) |

**Dataset.** 318 flattening demonstrations, 300 successful. Grades for the first 139: 133 GOLD / 5 SILVER / 1 REJECT
(GOLD: ≥ 95 % tracked, no gap over 1 s, ≤ 5 % of frames with hidden-marker error above 10 cm, no long error burst,
no gap at a grasp).

**Labels before vs. after IK** (first 139 demos, ≈ 123 k arm-frames). IK → FK moves the target by p95 ≈ 53 mm overall
and ≈ 58 mm at grasps. About 20 % of frames need a large IK correction; by cause: wrist-roll relaxation (mostly left),
tool flip (mostly right), one joint at its limit, and ≈ 1 % truly out of reach.

**Policy.** Not yet trained on these labels.

## 8. Synthetic example (this repository)

On the synthetic data in [`../examples/`](../examples), the tools in `scripts/` give (seed 0):

| comparison | right | left |
|---|---|---|
| gap filling: measured / interpolated / left unfilled | 91.9 / 4.8 / 3.3 % | 87.6 / 5.2 / 7.2 % |
| robotization lateral scale; displacement p95 | 0.83; 43.9 mm | 1.00; 8.4 mm |
| robotized vs. simulated replay: lag, RMSE | 0.22 s, 5.7 mm | 0.26 s, 5.8 mm |
| raw human vs. simulated replay: RMSE | 17.0 mm | 6.8 mm |

These numbers describe the synthetic generator, not the real system.
