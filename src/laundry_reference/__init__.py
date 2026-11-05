"""laundry_reference: generic, sensor-free utilities for bimanual jaw-tip trajectories.

One module per pipeline component (flat since the OA-style refactor; the former sub-package path is in brackets):

* ``laundry_reference.trajio``           - CSV trajectory format helpers                          (io.trajio)
* ``laundry_reference.gap_fill``         - fill short marker gaps (lerp/PCHIP + slerp) and flag them  (pose.gap_fill)
* ``laundry_reference.relative_se3``     - T_rel = T_0^{-1} T(t) conversion and round-trip check  (pose.relative_se3)
* ``laundry_reference.robotize``         - continuous warps + per-frame displacement / quality    (retarget.robotize)
* ``laundry_reference.parity_metrics``   - RMSE / p95 / rotation error between two trajectories with lag search
                                           (evaluation.parity_metrics)
* ``laundry_reference.plot_example``     - figure of a processed trajectory                       (viz.plot_example)
* ``laundry_reference.cli``              - console-script functions named in pyproject.toml

These are small standalone re-implementations written for this repository; the code that ran on the real system is
in ``laundry_shared.visual_teleop``, ``laundry_shared.rebot_prep`` and ``scripts/laundry/``.
"""

__version__ = "0.1.0"
