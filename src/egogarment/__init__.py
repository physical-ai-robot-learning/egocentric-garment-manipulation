"""egogarment: generic, sensor-free utilities for bimanual jaw-tip trajectories.

Submodules (one per pipeline component):

* ``egogarment.io.trajio``                   - CSV trajectory format helpers
* ``egogarment.pose.gap_fill``               - fill short marker gaps (lerp/PCHIP + slerp) and flag them
* ``egogarment.pose.relative_se3``           - T_rel = T_0^{-1} T(t) conversion and round-trip check
* ``egogarment.retarget.robotize``           - continuous warps + per-frame displacement / quality
* ``egogarment.evaluation.parity_metrics``   - RMSE / p95 / rotation error between two trajectories with lag search
* ``egogarment.viz.plot_example``            - figure of a processed trajectory

These are small standalone re-implementations written for this repository; the code that ran on the real system is
in ``visual_teleop``, ``rebot_prep`` and ``scripts/``.
"""

__version__ = "0.1.0"
