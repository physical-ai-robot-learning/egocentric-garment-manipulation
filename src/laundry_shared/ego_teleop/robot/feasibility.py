"""Kinematic feasibility gate for one arm (2026-10-13, Stage 0 / Preview). Wraps the checks HttpRebotClient does inline
(NaN, IK residual, JointLimits incl. the per-command joint step) into ONE verdict, so an offline replay and the live client
refuse the same commands. Limits default to the MIT bridge SEEED_LIM (rebot_client.bridge_joint_limits_rad, margin 0).

Any failure => HOLD (nothing sent; the safety pipeline keeps the last EXECUTED command, see safety.ArmSafetyPipeline).

TODO (not in P0, documented plan):
  * self-collision between the two arms (pyroki collision support UNVERIFIED) -> `self_collision` stays "unknown";
  * table collision on the JAW TIP (FK TCP is ~82 mm ahead of the tip per memory hra-shared-base-frame, UNVERIFIED), needs a
    measured T_RB_table; until then only the workspace box floor guards the table."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .rebot_client import JointLimits


@dataclass
class FeasibilityConfig:
    max_ik_pos_err_m: float = 0.01
    max_ik_rot_err_deg: float = 5.0
    limits: JointLimits = field(default_factory=JointLimits)          # bridge SEEED_LIM, max_step_rad 0.05
    # The IK's soft URDF limits overshoot by float noise (Stage 0: j2 = +0.000..0.008 deg past its 0 bound, which is the SAME
    # bound in URDF and bridge). Within this tolerance q is PROJECTED onto the limit (so the command is inside SEEED_LIM and
    # the bridge never clips); beyond it the command is refused as before.
    projection_tol_deg: float = 0.05


@dataclass
class FeasibilityVerdict:
    ok: bool
    reasons: list[str]
    q: np.ndarray | None
    ik_pos_err_m: float
    ik_rot_err_deg: float
    joint_step_rad: float
    self_collision: str = "unknown"          # TODO P1
    table_clearance: str = "unknown"         # TODO P1 (jaw tip vs measured table plane)
    projected: bool = False                  # q was projected onto a joint limit (within projection_tol_deg)


class FeasibilityGate:
    def __init__(self, cfg: FeasibilityConfig | None = None) -> None: self.cfg = cfg or FeasibilityConfig()

    def check(self, q_prev_exec: np.ndarray, q_new, ik_pos_err_m: float, ik_rot_err_deg: float) -> FeasibilityVerdict:
        q_prev = np.asarray(q_prev_exec, np.float64)
        if q_new is None or not np.all(np.isfinite(np.asarray(q_new, np.float64))):
            return FeasibilityVerdict(False, ["ik_nan"], None, float(ik_pos_err_m), float(ik_rot_err_deg), float("nan"))
        q = np.asarray(q_new, np.float64).copy(); reasons: list[str] = []
        lo, hi = np.asarray(self.cfg.limits.lower_rad, np.float64), np.asarray(self.cfg.limits.upper_rad, np.float64)
        tol = np.radians(self.cfg.projection_tol_deg)
        near = ((q < lo) & (q >= lo - tol)) | ((q > hi) & (q <= hi + tol)); projected = bool(near.any())
        if projected: q = np.where(near, np.clip(q, lo, hi), q)
        if not (ik_pos_err_m <= self.cfg.max_ik_pos_err_m and ik_rot_err_deg <= self.cfg.max_ik_rot_err_deg):
            reasons.append(f"ik_unreachable:{ik_pos_err_m*1e3:.1f}mm/{ik_rot_err_deg:.1f}deg")
        reasons += self.cfg.limits.check(q_prev, q)
        step = float(np.max(np.abs(q - q_prev)))
        return FeasibilityVerdict(not reasons, reasons, q, float(ik_pos_err_m), float(ik_rot_err_deg), step, projected=projected)
