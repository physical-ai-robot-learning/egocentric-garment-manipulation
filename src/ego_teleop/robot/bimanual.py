"""Single-writer dual-arm command composition -- DESIGN for P1 (2026-10-13). Nothing here talks to a robot.

Why: /execute_step takes ONE 14-vector [L j1..j6 rad, L jaw cmd, R j1..j6 rad, R jaw cmd] (leader frame) and replaces BOTH
arms' goals. HttpRebotClient.build_action re-sends the OTHER arm at its (lagging) observation, so two single-side clients
would fight: each POST resets the other arm's goal backwards. The MIT bridge additionally drops goal-interpolation segments
whenever another writer replaces a goal (robot_service_mit.py:36-43).

Design:
  1. Exactly ONE BimanualCommandWriter per robot URL per process (enforced below) and, P1, per host: an exclusive flock on
     /tmp/ego_teleop_writer_<port>.lock taken in open() (mock-tested only; robot_service_mit already uses per-arm flocks
     for the BUS, which do not protect the HTTP goal).
  2. Each arm's pipeline (filter -> safety.propose -> IK -> FeasibilityGate) STAGES a candidate (q6 follower rad, g).
  3. compose() builds one vector per tick from the staged candidates; an arm without a fresh staged candidate is sent at
     its last EXECUTED command (never at its observation -> no backwards reset of a lagging arm).
  4. After the single POST, commit() promotes the staged candidates to "executed" and each arm's ArmSafetyPipeline.commit();
     on failure every arm hold()s (the whole vector was refused).
  5. Gripper: g (0 closed, 1 open) -> jaw cmd = 45 g (rebot_client.gripper_cmd_from_g, mapping A).
  6. The outbound guard checks per-arm joint step and jaw step against the last executed vector (OutboundGuard variant
     that permits the other arm + jaws within bounds).
TODO P1: send() (needs deadman + live OutboundGuard), the flock, and a dual-arm coordinator."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .rebot_client import GRIPPER_INDEX, N_ACTION, SIDE_SLICE, gripper_cmd_from_g, leader_action

_WRITERS: dict[str, "BimanualCommandWriter"] = {}


def build_bimanual_action(q_left_rad, q_right_rad, g_left: float, g_right: float) -> list[float]:
    """Both arms (follower rad) + both grippers (g: 0 closed, 1 open) -> ONE /execute_step leader-frame vector."""
    q = [0.0] * N_ACTION
    for side, qs, g in (("left", q_left_rad, g_left), ("right", q_right_rad, g_right)):
        qs = np.asarray(qs, np.float64).reshape(6)
        if not np.all(np.isfinite(qs)): raise ValueError(f"{side} joints not finite")
        q[SIDE_SLICE[side]] = [float(v) for v in qs]; q[GRIPPER_INDEX[side]] = gripper_cmd_from_g(g)
    return leader_action(q)


@dataclass
class _ArmSlot:
    q_exec: np.ndarray
    g_exec: float
    q_staged: np.ndarray | None = None
    g_staged: float | None = None


class BimanualCommandWriter:
    """The only object allowed to compose /execute_step vectors for a robot URL (see module docstring)."""

    def __init__(self, base_url: str, q_exec: dict, g_exec: dict) -> None:
        key = base_url.rstrip("/")
        if key in _WRITERS: raise RuntimeError(f"a BimanualCommandWriter for {key} already exists: two writers would fight over the 14-vector")
        self.base_url = key
        self.arms = {s: _ArmSlot(np.asarray(q_exec[s], np.float64).copy(), float(g_exec[s])) for s in SIDE_SLICE}
        _WRITERS[key] = self

    def close(self) -> None: _WRITERS.pop(self.base_url, None)

    def stage(self, side: str, q6, g: float | None = None) -> None:
        a = self.arms[side]; a.q_staged = np.asarray(q6, np.float64).copy(); a.g_staged = a.g_exec if g is None else float(g)

    def compose(self) -> list[float]:
        L, R = self.arms["left"], self.arms["right"]
        pick = lambda a: (a.q_staged if a.q_staged is not None else a.q_exec, a.g_staged if a.g_staged is not None else a.g_exec)
        (qL, gL), (qR, gR) = pick(L), pick(R)
        return build_bimanual_action(qL, qR, gL, gR)

    def commit(self) -> None:
        for a in self.arms.values():
            if a.q_staged is not None: a.q_exec = a.q_staged
            if a.g_staged is not None: a.g_exec = a.g_staged
            a.q_staged = a.g_staged = None

    def discard(self) -> None:
        for a in self.arms.values(): a.q_staged = a.g_staged = None

    def send(self, *a, **k):  # pragma: no cover - P1
        raise NotImplementedError("P1: live dual-arm sending needs the deadman + a live OutboundGuard; Stage 0 is offline only")
