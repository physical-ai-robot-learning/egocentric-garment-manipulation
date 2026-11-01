"""Deadman switch -- STUB for P0 (2026-10-13). Live visual teleop must not start without it.

Plan (VISUAL_TELEOP_MVP_GAP.md 1b / 3.1), TODO P1:
  * `Deadman.pressed(now_ns) -> bool`; implementations HttpDeadman (UI button heartbeat; released when no heartbeat for
    150 ms), keyboard, ScriptedDeadman (replay/tests), AlwaysPressed (preview only; refused when live).
  * Coordinator: released => HOLD (no POST) and ArmSafetyPipeline.hold() (limiter stays at the last EXECUTED command);
    re-press => re-anchor like a clutch release (zero jump by construction, arm_relative_se3.release).
  * Live gating: OutboundGuard(live=True) requires a non-AlwaysPressed deadman."""
from __future__ import annotations
from typing import Protocol


class Deadman(Protocol):
    def pressed(self, now_ns: int) -> bool: ...


class ScriptedDeadman:
    """Released inside any [t0, t1) ns window."""
    def __init__(self, released_windows_ns=()) -> None: self.windows = [tuple(map(int, w)) for w in released_windows_ns]
    def pressed(self, now_ns: int) -> bool: return not any(a <= int(now_ns) < b for a, b in self.windows)


class AlwaysPressed:
    """Offline replay / preview only."""
    live_allowed = False
    def pressed(self, now_ns: int) -> bool: return True


class HttpDeadman:  # pragma: no cover - P1
    def __init__(self, *a, **k) -> None: raise NotImplementedError("P1: UI heartbeat deadman (150 ms timeout) not implemented")
