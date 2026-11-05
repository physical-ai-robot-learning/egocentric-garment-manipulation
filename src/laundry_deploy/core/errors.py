"""Errors of the deploy package. Every message is shown to the operator verbatim (Korean where the tools speak Korean): the CLI
prints it and exits with status 1, the UI shows it in its status line, a running session ends with it as the stop reason --
exactly where these refusals used to be SystemExit."""

from __future__ import annotations


class DeployError(RuntimeError):
    """A deploy step refuses to start or cannot continue; the message says why."""


class CheckpointError(DeployError):
    """A checkpoint or the policy server does not match the deploy contract."""


class HardwareError(DeployError):
    """A camera, the robot bridge or the robot state is missing, stale or in the wrong state."""


class RunRefused(DeployError):
    """The run / move arguments or the robot pose make the requested motion unsafe or invalid."""
