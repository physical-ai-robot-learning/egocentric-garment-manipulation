"""Hardware side of the MIT bridge: USB port lookup, gain / Ki overrides from the environment, building one arm on the
colleague's reBot plugin (MIT mode, integral feed-forward law, per-arm lock, jaw close endpoint, collision-guard latch
replaced by a callback), reading an arm, and opening the cameras. Needs the colleague's env (bh_indy7_lerobot,
lerobot_plugin_rebot) and `usb_arm_ports` from the robot-cockpit checkout on PYTHONPATH. Deploy stage: robot bridge.
"""

from __future__ import annotations
import math
import os

import numpy as np

from . import config
from .arm_state import ArmState

N_GAIN_VALUES = 6
KP_MAX = 200.0  # sanity bound of MIT_KP values
KD_MAX = 5.0  # sanity bound of MIT_KD values
KI_MAX = 200.0  # N*m/(rad*s), sanity bound of MIT_KI / MIT_KI_JOINTS


class BridgeConfigError(RuntimeError):
    """The launch environment of the bridge (ports, gains) is missing or out of range."""


def armPorts() -> dict:
    """{side: serial port} for every side in MIT_SIDES from LEFT_LOC / RIGHT_LOC USB locationIDs (never guessed)."""
    if config.MOCK:
        return {"left": "mock-left", "right": "mock-right"}
    from usb_arm_ports import arm_ports

    tMap = arm_ports()
    tLocs = {"left": os.environ.get("LEFT_LOC"), "right": os.environ.get("RIGHT_LOC")}
    if not all(tLocs[tSide] for tSide in config.SIDES):
        raise BridgeConfigError(
            "set LEFT_LOC / RIGHT_LOC for every side in MIT_SIDES (USB locationIDs, see usb_arm_ports.py) "
            "-- ports are never guessed here"
        )
    tPorts = {tSide: tMap.get(int(tLocs[tSide])) for tSide in config.SIDES}
    if not all(tPorts.values()):
        raise BridgeConfigError(f"arm ports not found for {tLocs} (sides {config.SIDES}): {tMap}")
    return tPorts


def gainOverrides() -> dict:
    """10-20 replay parity (j2/j3 steady error 0.5-1.1 deg at kp 64): MIT_KP / MIT_KD = 6 comma-separated values (joints 1..6),
    unset = the colleague's defaults (64,64,64,28,28,28) / (3,3,3,2,2,2); MIT_KI = integral law Ki (default 28). Read at
    /connect, so a change needs a bridge restart."""
    tOut = {}
    for tEnv, tKey, tHi in (("MIT_KP", "arm_mit_kp", KP_MAX), ("MIT_KD", "arm_mit_kd", KD_MAX)):
        tValue = os.environ.get(tEnv)
        if tValue:
            tVals = tuple(float(x) for x in tValue.split(","))
            if len(tVals) != N_GAIN_VALUES or min(tVals) < 0 or max(tVals) > tHi:
                raise BridgeConfigError(f"{tEnv} must be 6 values in [0, {tHi}], got {tValue}")
            tOut[tKey] = tVals
    if os.environ.get("MIT_KI"):
        tKi = float(os.environ["MIT_KI"])
        if not 0 < tKi <= KI_MAX:
            raise BridgeConfigError(f"MIT_KI must be in (0, 200], got {tKi}")
        tOut["arm_integral_ki_nm_per_rad_s"] = tKi
    return tOut


def kiJoints() -> tuple | None:
    """10-20 replay parity: MIT_KI_JOINTS = 6 per-joint Ki values (N*m/(rad*s)); the colleague's IntegralTorqueTerm takes ONE Ki
    per arm, so the bridge installs a subclass that scales each joint's step by Ki_j / Ki. None = stock."""
    tValue = os.environ.get("MIT_KI_JOINTS")
    if not tValue:
        return None
    tVals = tuple(float(x) for x in tValue.split(","))
    if len(tVals) != N_GAIN_VALUES or min(tVals) <= 0 or max(tVals) > KI_MAX:
        raise BridgeConfigError(f"MIT_KI_JOINTS must be 6 values in (0, 200], got {tValue}")
    return tVals


def perJointIntegralTerm(kiJointValues, kiBase: float):
    from bh_indy7_lerobot.feedforward import integral as integ

    class PerJointIntegralTerm(integ.IntegralTorqueTerm):
        def _advanced(self, joint, held_nm, measured_rad, context, bound_nm):
            tScale = kiJointValues[joint] / kiBase if joint < len(kiJointValues) else 1.0
            tErrorRad = float(context.q_command_rad[joint]) - float(measured_rad[joint])
            tAdvanced = held_nm + tScale * float(context.integral_ki_nm_per_rad_s) * tErrorRad * float(context.integration_step_s)
            tHandover = context.handover_nm
            if joint < len(tHandover):
                tAdvanced += float(tHandover[joint])
            return integ._clamped(tAdvanced, abs(float(bound_nm[joint])) if joint < len(bound_nm) else math.inf)

    return PerJointIntegralTerm()


def armMode(fol, cfg) -> str:
    try:
        return fol.arm_control_mode
    except Exception:  # the mock backend has no mode register
        return f"{cfg.arm_control_mode} (mock)"


def readArm(arm: ArmState) -> tuple:
    """-> (raw joint deg (6,), jaw norm) of one arm."""
    tFol = arm.mFol
    tFol.refresh_arm_state()
    tObs = tFol.read_observation()
    tQ = np.array([float(tObs[f"joint_{i}.pos"]) for i in range(config.ARM_DOF)])
    return tQ, float(tObs.get("gripper.pos", 0.0))


def buildArm(side: str, port: str, onGuardTrip, gripDegS: float) -> ArmState:
    """lerobot_plugin_rebot.ReBot(MIT, INTEGRAL law) -> connect -> integral law armed exactly as bh_indy7_lerobot_gui/infer_cli.py
    _arm_rebot_feedforward does (gravity model disarmed) -> arm torque on. onGuardTrip(side, reason) replaces the guard latch
    unless MIT_COLLISION_GUARD=1."""
    from bh_indy7_lerobot.feedforward.law import FeedforwardLaw, build_law
    from bh_indy7_lerobot_gui.workers.rebot_gui_adapter import ReBotGuiFollower
    from lerobot_plugin_rebot import ReBot
    from lerobot_plugin_rebot.config import ReBotConfig

    tCfg = ReBotConfig(
        id=f"bh_bimanual_{side}",
        port=port,
        can_adapter="damiao_serial",
        cameras={},
        arm_control_mode="mit",
        arm_feedforward_law=FeedforwardLaw.INTEGRAL,
        gripper_velocity_deg_s=gripDegS,
        control_tick_hz=config.TICK_HZ,  # 10-01: the plugin's integral / vel-FF laws step per configured tick
        **gainOverrides(),
    )
    tBackend = None
    if config.MOCK:
        from lerobot_plugin_rebot.arm_backend.mock_backend import MockReBotArmBackend

        tBackend = MockReBotArmBackend()
    tRebot = ReBot(tCfg, arm_backend=tBackend)
    if not config.MOCK:
        # 10-01: the colleague's transport takes ONE host-wide flock (one reBot per computer), so the second arm was refused.
        # Its _lockfile_path() is a per-instance seam: one lock per arm.
        tRebot.arm_backend._transport._lockfile_path = lambda _p=f"/tmp/bh_indy7_rebot_{side}.lock": _p
    tRebot.connect()
    # 11-02 user "왼쪽 약간 헐거워져서 ... 0보다 조금 더 가게": per-side jaw close endpoint past 0 deg (MIT_GRIP_CLOSE_DEG_<SIDE>)
    tCloseDeg = os.environ.get(f"MIT_GRIP_CLOSE_DEG_{side.upper()}")
    if tCloseDeg and not config.MOCK:
        tRebot.arm_backend.set_gripper_endpoint("close", math.radians(float(tCloseDeg)))
        print(f"[mit] {side} jaw close endpoint -> {float(tCloseDeg):+.1f} deg", flush=True)
    if os.environ.get("MIT_COLLISION_GUARD", "0") != "1":
        # 10-04 user "충돌 정지 이런것도 없애줘 오히려 연결이 안되": the plugin's guard keeps running but can no longer LATCH; a
        # would-be trip is only recorded (/status auto_events). MIT_COLLISION_GUARD=1 restores the latch.
        tGuard = tRebot._guard
        tGuard._latch = lambda reason, _side=side: onGuardTrip(_side, reason)
        tGuard.reset()
        print(f"[mit] {side}: collision-guard latch DISABLED (MIT_COLLISION_GUARD!=1)", flush=True)
    tFol = ReBotGuiFollower(tRebot)
    # == infer_cli._arm_rebot_feedforward, integral branch: BEFORE the enable, so the arm is never energised at t_ff = 0 with the
    # wrong law installed. The model is disarmed explicitly (the integral law uses no calibration).
    tLaw = build_law(tCfg.arm_feedforward_law, friction_supplier=None)
    if tLaw.arms_gravity:
        raise BridgeConfigError("integral law expected")
    try:
        tFol.set_gravity_feedforward(None, False)
        tKij = kiJoints()
        tTerms = (perJointIntegralTerm(tKij, float(tCfg.arm_integral_ki_nm_per_rad_s)),) if tKij else tLaw.extra_terms()
        tFol.set_feedforward_terms(tTerms)
        tFol.set_gain_schedule(None, float(tCfg.arm_integral_ki_nm_per_rad_s))
        tArmed = f"law={tCfg.arm_feedforward_law} terms={tLaw.term_names()} ki={tCfg.arm_integral_ki_nm_per_rad_s}" + (
            f" ki_joints={tKij}" if tKij else ""
        )
    except Exception as tErr:  # the mock backend has no torque path
        if not config.MOCK:
            raise
        tArmed = f"mock (no torque path: {tErr.__class__.__name__})"
    tFol.set_arm_torque(True)
    tMode = armMode(tFol, tCfg)
    print(f"[mit] {side} arm on {port}: mode={tMode} {tArmed} kp={tCfg.arm_mit_kp} kd={tCfg.arm_mit_kd}", flush=True)
    return ArmState(tRebot, tFol, tCfg)


def openCameras() -> dict:
    if config.MOCK or config.NO_CAMERAS:
        return {}
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    tCams = {}
    for tName, tIdx in config.CAMERAS.items():
        tCam = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=tIdx,
                width=config.CAM_WIDTH_PX,
                height=config.CAM_HEIGHT_PX,
                fps=config.CAM_FPS,
                warmup_s=config.CAM_WARMUP_S,
            )
        )
        tCam.connect()
        tCams[tName] = tCam
    return tCams
