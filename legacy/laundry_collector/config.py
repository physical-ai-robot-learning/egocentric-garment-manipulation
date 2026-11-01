"""Configuration for the bare-hand laundry collector (v1: 2 wrist RGB + 1 fixed global RGB-D, no IMU, no gripper).

Files (configs/laundry/):
  hardware_<profile>.yaml   cameras (identity, mode)            profiles: laundry_v1 (Orbbec + 2 Arducam fisheye), mock
  collector.yaml            dataset root, protocol timings, UI   (one file for all profiles)
  world_config.yaml         table/world frame + X_L / X_R init markers, copied into every episode's calibration/
Unknown keys raise: a typo in a yaml must never silently fall back to a default.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs" / "laundry"
CAL_DIR = REPO / "configs" / "calibration"
STREAMS = ("global", "left_wrist", "right_wrist")


@dataclass
class CameraCfg:
    name: str                          # global | left_wrist | right_wrist
    backend: str = "uvc"               # orbbec | uvc | mock
    match_name: str | None = None      # AVFoundation product-name substring (uvc); wrists have distinct names
    ordinal: int = 0
    index: int | None = None           # explicit OpenCV index (wins over match_name)
    serial: str | None = None          # expected USB / Orbbec serial (provenance; verified when strict_identity)
    strict_identity: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    depth: bool = False                # global only: record aligned depth
    depth_width: int | None = None     # orbbec depth mode (None = color size); aligned to color either way
    depth_height: int | None = None
    codec: str = "h264_videotoolbox"
    bitrate_kbps: int = 12000
    intrinsics: str | None = None      # calibration prefix in configs/calibration (latest version wins), e.g. fisheye_left
    required: bool = True


@dataclass
class HardwareCfg:
    profile: str
    hardware_version: str
    cameras: list[CameraCfg]

    def camera(self, name: str) -> CameraCfg:
        return next(c for c in self.cameras if c.name == name)


@dataclass
class ProtocolCfg:
    init_hold_s: float = 3.0           # Phase 1: both hands still on X_L / X_R
    end_hold_s: float = 2.0            # Phase 3: hands released and still
    min_manip_s: float = 3.0           # STOP earlier than this is refused (accidental double press)


@dataclass
class CollectorCfg:
    dataset_root: str = "~/datasets/laundry_raw"
    dataset: str = "laundry_pilot"
    task: str = "tshirt_folding"
    operator: str = ""
    preview_hz: int = 15
    buffer_s: float = 4.0              # per-camera ring buffer (pre-roll is NOT recorded; buffers are drained at START)
    depth_writers: int = 3             # PNG writer threads for global depth
    qa_sample_every: int = 15          # sharpness / exposure / depth-valid sampling stride (frames)
    initial_states: list[str] = field(default_factory=lambda: ["flat", "semi_flat", "crumpled", "rotated"])
    garments: list[str] = field(default_factory=lambda: ["shirt_01"])
    protocol: ProtocolCfg = field(default_factory=ProtocolCfg)


@dataclass
class AppConfig:
    hardware: HardwareCfg
    collector: CollectorCfg
    world: dict
    mock: bool = False

    @property
    def dataset_dir(self) -> Path:
        root = Path(self.collector.dataset_root).expanduser()
        return root / (self.collector.dataset + ("_mock" if self.mock else ""))


def _build(cls, d: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    bad = set(d) - names
    if bad:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(bad)}")
    kw = dict(d)
    if cls is CollectorCfg and "protocol" in kw:
        kw["protocol"] = _build(ProtocolCfg, kw["protocol"])
    return cls(**kw)


def load_config(profile: str = "laundry_v1", *, mock: bool = False, config_dir: Path | None = None) -> AppConfig:
    cd = Path(config_dir or CONFIG_DIR)
    hw = yaml.safe_load((cd / f"hardware_{'mock' if mock else profile}.yaml").read_text())
    cams = [_build(CameraCfg, c) for c in hw.pop("cameras")]
    if sorted(c.name for c in cams) != sorted(STREAMS):
        raise ValueError(f"hardware profile must define exactly {STREAMS}, got {[c.name for c in cams]}")
    if not cams[[c.name for c in cams].index("global")].depth:
        raise ValueError("the global camera must record depth (depth: true)")
    hardware = _build(HardwareCfg, dict(hw, cameras=cams))
    coll = _build(CollectorCfg, yaml.safe_load((cd / "collector.yaml").read_text()) or {})
    world = yaml.safe_load((cd / "world_config.yaml").read_text())
    return AppConfig(hardware=hardware, collector=coll, world=world, mock=mock)
