"""Episode directories (spec §7): <dataset>/episode_NNNNNN/{global,left_wrist,right_wrist,calibration}/ + metadata.json.

Lifecycle: create() writes `.incomplete` and the calibration snapshot BEFORE the first frame; finalize(keep=True) writes
metadata.json + sync_report.{json,txt} (tmp + replace), removes `.incomplete`, touches `.complete`. A cancelled / discarded
episode is MOVED to <dataset>/_discarded/ (never deleted). A directory with `.incomplete` is a crashed take.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import yaml

from .config import CAL_DIR, AppConfig

COLLECTOR_VERSION = "v1"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str)); tmp.replace(path)


def latest_calibration(prefix: str) -> tuple[Path | None, dict | None]:
    files = sorted(CAL_DIR.glob(f"{prefix}_v[0-9][0-9][0-9].yaml"))
    if not files:
        return None, None
    return files[-1], yaml.safe_load(files[-1].read_text())


class EpisodeStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.root = cfg.dataset_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> int:
        ids = [int(m.group(1)) for p in list(self.root.glob("episode_*")) + list((self.root / "_discarded").glob("episode_*"))
               if (m := re.match(r"episode_(\d+)", p.name))]
        return max(ids, default=0) + 1

    def create(self, devices: dict) -> tuple[int, Path]:
        eid = self.next_id()
        ep = self.root / f"episode_{eid:06d}"
        ep.mkdir()
        (ep / ".incomplete").touch()
        cal = ep / "calibration"; cal.mkdir()
        # global: intrinsics read from the device at open (Orbbec SDK) -- the camera itself is the source of truth
        g = devices.get("global", {})
        _atomic_json(cal / "global_intrinsics.json", dict(source="device", stream="color", depth_aligned_to="color",
                                                          depth_scale_mm_per_unit=g.get("depth_scale_mm_per_unit"),
                                                          **(g.get("intrinsics") or {})))
        for side in ("left", "right"):
            cc = self.cfg.hardware.camera(f"{side}_wrist")
            path, data = latest_calibration(cc.intrinsics) if cc.intrinsics else (None, None)
            _atomic_json(cal / f"{side}_intrinsics.json",
                         dict(data or {}, calibration_file=str(path) if path else None, calibrated=bool(data),
                              note=None if data else "no calibration file -- run fisheye calibration before preprocessing"))
        _atomic_json(cal / "world_config.json", self.cfg.world)
        _atomic_json(cal / "devices.json", devices)
        return eid, ep

    def finalize(self, ep: Path, *, keep: bool, metadata: dict, report: dict | None, report_txt: str | None,
                 recorder_summary: dict | None) -> Path:
        if recorder_summary is not None:
            _atomic_json(ep / "recorder_summary.json", recorder_summary)
        if report is not None:
            _atomic_json(ep / "sync_report.json", report)
            (ep / "sync_report.txt").write_text(report_txt or "")
        _atomic_json(ep / "metadata.json", dict(metadata, collector_version=COLLECTOR_VERSION, kept=keep))
        (ep / ".incomplete").unlink(missing_ok=True)
        if keep:
            (ep / ".complete").touch()
            return ep
        dst = self.root / "_discarded" / ep.name
        dst.parent.mkdir(exist_ok=True)
        shutil.move(str(ep), str(dst))
        return dst


def base_metadata(cfg: AppConfig, eid: int, form: dict, t: dict) -> dict:
    return dict(episode_id=eid, task=form.get("task") or cfg.collector.task, garment_id=form.get("garment_id"),
                initial_state=form.get("initial_state"), success=form.get("success"), notes=form.get("notes", ""),
                operator=form.get("operator") or cfg.collector.operator, dataset=cfg.collector.dataset,
                hardware_profile=cfg.hardware.profile, world_frame="table", left_init_marker="X_L", right_init_marker="X_R",
                recording_start_time=t.get("start_wall"), recording_end_time=t.get("end_wall"),
                recording_start_host_ns=t.get("start_ns"), recording_end_host_ns=t.get("end_ns"),
                phases=t.get("phases", []), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
