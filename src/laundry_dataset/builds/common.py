"""Shared plumbing of the build scripts: launching the dataset builder (tag_rig_unsplit) through the IK interpreter, reading the
episode / frame counts of the LeRobot roots it wrote, rsync uploads and the timestamped log (dataset stage)."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from laundry_common import paths as egopaths

BUILDER_MODULE = "laundry_dataset.tag_rig_unsplit"
LEROBOT_DIR = "data/lerobot"         # builder output, relative to the training root (its working directory)
RSYNC = "/opt/homebrew/bin/rsync"    # brew rsync 3.x (macOS openrsync lacks the options used here)
TAIL_CHARS = 300                     # builder / rsync output kept in a log line
UPLOAD_TAIL_CHARS = 200
BUILD_FAIL_TAIL_CHARS = 400            # builder output of a failed per-take build in its log line
TAKE_DATE_LEN = 9                    # "YYYYMMDD_" in front of a take name; set names keep only the HHMMSS part


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def builderEnv(**extra) -> dict:
    """environment of the builder sub-process: src/ on PYTHONPATH + extra variables (e.g. FILL_MASK_S)"""
    tPath = os.pathsep.join(tP for tP in (str(egopaths.SRC_ROOT), os.environ.get("PYTHONPATH", "")) if tP)
    return dict(os.environ, PYTHONPATH=tPath, **extra)


def runBuilder(args: list, logFile: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """tag_rig_unsplit `args` with the IK interpreter in the training root; stdout + stderr -> logFile"""
    tCmd = [str(egopaths.IK_PYTHON), "-m", BUILDER_MODULE, *args]
    tR = subprocess.run(tCmd, cwd=egopaths.TRAIN_ROOT, capture_output=True, text=True,
                        env=env if env is not None else builderEnv())
    logFile.write_text(tR.stdout + tR.stderr)
    return tR


def lerobotRoot(name: str, split: str) -> Path:
    return egopaths.TRAIN_ROOT / LEROBOT_DIR / f"{name}_{split}"


def builtCounts(root: Path):
    """(total_episodes, total_frames) of a written LeRobot root, None when it has no meta/info.json"""
    tInfo = root / "meta/info.json"
    if not tInfo.exists():
        return None
    tMeta = json.loads(tInfo.read_text())
    return tMeta["total_episodes"], tMeta["total_frames"]


def readKeys(path: Path) -> list:
    """whitespace-separated take/demoNN keys of a key-list file"""
    return Path(path).read_text().split()
