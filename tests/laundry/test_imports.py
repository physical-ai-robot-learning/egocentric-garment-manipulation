"""Import smoke test: every module under src/ either imports, or is skipped because an optional heavy dependency
(GPU / camera / robot / GUI stack) is not installed in this environment. Anything else is a failure."""
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Optional dependencies (top-level import names). A module that needs one of these and cannot find it is skipped.
OPTIONAL = {
    "torch", "torchvision", "lerobot", "transformers", "diffusers", "ultralytics", "onnxruntime", "timm",
    "pinocchio", "pink", "qpsolvers",                     # IK
    "cv2", "pupil_apriltags", "mediapipe", "pyorbbecsdk", "pyrealsense2",   # cameras / vision
    "PySide6", "PyQt5", "pyqtgraph", "flask", "requests",  # UI / HTTP
    "gtsam", "serial", "hid", "scservo_sdk", "feetech_servo_sdk",           # factor graphs / hardware buses
    "pandas", "pyarrow", "datasets", "huggingface_hub", "imageio", "av", "mcap", "tqdm", "PIL", "h5py",
    "cotracker", "rerun", "trimesh", "sklearn",
}

# Modules that do their work at import time. Every laundry_* module has a main(argv) since the 11-04 OA refactor; the
# remaining entries are laundry_shared copies, kept as they were.
RUN_AT_IMPORT = {
    "laundry_shared.ego_collector.__main__": "entry point: runs the ego-collector command line (argparse exits without arguments)",
    "laundry_shared.handumi_collector.tools.metric_qa": "script: reads the CSV given in sys.argv[1]",
}

# Modules whose dependencies are not part of this repository (listed in docs/provenance.md).
NOT_VENDORED = {
    "laundry_shared.visual_teleop.vision.stereo_vio_tracker": "needs stereo_openvins_eval / stereo_basalt_eval and the OpenVINS backend of the "
                                               "dropped stereo-VIO experiment (not in this repository, see docs/provenance.md)",
}


def _modules():
    out = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if "__pycache__" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts and all(SRC.joinpath(*parts[:i], "__init__.py").exists() for i in range(1, len(parts))):
            out.append(".".join(parts))
    return out


MODULES = _modules()


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    if module in RUN_AT_IMPORT:
        pytest.skip(f"not imported: {RUN_AT_IMPORT[module]}")
    if module in NOT_VENDORED:
        pytest.skip(f"not importable from this repository: {NOT_VENDORED[module]}")
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        top = (e.name or "").split(".")[0]
        if top in OPTIONAL:
            pytest.skip(f"optional dependency '{top}' not installed")
        raise
