"""Import link to the shared UMI contract in ~/ego_cart20 (relpose.py, geometry/rotation6d.py). That code is THE one
implementation (its docstrings say so) -- ego_teleop imports it, never copies it. ego_cart20 is not pip-installed in
ego_collector/.venv or ~/xvla-mac, so its repo root is put on sys.path here (EGO_CART20_ROOT overrides ~/ego_cart20).
ego_cart20 has no git commits yet (VISUAL_TELEOP_MVP_GAP.md risk 8): the contract version cannot be pinned."""
from __future__ import annotations
import os
import sys
from pathlib import Path

EGO_CART20_ROOT = Path(os.environ.get("EGO_CART20_ROOT", str(Path.home() / "ego_cart20")))


def ensure_ego_cart20() -> Path:
    try:
        import ego_cart20  # noqa: F401
        return Path(ego_cart20.__file__).resolve().parents[1]
    except ImportError:
        pass
    if not (EGO_CART20_ROOT / "ego_cart20" / "umi_fusion" / "relpose.py").exists():
        raise ImportError(f"ego_cart20 not found at {EGO_CART20_ROOT} (set EGO_CART20_ROOT)")
    sys.path.insert(0, str(EGO_CART20_ROOT))
    return EGO_CART20_ROOT


def relpose():
    ensure_ego_cart20()
    from ego_cart20.umi_fusion import relpose as m
    return m


def rotation6d():
    ensure_ego_cart20()
    from ego_cart20.geometry import rotation6d as m
    return m
