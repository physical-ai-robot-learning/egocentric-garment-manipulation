"""[2026-10-23] Inference web UI for the tag-rig laundry policies on the bimanual reBot (http://localhost:8770) -- entry point.

Since 2026-11-01 the code is split into tagrig_ui/ (settings, run control, Flask server, page.html; map in its __init__.py);
the full description is the docstring of tagrig_ui/server.py, printed by --help. Settings stay in act_tagrig_ui_settings.json
next to this file (same keys and defaults). Commands and imports are unchanged:
    python -m garment_pipeline.eval_deploy.act_tagrig_ui [--port 8770] [--bridge-url http://127.0.0.1:8031 (mock)] [--dry-only]
"""
from __future__ import annotations

from egogarment import paths as egopaths  # noqa: F401
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: F401
from garment_pipeline.eval_deploy import tagrig_deploy as dep  # noqa: F401
from garment_pipeline.eval_deploy.tagrig_ui.settings import (
    HERE, SETTINGS, XVLA_FT, DEFAULTS, CK_DIRS, HB_TIMEOUT_S, load_settings, save_settings, ckpts)  # noqa: F401
from garment_pipeline.eval_deploy.tagrig_ui.control import UICtl, App, move_ready, dep_parser  # noqa: F401
from garment_pipeline.eval_deploy.tagrig_ui.page import PAGE  # noqa: F401
from garment_pipeline.eval_deploy.tagrig_ui.server import create_app, main  # noqa: F401

if __name__ == "__main__":
    main()
