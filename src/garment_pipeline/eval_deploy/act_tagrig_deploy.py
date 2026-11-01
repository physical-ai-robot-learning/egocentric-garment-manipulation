"""[2026-10-23] Closed-loop real-robot deploy of the tag-rig laundry policies on the bimanual reBot B601-DM -- entry point.

Since 2026-11-01 the code is split by area into tagrig_deploy/ (map in its __init__.py); the full description of the loop is
the docstring of tagrig_deploy/cli.py, printed by --help. Commands and imports are unchanged:
    python -m garment_pipeline.eval_deploy.act_tagrig_deploy {cams,wave,start,run} ...
    from garment_pipeline.eval_deploy.act_tagrig_deploy import Rig, run_session, CAMS_CFG, ...   (every name of the old file)
"""
from __future__ import annotations

from egogarment import paths as egopaths  # noqa: F401
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: F401
from garment_pipeline.eval_deploy.tagrig_deploy import *  # noqa: F401,F403
from garment_pipeline.eval_deploy.tagrig_deploy import __all__, main  # noqa: F401

if __name__ == "__main__":
    main()
