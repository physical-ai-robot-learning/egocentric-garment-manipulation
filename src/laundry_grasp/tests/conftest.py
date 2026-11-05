from __future__ import annotations

import importlib.util

import pytest

from laundry_common import paths as egopaths
from laundry_grasp import events
from laundry_grasp.tests import synthetic

# base install without the vision group: the modules that load the ResNet-18 recipe import torch at import time
collect_ignore = [] if importlib.util.find_spec("torch") else ["test_model_gold.py", "test_pgrasp.py", "test_sidefile.py"]


@pytest.fixture(scope="session")
def syntheticRoot(tmp_path_factory):
    tRoot = tmp_path_factory.mktemp("tag_rig")
    synthetic.writeTake(tRoot)
    synthetic.writeTrim(tRoot / "idle_trim.csv")
    return tRoot


@pytest.fixture
def synthTakes(syntheticRoot, monkeypatch):
    """egopaths.TAG_RIG / IDLE_TRIM_CSV pointed at the synthetic take; the per-process jaw cache cleared around the test"""
    monkeypatch.setattr(egopaths, "TAG_RIG", syntheticRoot)
    monkeypatch.setattr(egopaths, "IDLE_TRIM_CSV", syntheticRoot / "idle_trim.csv")
    events.jawV2.cache_clear()
    yield syntheticRoot
    events.jawV2.cache_clear()
