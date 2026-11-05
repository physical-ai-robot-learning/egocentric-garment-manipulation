"""Test fixtures of laundry_training: a golden-value recorder and a stand-in for the parts of lerobot / accelerate the package
touches (lerobot is not installed in the test environment).

`golden`  compares every value a test hands it against tests/golden_training.npz, exactly (array_equal, NaN == NaN). With
          LAUNDRY_CHAR_RECORD=1 the values are recorded instead (the characterization run against the pre-refactor code).
`stubbed` installs fake `lerobot.*` / `accelerate` modules for one test module, imports the package under them, and removes
          every module it added afterwards, so the import smoke test of tests/laundry still sees the real environment.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
try:
    import torch
except ImportError:                  # base install without the vision / train group: these tests are not collected
    torch = None

collect_ignore_glob = ["test_*.py"] if torch is None else []

GOLDEN_FILE = Path(__file__).with_name("golden_training.npz")
RECORD = os.environ.get("LAUNDRY_CHAR_RECORD") == "1"


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for tKey in sorted(value, key=str):
            _flatten(f"{prefix}/{tKey}", value[tKey], out)
    elif isinstance(value, (list, tuple)) and not (value and all(isinstance(tX, (int, float, bool, np.number)) for tX in value)):
        out[f"{prefix}#len"] = np.asarray(len(value))
        for tIndex, tItem in enumerate(value):
            _flatten(f"{prefix}[{tIndex}]", tItem, out)
    elif torch.is_tensor(value):
        out[prefix] = value.detach().cpu().numpy()
    elif value is None:
        out[prefix] = np.asarray("<None>")
    elif isinstance(value, (str, Path)):
        out[prefix] = np.asarray(str(value))
    else:
        out[prefix] = np.asarray(value)


class Golden:
    def __init__(self):
        self.mStore = dict(np.load(GOLDEN_FILE, allow_pickle=False)) if GOLDEN_FILE.exists() else {}
        self.mNew = {}

    def check(self, name, value):
        tFlat = {}
        _flatten(name, value, tFlat)
        if RECORD:
            self.mNew.update(tFlat)
            return
        tExpectedKeys = sorted(tK for tK in self.mStore if tK == name or tK.startswith(name + "/") or tK.startswith(name + "[")
                               or tK.startswith(name + "#"))
        assert sorted(tFlat) == tExpectedKeys, f"{name}: keys differ: {sorted(set(tFlat) ^ set(tExpectedKeys))[:10]}"
        for tKey, tValue in tFlat.items():
            tExp = self.mStore[tKey]
            assert tValue.shape == tExp.shape, f"{tKey}: shape {tValue.shape} != {tExp.shape}"
            if tExp.dtype.kind in "fc" or tValue.dtype.kind in "fc":
                assert np.array_equal(tValue.astype(np.float64), tExp.astype(np.float64), equal_nan=True), \
                    f"{tKey}: max diff {np.nanmax(np.abs(tValue.astype(np.float64) - tExp.astype(np.float64)))}"
            else:
                assert np.array_equal(tValue, tExp), f"{tKey}: {tValue!r} != {tExp!r}"


_GOLDEN = Golden()


@pytest.fixture(scope="session")
def golden():
    yield _GOLDEN
    if RECORD and _GOLDEN.mNew:
        tAll = dict(_GOLDEN.mStore)
        tAll.update(_GOLDEN.mNew)
        np.savez_compressed(GOLDEN_FILE, **tAll)


# ---------------------------------------------------------------- stand-ins for lerobot / accelerate
class FakeHf:
    """`ds.hf_dataset.with_format("numpy")`: column access by name, row slices -> dict of columns, len()"""

    def __init__(self, cols):
        self.mCols = cols

    def with_format(self, _fmt):
        return self

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {tK: tV[key] for tK, tV in self.mCols.items()}
        return self.mCols[key]

    def __len__(self):
        return len(next(iter(self.mCols.values())))


REGISTRY = {}


def registerDataset(root, cols, features=None, items=None, stats=None, info=None, cameras=(), fps=20):
    REGISTRY[str(root)] = dict(cols=cols, features=features if features is not None else {tK: {} for tK in cols}, items=items,
                               stats=stats or {}, info=info or {"features": {}}, cameras=list(cameras), fps=fps)


def _makeLerobot():
    tMods = {}

    def mod(name, **attrs):
        tM = types.ModuleType(name)
        tM.__dict__.update(attrs)
        tMods[name] = tM
        return tM

    class LeRobotDatasetMetadata:
        def __init__(self, repo_id, root=None, **kw):
            tD = REGISTRY[str(root)]
            self.repo_id, self.root = repo_id, root
            self.fps = tD["fps"]
            self.camera_keys = list(tD["cameras"])
            self.stats = tD["stats"]
            self.info = tD["info"]

    class LeRobotDataset:
        CREATED = []

        def __init__(self, repo_id=None, root=None, **kw):
            tD = REGISTRY[str(root)]
            self.repo_id, self.root, self.kw = repo_id, root, kw
            self.hf_dataset = FakeHf(tD["cols"])
            self.features = tD["features"]
            self.meta = LeRobotDatasetMetadata(repo_id, root=root)
            self.meta.stats = {tK: dict(tV) for tK, tV in tD["stats"].items()}
            self._items = tD["items"]
            self.num_episodes = len(np.unique(tD["cols"]["episode_index"])) if "episode_index" in tD["cols"] else 0

        def __getitem__(self, idx):
            return {tK: (tV.clone() if torch.is_tensor(tV) else tV) for tK, tV in self._items[idx].items()}

        def __len__(self):
            return len(self._items) if self._items is not None else len(self.hf_dataset)

    class Accelerator:
        def clip_grad_norm_(self, parameters, max_norm, *a, **kw):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    class DiffuserSchedulerConfig:
        def build(self, optimizer, num_training_steps):
            return ("built", num_training_steps)

    class MultiTaskDiTConfig:
        def get_optimizer_preset(self):
            return SimpleNamespace(grad_clip_norm=10.0, lr=2e-5)

    class ImageTransforms:
        def __init__(self, cfg):
            self.mCfg = cfg

    tLt = mod("lerobot.scripts.lerobot_train", update_policy=None, make_dataset=None, EpisodeAwareSampler=None, main=None)
    mod("lerobot")
    mod("lerobot.datasets")
    mod("lerobot.datasets.lerobot_dataset", LeRobotDataset=LeRobotDataset)
    mod("lerobot.datasets.dataset_metadata", LeRobotDatasetMetadata=LeRobotDatasetMetadata)
    mod("lerobot.scripts")
    mod("lerobot.policies")
    mod("lerobot.policies.multi_task_dit")
    mod("lerobot.policies.multi_task_dit.modeling_multi_task_dit", MultiTaskDiTPolicy=None, F=torch.nn.functional)
    mod("lerobot.policies.multi_task_dit.configuration_multi_task_dit", MultiTaskDiTConfig=MultiTaskDiTConfig)
    mod("lerobot.optim")
    mod("lerobot.optim.schedulers", DiffuserSchedulerConfig=DiffuserSchedulerConfig)
    mod("lerobot.transforms", ImageTransforms=ImageTransforms)
    mod("lerobot.utils")
    mod("lerobot.utils.constants", IMAGENET_STATS={"mean": [[[0.485]], [[0.456]], [[0.406]]], "std": [[[0.229]], [[0.224]],
                                                                                                      [[0.225]]]})
    mod("accelerate", Accelerator=Accelerator)
    return tMods, tLt


@pytest.fixture(scope="module")
def stubbed():
    """-> SimpleNamespace(mods=fake modules by name, lt=fake lerobot_train, imp=import function for laundry_training modules)"""
    tBefore = dict(sys.modules)
    for tName in [tN for tN in sys.modules if tN == "laundry_training" or tN.startswith("laundry_training.")]:
        del sys.modules[tName]
    tMods, tLt = _makeLerobot()
    tReal = {}
    for tName, tMod in tMods.items():
        if tName in sys.modules:
            tReal[tName] = sys.modules[tName]
        sys.modules[tName] = tMod
    try:
        yield SimpleNamespace(mods=tMods, lt=tLt, imp=importlib.import_module)
    finally:
        for tName in list(sys.modules):
            if tName not in tBefore:
                del sys.modules[tName]
        sys.modules.update(tBefore)
