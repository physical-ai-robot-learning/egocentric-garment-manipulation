"""The tag-rig deploy UI (garment_pipeline.eval_deploy.act_tagrig_ui, split into tagrig_ui/ / tagrig_deploy/ / tagrig_core/ on
2026-11-01) against a golden record.

tests/ui/tagrig_ui_probe.py builds the Flask app through act_tagrig_ui.main() over a stubbed App (no cameras, no robot, no
network, empty HOME), calls every route except the endless MJPEG streams, round-trips the persisted settings, reads the deploy
CLI defaults and evaluates the pure helpers (streamer timeline, temporal ensemble, chunk decode, RTC weights, ...) on synthetic
input. tests/ui/golden/tagrig_ui_routes.json is its output on the single-file modules before the split (commit 49e7542); the
test asserts the same output now: url map, the served page (sha256), JSON of every route, settings file location / keys /
defaults, CLI defaults and helper values (floats to 1e-9 relative). Names of the old modules must all still exist.

Needs flask, cv2, numpy, scipy in UI_TEST_PYTHON (default: this interpreter); skips otherwise, e.g.
    UI_TEST_PYTHON=~/xvla-mac/bin/python pytest tests/test_ui_routes.py -q
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests/ui/tagrig_ui_probe.py"
GOLDEN = REPO / "tests/ui/golden/tagrig_ui_routes.json"
PY = os.path.expanduser(os.environ.get("UI_TEST_PYTHON", sys.executable))


def _usable(py):
    try:
        return subprocess.run([py, "-c", "import flask, cv2, numpy, scipy"], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _same(a, b, path="$"):
    num = (int, float)
    if (isinstance(a, float) or isinstance(b, float)) and isinstance(a, num) and isinstance(b, num) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        assert (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {a} != {b}"
        return
    if isinstance(a, dict) and isinstance(b, dict):
        assert sorted(a) == sorted(b), f"{path}: keys differ {sorted(set(a) ^ set(b))}"
        for k in a: _same(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)): _same(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def probe():
    if not _usable(PY):
        pytest.skip(f"{PY}: needs flask, cv2, numpy, scipy (set UI_TEST_PYTHON)")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "probe.json"
        r = subprocess.run([PY, str(PROBE), str(REPO), str(out)], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        return json.loads(out.read_text())


def test_page_url_map_and_files(probe):
    g = json.loads(GOLDEN.read_text())
    assert (probe["page_sha256"], probe["page_len"]) == (g["page_sha256"], g["page_len"])
    assert probe["url_map"] == g["url_map"] and probe["app_name"] == g["app_name"]
    assert probe["files"] == g["files"]          # settings / camera map / start pose stay where they were


def test_routes_settings_cli_and_helpers(probe):
    g = json.loads(GOLDEN.read_text())
    assert sorted(probe) == sorted(g)
    for k in g:
        if k == "names":
            for mod, names in g[k].items():
                assert set(names) <= set(probe[k][mod]), f"{mod}: missing {sorted(set(names) - set(probe[k][mod]))}"
        else:
            _same(probe[k], g[k], k)
