"""The tag-rig deploy UI (laundry_deploy.ui over laundry_deploy.session / .core) against a golden record.

tests/laundry/ui/tagrig_ui_probe.py builds the Flask app through laundry_deploy.ui.server.main() over a stubbed UiApp (no cameras,
no robot, no network, empty HOME), calls every route except the endless MJPEG streams, round-trips the persisted settings, reads
the deploy CLI defaults and evaluates the pure helpers (streamer timeline, temporal ensemble, chunk decode, RTC weights, ...) on
synthetic input. tests/laundry/ui/golden/tagrig_ui_routes.json was first recorded on the single-file modules (commit 7e501df) and
re-recorded after the OA-style rewrite with every value unchanged (only the module public names and the two module-folder paths
here_ui / here_dep differ); the test asserts the same output now: url map, the served page (sha256), JSON of every route, settings
file location / keys / defaults, CLI defaults and helper values (floats to 1e-9 relative). The recorded public names must exist.

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

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tests/laundry/ui/tagrig_ui_probe.py"
GOLDEN = REPO / "tests/laundry/ui/golden/tagrig_ui_routes.json"
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


def test_state_dir_override(tmp_path):
    """TAGRIG_STATE_DIR (laundry_common.paths) moves every working-state file of the deploy tools, nothing else"""
    if not _usable(PY):
        pytest.skip(f"{PY}: needs flask, cv2, numpy, scipy (set UI_TEST_PYTHON)")
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1])\n"
        "from laundry_deploy.ui import settings as U\n"
        "from laundry_deploy.session import paths as D\n"
        "from laundry_deploy.core import height_calib as HC\n"
        "print(json.dumps(dict(settings=str(U.SETTINGS_FILE), cams=str(D.CAMS_FILE), start=str(D.START_FILE), hc=str(HC.CALIB_FILE),\n"
        "                      pending=str(HC.PENDING_FILE), here_ui=U.__file__, here_dep=D.__file__, out=str(D.RUN_OUT_DIR))))\n")
    env = dict(os.environ, TAGRIG_STATE_DIR=str(tmp_path))
    r = subprocess.run([PY, "-c", code, str(REPO / "src")], capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    names = dict(settings="act_tagrig_ui_settings.json", cams="act_tagrig_cams.json", start="start_pose.json",
                 hc="height_calib.json", pending="height_calib_pending.json")
    for k, n in names.items():
        assert got[k] == str(tmp_path / n), (k, got[k])
    ed = REPO / "src/laundry_deploy"
    assert got["here_ui"] == str(ed / "ui/settings.py") and got["here_dep"] == str(ed / "session/paths.py")   # modules stay in the package
    assert not got["out"].startswith(str(tmp_path))                 # run outputs are not state


def test_height_calib_path(tmp_path):
    """--height-calib (11-01): startCmd shifts the start TCP by -xy_B_minus_robot per arm; runSession adds dzAt(row xy), ramped
    over the first 16 rows, to every committed row's z and leaves xy alone (tests/laundry/ui/height_calib_probe.py, robot-free, dry)"""
    probe = REPO / "tests/laundry/ui/height_calib_probe.py"
    chk = "import flask, cv2, numpy, scipy, pinocchio, pink, ego_teleop; from laundry_common import paths as p; assert (p.HANDUMI_SW_ROOT / 'assets/rebot_b601/rebot_b601_bimanual.urdf').exists()"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src")] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    if subprocess.run([PY, "-c", chk], capture_output=True, timeout=300, env=env).returncode != 0:
        pytest.skip(f"{PY}: needs flask, cv2, scipy, pinocchio, pink, ego_teleop and the reBot URDF")
    out = tmp_path / "hc.json"
    r = subprocess.run([PY, str(probe), str(REPO), str(out)], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    R = json.loads(out.read_text())
    assert all(f.startswith(R["state_files"][0].rsplit("/", 1)[0]) for f in R["state_files"])    # both in TAGRIG_STATE_DIR
    xy = {"left": [0.006, -0.004], "right": [-0.003, 0.008]}
    for s, v in xy.items():
        assert R["start_shift"][s] == pytest.approx([-v[0], -v[1]], abs=1e-12)
    assert R["run_hc_stop"] == R["run_nohc_stop"] == "test end"
    assert R["run_dz_err"] < 1e-12 and R["run_xy_equal"]
    assert abs(R["run_dz_mm_row8"]["left"]) > 1 and abs(R["run_dz_mm_row8"]["right"]) > 1        # the correction is really applied
    assert any(f.startswith("INFO:z_offset_") for f in R["run_hc_flags"][0]) and not any(f.startswith("INFO:z_offset_") for f in R["run_nohc_flags"][0])
