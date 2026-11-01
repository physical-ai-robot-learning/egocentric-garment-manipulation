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


def test_state_dir_override(tmp_path):
    """TAGRIG_STATE_DIR (egogarment.paths) moves every working-state file of the deploy tools, nothing else"""
    if not _usable(PY):
        pytest.skip(f"{PY}: needs flask, cv2, numpy, scipy (set UI_TEST_PYTHON)")
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1])\n"
        "from garment_pipeline.eval_deploy import act_tagrig_ui as U, act_tagrig_deploy as D, height_calib as HC\n"
        "print(json.dumps(dict(settings=str(U.SETTINGS), cams=str(D.CAMS_CFG), start=str(D.START_FILE), hc=str(HC.OUT),\n"
        "                      pending=str(HC.PENDING), here_ui=str(U.HERE), here_dep=str(D.HERE), out=str(D.OUT))))\n")
    env = dict(os.environ, TAGRIG_STATE_DIR=str(tmp_path))
    r = subprocess.run([PY, "-c", code, str(REPO / "src")], capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    names = dict(settings="act_tagrig_ui_settings.json", cams="act_tagrig_cams.json", start="start_pose.json",
                 hc="height_calib.json", pending="height_calib_pending.json")
    for k, n in names.items():
        assert got[k] == str(tmp_path / n), (k, got[k])
    ed = str(REPO / "src/garment_pipeline/eval_deploy")
    assert got["here_ui"] == ed and got["here_dep"] == ed          # module folders unchanged
    assert not got["out"].startswith(str(tmp_path))                 # run outputs are not state


def test_height_calib_path(tmp_path):
    """--height-calib (11-01): start_cmd shifts the start TCP by -xy_B_minus_robot per arm; run_session adds dz_at(row xy), ramped
    over the first 16 rows, to every committed row's z and leaves xy alone (tests/ui/height_calib_probe.py, robot-free, dry)"""
    probe = REPO / "tests/ui/height_calib_probe.py"
    chk = "import flask, cv2, numpy, scipy, pinocchio, pink, ego_teleop; from egogarment import paths as p; assert (p.HANDUMI_SW_ROOT / 'assets/rebot_b601/rebot_b601_bimanual.urdf').exists()"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src")] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    if subprocess.run([PY, "-c", chk], capture_output=True, timeout=300, env=env).returncode != 0:
        pytest.skip(f"{PY}: needs flask, cv2, scipy, pinocchio, pink, the egocentric package and the reBot URDF")
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
