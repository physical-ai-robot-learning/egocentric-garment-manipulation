"""Characterization of the fixed-camera pose stage end to end on the synthetic world (synthetic_world.py): every CLI of the chain
runs as a sub-process exactly as the production chain calls it (new_take_prep -> tag rotations -> Orbbec tracker -> candidate ->
CoTracker pack -> [stand-in tracks] -> solve -> gates -> fusion variants -> metrics -> Cg derived root -> episode quality ->
out-of-view -> edge-gap / tighten trims -> QA renders -> whole-demo stitch), plus the library calls that read the trees
(world_calls.py).  Printed lines and every written file are compared with the golden record taken from the pre-refactor code."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from laundry_tracking.tests import golden_record as G
from laundry_tracking.tests import synthetic_world as W

pytest.importorskip("cv2")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

REPO_SRC = Path(__file__).resolve().parents[2]
PACKAGE = REPO_SRC / "laundry_tracking"
TAKE = W.TAKE
TAG_TAKE = "20261030_130000"
STEP_TIMEOUT_S = 900
# Module paths of the code under test (the only lines that change with a module rename).
M_NEW_TAKE_PREP = "laundry_tracking.new_take_prep"
M_TAG_ROT = "laundry_tracking.tag_rotations"
M_TRACKER = "laundry_tracking.orbbec_tracker"
M_CANDIDATE = "laundry_tracking.orbbec_candidate"
M_PACK = "laundry_tracking.cotrack.make_pack"
M_SOLVE = "laundry_tracking.cotrack.point_solver"
M_GATE = "laundry_tracking.cotrack.gate"
M_FUSION = "laundry_tracking.fusion.trajectory_fusion"
M_METRICS = "laundry_tracking.fusion.fusion_metrics"
M_DERIVED = "laundry_tracking.fusion.make_derived"
M_QUALITY = "laundry_tracking.cotrack.episode_quality"
M_OUT_OF_VIEW = "laundry_tracking.cotrack.out_of_view"
M_EDGE_TRIM = "laundry_tracking.edge_gap_trim"
M_TIGHTEN = "laundry_tracking.tighten_trim"
M_RENDER_QA = "laundry_tracking.fusion.render_qa"
M_RENDER_SOURCES = "laundry_tracking.fusion.render_sources"
M_STITCH = "laundry_tracking.whole_demo_stitch"


def _codeAndWork(tmp: Path) -> tuple[Path, Path]:
    """(PYTHONPATH entry with the code under test, EGO_FUSION_WORK).  The golden record was taken from the pre-refactor code, which
    used the folders next to its modules as the working tree (run from a package copy that doubled as FUSION_WORK); the package now
    takes every working path from laundry_common.paths.FUSION_WORK, so it runs in place with a separate work tree."""
    tWork = tmp / "work"
    tWork.mkdir()
    return REPO_SRC, tWork


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tTmp = tmp_path_factory.mktemp("lt_world")
    tCode, tWork = _codeAndWork(tTmp)
    W.writeWorld(tTmp, tWork, withTrims=False)
    tEnv = dict(os.environ)
    tEnv.update(W.worldEnv(tTmp, tWork))
    tEnv["PYTHONPATH"] = os.pathsep.join([str(tCode), str(REPO_SRC)])
    tEnv["PYTHONHASHSEED"] = "0"
    tEnv.pop("V31_P", None)
    tEnv.pop("V31_OUT", None)
    tLog = []

    def run(name, args, extraEnv=None, expectFail=False):
        tE = dict(tEnv)
        tE.update(extraEnv or {})
        tProc = subprocess.run([sys.executable, *args], cwd=tTmp, env=tE, capture_output=True, text=True, timeout=STEP_TIMEOUT_S)
        if not expectFail:
            assert tProc.returncode == 0, f"{name} failed:\n{tProc.stdout[-3000:]}\n{tProc.stderr[-5000:]}"
        tOut = tProc.stdout.replace(str(tWork), "<WORK>").replace(str(tTmp), "<TMP>")
        tOut = re.sub(r"\.bak_(tight|edge)_\d{4}_\d{6}", r".bak_\1_<STAMP>", tOut)
        tErr = tProc.stderr.replace(str(tWork), "<WORK>").replace(str(tTmp), "<TMP>")
        tLog.append(dict(step=name, rc=tProc.returncode, stdout=tOut,
                         stderr_tail=tErr.strip().splitlines()[-1:] if expectFail else []))
        return tProc

    tPack = tTmp / "pack"
    run("new_take_prep", ["-m", M_NEW_TAKE_PREP, TAKE])
    run("tag_rot", ["-m", M_TAG_ROT, str(tTmp / "ego/datasets/tag_rig" / TAG_TAKE)])
    run("tracker_speed", ["-m", M_TRACKER, TAKE], dict(V31_P='{"JUMP":0.10,"V_GATE":1.0}', V31_OUT="trackv31speed"))
    run("tracker_default", ["-m", M_TRACKER, TAKE])
    run("candidate", ["-m", M_CANDIDATE, "--takes", TAKE, "--derived-root", str(tTmp / "cand"), "--conf-dir", str(tTmp / "conf"),
                      "--track", "trackv31speed"])
    run("pack", ["-m", M_PACK, str(tPack), f"{TAKE}:0", f"{TAKE}:1"])
    tFake = subprocess.run([sys.executable, "-c",
                            "import sys; from pathlib import Path; from laundry_tracking.tests import synthetic_world as W\n"
                            "for d in sorted(Path(sys.argv[1]).iterdir()): W.fakeTracks(d)",
                            str(tPack)], env=tEnv, capture_output=True, text=True)
    assert tFake.returncode == 0, tFake.stderr
    run("solve", ["-m", M_SOLVE, str(tPack)])
    run("solve_h1", ["-m", M_SOLVE, str(tPack), "1"])
    run("gate", ["-m", M_GATE, str(tPack)])
    run("gate_h1", ["-m", M_GATE, str(tPack), "1"])
    run("fusion_prod", ["-m", M_FUSION, str(tPack), "R,Cg", "--ct", "solve3", "--suffix", "_g"])
    run("fusion_ablation", ["-m", M_FUSION, str(tPack), "A,B,C,E", "--demos", "0"])
    run("fusion_strict", ["-m", M_FUSION, str(tPack), "C", "--demos", "1", "--strict", "--suffix", "_s"], expectFail=True)
    run("fusion_d", ["-m", M_FUSION, str(tPack), "D"], expectFail=True)
    (tWork / "stitch/qa/eval_t1.csv").write_text(f"var,take,arm,demo,e32_med,e32_p90,n\nfus_Cg_g,{TAKE},left,0,12.5,30.1,40\n"
                                                 f"fus_Cg_g,{TAKE},right,1,9.5,20.0,25\nfus_C,{TAKE},left,0,11.0,28.0,40\n")
    run("metrics", ["-m", M_METRICS, str(tPack), "t1", "fus_Cg_g,fus_R_g,fus_C"])
    run("derived", ["-m", M_DERIVED, str(tPack)], dict(CGG_SRC="tag_rig_derived", CGG_DST="tag_rig_derived_cgg_t"))
    (tTmp / "desktop/orbbec_v31speed_QA_all").mkdir(parents=True, exist_ok=True)
    (tTmp / "desktop/orbbec_v31speed_QA_all/episode_stats.csv").write_text(f"take,demo\n{TAKE},1\n")
    run("quality", ["-m", M_QUALITY, str(tTmp / "q.csv"), str(tPack)])
    (tTmp / "renders/GOLD").mkdir(parents=True)
    (tTmp / "renders/GOLD" / f"{TAKE}_d01.mp4").write_text("x")
    (tTmp / "renders/REJECT").mkdir(parents=True)
    (tTmp / "renders/REJECT" / f"{TAKE}_d00.mp4").write_text("y")
    (tTmp / "oov").mkdir()                       # the tool writes its table before creating any class folder
    run("out_of_view", ["-m", M_OUT_OF_VIEW, str(tTmp / "q.csv"), str(tTmp / "renders"), str(tTmp / "oov"), str(tPack)])
    run("render_qa", ["-m", M_RENDER_QA, "--pack", str(tPack), "--var", "fus_Cg_g", "--demos", f"{TAKE}:0", f"{TAKE}:1", "--out",
                      str(tTmp / "qa_vid"), "--raw-var", "fus_R_g", "--quality-csv", str(tTmp / "q.csv"), "--tag-csv",
                      "fusion_metrics_t1.csv", "--prefix", "q"])
    run("render_sources", ["-m", M_RENDER_SOURCES, "--pack", str(tPack), "--demo", f"{TAKE}:1", "--out", str(tTmp / "src.gif")])
    run("stitch_visual", ["-m", M_STITCH, TAKE, "right", "v3on", "--offsets", "0,-1", "--demos", "0",
                          "--visual", str(tWork / "visual_{take}_{arm}.npz")])
    run("stitch_raw", ["-m", M_STITCH, TAKE, "left", "v2", "--offsets", "-1", "--demos", "1", "--name", "v2_raw"])
    run("stitch_mode_a", ["-m", M_STITCH, TAKE, "left", "v3off", "--offsets", "2", "--demos", "0", "--mode", "A", "--abs-prior"])
    run("world_calls", ["-m", "laundry_tracking.tests.world_calls", str(tTmp / "world_calls.json")])
    run("edge_gap_trim", ["-m", M_EDGE_TRIM, str(tPack)])
    (tTmp / "aud.csv").write_text(f"take,demo,head_s,tail_s\n{TAKE},0,1.0,0.6\n{TAKE},1,nan,\n")
    run("tighten_trim", ["-m", M_TIGHTEN, str(tTmp / "aud.csv"), "0.2"])
    return dict(tmp=tTmp, work=tWork, log=tLog)


def _outputs(tTmp: Path, tWork: Path) -> dict:
    tOut = {}
    tRoots = [("WORK", tWork), ("TMP", tTmp)]
    for tLabel, tRoot in tRoots:
        for tPath in sorted(tRoot.rglob("*")):
            if not tPath.is_file() or "__pycache__" in tPath.parts or tPath.suffix in (".py", ".jpg", ".png", ".pyc"):
                continue
            if tLabel == "TMP" and (tWork in tPath.parents or (tTmp / "code") in tPath.parents):
                continue
            if tPath.name == "body_offset_by_rig_v1.json" or tPath.name == "world_calls.json":
                continue
            tRel = str(tPath.relative_to(tRoot))
            tRel = re.sub(r"\.bak_(tight|edge)_\d{4}_\d{6}", r".bak_\1_<STAMP>", tRel)
            tOut[f"{tLabel}/{tRel}"] = G.fileSummary(tPath)
    return tOut


def test_pipeline_stdout(world):
    G.checkGolden("pipeline_stdout", world["log"])


def test_pipeline_files(world):
    tText = G.dumps(_outputs(world["tmp"], world["work"])).replace(str(world["work"]), "<WORK>").replace(str(world["tmp"]),
            "<TMP>")
    G.checkGolden("pipeline_files", G.loads(tText))


def test_world_calls(world):
    G.checkGolden("world_calls", G.loads((world["tmp"] / "world_calls.json").read_text()))
