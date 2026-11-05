"""Characterization of the dataset-stage scripts: idle_trim (writes idle_trim.csv), ik_qa_stats, and the 11-02..11-04 flatten
build scripts of builds/ (split decisions, the builder commands they launch, the files they write). Every external program they
start (the IK interpreter running the dataset builder, the fusion QA renderer, rsync) is replaced by a recorder on a fake HOME /
EGO_* tree, so no dataset is built, nothing is rendered and nothing is uploaded."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .synthetic import T0_NS, TAKES

SRC = Path(__file__).resolve().parents[2]
PKG = SRC / "laundry_dataset"
PY = sys.executable

FAKE_TOOL = r'''#!{py}
"""records one call of an external tool; emulates the dataset builder's output roots and the QA renderer's video"""
import json, os, pathlib, sys
home = pathlib.Path(os.environ["HOME"])
argv = sys.argv[1:]
if argv and argv[0] == "-m":
    tool, rest = argv[1].rsplit(".", 1)[-1], argv[2:]
elif argv:
    tool, rest = pathlib.Path(argv[0]).name[:-3] if argv[0].endswith(".py") else pathlib.Path(sys.argv[0]).name, argv[1:]
if pathlib.Path(sys.argv[0]).name == "rsync":
    tool, rest = "rsync", argv
with open(home / "calls.jsonl", "a") as fh:
    fh.write(json.dumps(dict(tool=tool, args=rest, cwd=os.getcwd(), FILL_MASK_S=os.environ.get("FILL_MASK_S"))) + "\n")
if tool == "tag_rig_unsplit":
    out = pathlib.Path(rest[rest.index("--out") + 1])
    sp = json.loads(pathlib.Path(rest[rest.index("--old-split") + 1]).read_text())["episodes"]
    for s, ks in sp.items():
        ks = [k for k in ks if not k.endswith("demo09")]                 # one demo the builder drops
        r = out.parent / f"{out.name}_{s}"
        (r / "meta").mkdir(parents=True, exist_ok=True)
        if ks:
            (r / "meta" / "info.json").write_text(json.dumps(dict(total_episodes=len(ks), total_frames=100 * len(ks))))
    print("built", out.name)
if tool == "render_fusion_qa":
    out = pathlib.Path(rest[rest.index("--out") + 1]); dm = rest[rest.index("--demos") + 1]
    if not dm.endswith(":1"):
        (out / ("v_" + dm.replace(":", "_") + ".mp4")).write_text("mp4")
    else:
        print("no video for", dm)
'''


def _calls(home):
    tFile = home / "calls.jsonl"
    if not tFile.exists():
        return []
    tOut = [json.loads(tX) for tX in tFile.read_text().splitlines()]
    return sorted(json.dumps(_norm(tC, home), sort_keys=True, ensure_ascii=False) for tC in tOut)


def _norm(value, home):
    if isinstance(value, dict):
        return {tK: _norm(tV, home) for tK, tV in value.items()}
    if isinstance(value, list):
        return [_norm(tV, home) for tV in value]
    if isinstance(value, str):
        return value.replace(str(home), "<HOME>")
    return value


def _log(text, home):
    tText = re.sub(r"^\[\d\d:\d\d:\d\d\] ", "[T] ", text.replace(str(home), "<HOME>"), flags=re.M)
    return sorted(tText.splitlines())


@pytest.fixture
def home(world, tmp_path):
    """fake HOME == EGO_* roots: tag_rig -> the world's takes, idle_trim.csv, derived sets named by take, the fake tools"""
    tHome = tmp_path / "home"
    tTrain = tHome / "laundry_dit_training"
    (tHome / "ego_collector/datasets").mkdir(parents=True)
    os.symlink(world.tag_rig, tHome / "ego_collector/datasets/tag_rig")
    (tTrain / "reports").mkdir(parents=True)
    shutil.copy(world.trim, tTrain / "reports/idle_trim.csv")
    (tTrain / "datasets").mkdir()
    os.symlink(PKG / "tag_rig_to_lerobot.py", tTrain / "datasets/tag_rig_to_lerobot.py")
    tDer = tTrain / "data/der_test"
    tDer.mkdir(parents=True)
    for tName, tSrc in world.takes.items():
        os.symlink(world.derived / (tName + tSrc), tDer / tName)
    tTool = FAKE_TOOL.replace("{py}", PY)
    for tRel in ("xvla-mac/bin/python", "ego_collector/.venv/bin/python", "bin/rsync"):
        (tHome / tRel).parent.mkdir(parents=True, exist_ok=True)
        (tHome / tRel).write_text(tTool)
        (tHome / tRel).chmod(0o755)
    (tHome / "ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b/fusion").mkdir(parents=True)
    return tHome


def _env(home, **extra):
    tEnv = dict(os.environ, HOME=str(home), PYTHONPATH=str(SRC), EGO_DATA_ROOT=str(home / "ego_collector"),
                EGO_TRAIN_ROOT=str(home / "laundry_dit_training"), EGO_IK_PYTHON=str(home / "xvla-mac/bin/python"),
                EGO_DESKTOP=str(home / "Desktop"))
    for tKey in ("FILL_MASK_S", "GOLD_GAP_MAX", "NHOLD", "GOLDGAP_TO", "GOLDGAP_TO_VALTEST", "FIXED_SPLIT", "UPLOAD",
                 "RENDER_OUT"):
        tEnv.pop(tKey, None)
    tEnv.update(extra)
    return tEnv


def _run(cmd, env, cwd=None):
    tR = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert tR.returncode == 0, tR.stdout + tR.stderr
    return tR.stdout


# ---------------------------------------------------------------- idle_trim + ik_qa_stats
def test_idle_trim(golden, world, tmp_path):
    tTrain = tmp_path / "train"
    tTrain.mkdir()
    os.symlink(world.train / "data", tTrain / "data")
    (tTrain / "reports").mkdir()
    tEnv = dict(os.environ, PYTHONPATH=str(SRC), EGO_DATA_ROOT=str(world.ego), EGO_TRAIN_ROOT=str(tTrain))
    tOut = _run([PY, "-m", "laundry_dataset.idle_trim"], tEnv)
    golden.check("scr/idle_trim_csv", (tTrain / "reports/idle_trim.csv").read_text())
    golden.check("scr/idle_trim_stdout", tOut.replace(str(tTrain), "<TRAIN>"))


def test_ik_qa_stats(golden, world, tmp_path, monkeypatch, capsys):
    from laundry_dataset import ik_qa_stats as Q
    tRoot = tmp_path / "root"
    tRoot.mkdir()
    for tName, tSrc in world.takes.items():
        if tSrc:
            os.symlink(world.derived / (tName + tSrc), tRoot / tName)
    tQ = tmp_path / "q.csv"
    tQ.write_text("take,demo,class\n20261022_124754,0,GOLD\n20261022_124754,1,SILVER\n20261023_113550,0,REJECT\n"
                  "20261023_113550,1,GOLD\n")
    monkeypatch.setattr(Q, "TRIM_CSV", world.trim)
    assert Q.main([str(tRoot), str(tQ), str(tmp_path / "out")]) == 0
    for tName in ("ik_qa_summary.csv", "ik_qa_causes.csv", "ik_qa_episodes.csv"):
        golden.check(f"scr/ikqa_{tName}", (tmp_path / "out" / tName).read_text())
    golden.check("scr/ikqa_stdout", capsys.readouterr().out)
    tRng = np.random.default_rng(0)
    tD = dict(epos=tRng.uniform(0, 60, 50), erot=tRng.uniform(0, 60, 50), push=tRng.uniform(0, 1, 50), lim=tRng.uniform(0, 3, 50),
              roll=tRng.uniform(0, 1, 50), flip=True, zr=tRng.uniform(0, 0.1, 50), reach=tRng.uniform(0.4, 0.7, 50))
    golden.check("scr/ikqa_cause", Q.cause(tD).astype(str))


# ---------------------------------------------------------------- builds/
def _packs(home, world):
    """two CoTracker packs: <take>_d<N>/solve2_{left,right}.npz (t ns, conf) with a few zero-confidence runs"""
    tOut = []
    for tPk, tTakes in (("pkA", TAKES[:2]), ("pkB", TAKES[2:])):
        for tTake in tTakes:
            for tDemo in (0, 1):
                tD = home / tPk / f"{tTake}_d{tDemo}"
                tD.mkdir(parents=True)
                tT = (T0_NS + np.arange(0.0, 21.0, 1 / 30) * 1e9).astype(np.int64)
                for tSide, tGaps in (("left", [(4.0, 4.1)] if tDemo == 0 else []), ("right", [(6.0, 6.5), (14.0, 14.15)])):
                    tConf = np.ones(len(tT))
                    for tA, tB in tGaps:
                        tConf[(tT >= T0_NS + tA * 1e9) & (tT < T0_NS + tB * 1e9)] = 0
                    tSel = slice(3, None) if tSide == "left" else slice(None)
                    np.savez(tD / f"solve2_{tSide}.npz", t=tT[tSel], conf=tConf[tSel])
        (home / tPk / "notapack").mkdir()
        tOut.append(home / tPk)
    return tOut


def test_ct_gap_audit(golden, world, home):
    tPacks = _packs(home, world)
    tOut = home / "ctgap.csv"
    tLog = _run([PY, "-m", "laundry_dataset.builds.ct_gap_audit", str(tOut), *map(str, tPacks)], _env(home))
    golden.check("scr/ctgap_csv", tOut.read_text())
    golden.check("scr/ctgap_stdout", tLog)


def test_idle_audit(golden, world, home):
    tOut = home / "idle_audit.csv"
    tLog = _run([PY, "-m", "laundry_dataset.builds.idle_audit", str(tOut), "0.03", "0.4", "der_test", "2026102*"], _env(home))
    golden.check("scr/idle_audit_csv", tOut.read_text())
    golden.check("scr/idle_audit_stdout", tLog)
    tOut2 = home / "idle_audit2.csv"
    tNew = lambda name: "20261103_" + name.split("_")[1]
    tDer = home / "laundry_dit_training/data/tag_rig_derived_cgg_1103n"
    tDer.mkdir()
    os.unlink(home / "ego_collector/datasets/tag_rig")
    for tName, tSrc in world.takes.items():
        os.symlink(world.derived / (tName + tSrc), tDer / tNew(tName))
        (home / "ego_collector/datasets/tag_rig").mkdir(exist_ok=True)
        os.symlink(world.tag_rig / tName, home / "ego_collector/datasets/tag_rig" / tNew(tName))
    with open(world.trim) as tFh:
        tTrimText = tFh.read().replace("20261022_", "20261103_").replace("20261023_", "20261103_")
    (home / "laundry_dit_training/reports/idle_trim.csv").write_text(tTrimText)
    tLog2 = _run([PY, "-m", "laundry_dataset.builds.idle_audit", str(tOut2)], _env(home))
    golden.check("scr/idle_audit2_csv", tOut2.read_text())
    golden.check("scr/idle_audit2_stdout", tLog2)


def _keys():
    return [f"{tT}/demo{tD:02d}" for tT in ("20261104_100000", "20261104_110000", "20261104_120000") for tD in range(10)]


def _gsrInputs(work):
    tKeys = _keys()
    tCls = ["GOLD", "GOLD", "SILVER", "GOLD", "REJECT", "GOLD", "GOLD", "SILVER", "GOLD", "GOLD"]
    with open(work / "q1.csv", "w") as tFh:
        tFh.write("take,demo,class\n")
        for tI, tK in enumerate(tKeys[:20]):
            tT, tD = tK.split("/")
            tFh.write(f"{tT},{int(tD[4:])},{tCls[tI % 10]}\n")
    with open(work / "q2.csv", "w") as tFh:
        tFh.write("take,demo,class\n")
        for tI, tK in enumerate(tKeys[18:]):
            tT, tD = tK.split("/")
            tFh.write(f"{tT},{int(tD[4:])},{tCls[(tI + 3) % 10]}\n")
    with open(work / "g1.csv", "w") as tFh:
        tFh.write("take,demo,gap_frames,max_gap_s,L_gap_s,R_gap_s\n")
        for tI, tK in enumerate(tKeys[:-2]):
            tT, tD = tK.split("/")
            tGap = [0, 0, 3, 0, 0, 12, 0, 1, 0, 0][tI % 10]
            tFh.write(f"{tT},{int(tD[4:])},{tGap},{[0, 0, 0.1, 0, 0, 0.6, 0, 0.2, 0, 0][tI % 10]},0,0\n")


@pytest.mark.parametrize("variant", ["default", "goldgap_train", "goldgap_none", "nhold2_excl"])
def test_split_gold_silver_reject(golden, home, variant):
    tWork = home / "work"
    tWork.mkdir()
    _gsrInputs(tWork)
    tVariantEnv = {"goldgap_train": {"GOLDGAP_TO": "train"}, "goldgap_none": {"GOLDGAP_TO_VALTEST": "0", "GOLD_GAP_MAX": "0.15"},
                   "nhold2_excl": {"NHOLD": "2"}, "default": {}}
    tEnv = _env(home, **tVariantEnv[variant])
    tArgs = ["FLX", "data/der_x", "q1.csv,q2.csv", "g1.csv"]
    tArgs += ["20261104_100000/demo01,20261104_110000/demo03"] if variant == "nhold2_excl" else []
    tLog = _run([PY, "-m", "laundry_dataset.builds.split_gold_silver_reject", *tArgs, "--work-dir", str(tWork)], tEnv, cwd=tWork)
    golden.check(f"scr/gsr_{variant}_log", _log(tLog, home))
    golden.check(f"scr/gsr_{variant}_calls", _calls(home))
    for tF in sorted(tWork.glob("*_split.json")):
        golden.check(f"scr/gsr_{variant}_{tF.name}", tF.read_text())
    # 11-04 old run: + the copied flat_gsr_build.py
    golden.check(f"scr/gsr_{variant}_files", sorted(tP.name for tP in tWork.iterdir()))


@pytest.mark.parametrize("fixed", [False, True])
def test_flat1103n_build(golden, home, fixed):
    tWork = home / "work"
    tWork.mkdir()
    tKeys = _keys() + ["20261104_130000/demo00", "20261104_130000/demo01", "20261104_100000/demo03"]
    (tWork / "keys_fl1103n_gs.txt").write_text("\n".join(tKeys))
    if fixed:
        tHeld = {"val": ["20261104_100000/demo02"], "test": ["20261104_120000/demo07"]}
        (tWork / "fl1103n_split.json").write_text(json.dumps(tHeld))
    tLog = _run([PY, "-m", "laundry_dataset.builds.per_take_holdout_build", "--work-dir", str(tWork)],
                _env(home, UPLOAD="0", **({"FIXED_SPLIT": "1"} if fixed else {})), cwd=tWork)
    golden.check(f"scr/f1103n_{fixed}_log", _log(tLog, home))
    golden.check(f"scr/f1103n_{fixed}_calls", _calls(home))
    for tF in sorted(tWork.glob("*.json")):
        golden.check(f"scr/f1103n_{fixed}_{tF.name}", tF.read_text())


def test_flat1103n_upload(golden, home):
    tWork = home / "work"
    tWork.mkdir()
    (tWork / "keys_fl1103n_gs.txt").write_text("\n".join(_keys()))
    tLog = _run([PY, "-m", "laundry_dataset.builds.per_take_holdout_build", "--work-dir", str(tWork), "--rsync",
                 str(home / "bin/rsync")],
                _env(home), cwd=tWork)
    golden.check("scr/f1103n_upload_calls", _calls(home))
    golden.check("scr/f1103n_upload_log", _log(tLog, home))


def test_flat_build_upload(golden, home):
    tWork = home / "work"
    tWork.mkdir()
    tTakes = ("121342 130406 131156 132540 133624 134434 135136 140013 140903 142422 144657 145344 151622 152215 152909 "
              "154442").split()
    tKeys = [f"20261102_{tT}/demo{tD:02d}" for tT in tTakes for tD in range(3)]
    (tWork / "keys_fl1102a_gs.txt").write_text("\n".join(tKeys[:30]))
    (tWork / "keys_fl1102r_gs.txt").write_text("\n".join(tKeys[30:]))
    (tWork / "keys_f7_gs.txt").write_text("\n".join(tKeys[:2]))
    (tWork / "keys_flat1102_test_gap.txt").write_text("20261102_121342/demo01\n20261102_154442/demo02\n20261030_100000/demo09")
    (tWork / "keys_f58_pass.txt").write_text("20261030_100000/demo00 20261030_100000/demo09 20261030_110000/demo01")
    tLog = _run([PY, "-m", "laundry_dataset.builds.build_and_upload", "--host", "root@38.80.152.147", "--ssh-port", "39064",
                 "--work-dir", str(tWork),
                 "--rsync", str(home / "bin/rsync")], _env(home), cwd=tWork)
    golden.check("scr/fbu_log", _log(tLog, home))
    golden.check("scr/fbu_calls", _calls(home))
    for tF in sorted(tWork.glob("*.json")):
        golden.check(f"scr/fbu_{tF.name}", tF.read_text())


def test_render_qa_videos(golden, home):
    tWork = home / "work"
    tWork.mkdir()
    for tPk, tRows in (("pkA", [("20261103_100000", 0, "GOLD"), ("20261103_100000", 1, "SILVER"),
                                ("20261103_110000", 0, "REJECT")]),
                       ("pkB", [("20261103_120000", 2, "GOLD"), ("20261103_130000", 0, "GOLD")])):
        with open(tWork / f"q_{tPk}.csv", "w") as tFh:
            tFh.write("take,demo,class\n" + "".join(f"{tT},{tD},{tC}\n" for tT, tD, tC in tRows))
    (tWork / "exclude_takes_1103.txt").write_text("20261103_130000\n")
    tOut = home / "videos"
    tOut.mkdir()
    (tOut / "20261103_120000_d03_GOLD.mp4").write_text("already")
    tLog = _run([PY, "-m", "laundry_dataset.builds.render_qa_videos", "pkA", "pkB", "--work-dir", str(tWork)],
                _env(home, RENDER_OUT=str(tOut)), cwd=tWork)
    golden.check("scr/render_log", _log(tLog, home))
    golden.check("scr/render_calls", _calls(home))
    golden.check("scr/render_files", sorted(tP.name for tP in tOut.iterdir()))


def test_split_rules_of_11_04_evening():
    # 0.25 s gap threshold, gap-free SILVER -> train, FORCE_TRAIN, no clean holdout (NHOLD=0)
    import pandas as pd

    from laundry_dataset.builds.split_gold_silver_reject import assignSplits
    tCls = pd.Series({"t1/demo01": "GOLD", "t1/demo02": "GOLD", "t1/demo03": "GOLD", "t1/demo04": "SILVER",
                      "t1/demo05": "SILVER", "t1/demo06": "REJECT"})
    tGapFrames = pd.Series({"t1/demo01": 0, "t1/demo02": 5, "t1/demo03": 9, "t1/demo04": 0, "t1/demo05": 3, "t1/demo06": 0})
    tMaxGap = pd.Series({"t1/demo01": 0.0, "t1/demo02": 0.22, "t1/demo03": 0.4, "t1/demo04": 0.0, "t1/demo05": 0.1,
                         "t1/demo06": 0.0})
    _, tSplit, tHeld = assignSplits(tCls, tGapFrames, tMaxGap, set(), 0, 0.25, "", "1", "1", ())
    assert tSplit == {"t1/demo01": "train", "t1/demo02": "train", "t1/demo03": "val", "t1/demo04": "train",
                      "t1/demo05": "silver", "t1/demo06": "reject"}
    assert tHeld == {"val": ["t1/demo03"], "test": []}
    _, tSplit, tHeld = assignSplits(tCls, tGapFrames, tMaxGap, set(), 0, 0.25, "", "1", "1", ("t1/demo03",))
    assert tSplit["t1/demo03"] == "train" and tHeld == {"val": [], "test": []}
