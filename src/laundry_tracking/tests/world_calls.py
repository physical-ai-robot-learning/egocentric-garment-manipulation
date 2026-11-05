"""Library calls of laundry_tracking that read the recording / fusion trees, run inside the synthetic world (test helper).

    python -m laundry_tracking.tests.world_calls OUT_JSON

Runs with the EGO_* environment of synthetic_world.worldEnv (laundry_common.paths reads it at import, so this must be a fresh
process) and writes {name: golden_record.summarize(result)}."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from laundry_tracking.tests import golden_record as G
from laundry_tracking.tests import synthetic_world as W


class _Captured(Exception):
    pass


def _parsedArgs(run, argv: list[str]) -> dict:
    """the argparse Namespace a CLI main builds from argv (parse_args is intercepted, nothing runs)"""
    tOriginal = argparse.ArgumentParser.parse_args
    tBox = {}

    def _capture(self, args=None, namespace=None):
        tBox["ns"] = tOriginal(self, args, namespace)
        raise _Captured()
    argparse.ArgumentParser.parse_args = _capture
    tSaved = sys.argv
    sys.argv = ["prog"] + list(argv)
    try:
        run(argv)
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = tOriginal
        sys.argv = tSaved
    return vars(tBox["ns"])


def main(argv: list[str] | None = None) -> int:
    tOutPath = Path((argv if argv is not None else sys.argv[1:])[0])
    from laundry_tracking import camera_calibration as camcal
    from laundry_tracking import slam_consistency as C
    from laundry_tracking.fgo import arm_data as FD
    from laundry_tracking.fgo import pose_smoother
    from laundry_tracking import imu_motion as IM
    from laundry_tracking import arm_rotation as V
    from laundry_common import paths

    tRes = {}
    tRes["trims"] = {tTake: [dict(demo=tW.mDemo, a0=tW.mA0, w0=tW.mW0, w1=tW.mW1, kept_s=tW.mKeptS) for tW in tWs]
                     for tTake, tWs in FD.loadTrimWindows().items()}
    tRes["assignment"] = FD.loadAssignment(W.TAKE)
    tRes["handeye_1022"] = FD.meanHandEye1022("wrist")
    tRes["slam_ready"] = [FD.slamReady(W.TAKE, "wrist"), FD.slamReady(W.TAKE, "wrist_left"), FD.slamReady("20261030_130000",
            "wrist")]
    for tArm in W.ARMS:
        tD = FD.loadArm(W.TAKE, tArm, withSlam=True)
        tRes[f"load_arm_{tArm}"] = {tKey: tVal for tKey, tVal in tD.items() if tKey not in ("raw",)}
    tD = FD.loadArm(W.TAKE, "right", withSlam=False)
    tVisual = paths.FUSION_WORK / f"visual_{W.TAKE}_right.npz"
    FD.setVisual(tD, tVisual, timeOffsetNs=5_000_000)
    tRes["set_visual"] = tD["segs"]
    with tempfile.TemporaryDirectory() as tTmp:
        FD.exportVisual(tD, Path(tTmp) / "v.npz")
        tRes["export_visual"] = G.fileSummary(Path(tTmp) / "v.npz")
    tD = FD.loadArm(W.TAKE, "left", withSlam=False)
    FD.setVisual(tD, paths.FUSION_WORK / f"visual_{W.TAKE}_left.npz", minConf=0.5)
    tRes["set_visual_minconf"] = tD["segs"]
    tRes["camcal"] = {tTake: [str(camcal.calibPath(tTake).relative_to(paths.DATA_CALIB)), camcal.baseFromCamera(tTake), camcal.cameraFromBase(tTake)]
                      for tTake in ("20261020_100000", "20261026_111328", "20261026_115441", "20261026_120052", W.TAKE)}
    tRes["load_segs_raw"] = C.loadRawSegments(W.TAKE, "wrist")
    tDf, tCal, _, tSegs = C.evaluateArm(W.TAKE, "right", demos=[0])
    tRes["evaluate_arm_right"] = dict(df={tCol: tDf[tCol].to_numpy() for tCol in tDf.columns if tDf[tCol].dtype.kind in "fiub"},
                                      cols=sorted(tDf.columns), cal=tCal)
    tDf, tCal, _, _ = C.evaluateArm(W.TAKE, "left")
    tRes["evaluate_arm_left"] = dict(df={tCol: tDf[tCol].to_numpy() for tCol in tDf.columns if tDf[tCol].dtype.kind in "fiub"},
                                     cal=tCal)
    tD = FD.loadArm(W.TAKE, "left", withSlam=False)
    pose_smoother.prepGyro(tD)
    tRes["rest_bias"] = IM.restBias(tD, FD.loadTrimWindows()[W.TAKE])
    tTrack = np.load(paths.FUSION_WORK / "orbbec/v2b/out" / f"trackv31speed_{W.TAKE}.npz")
    tRes["arm_rotations"] = V.armRotations(W.TAKE, tTrack["t_ns"], tTrack["c_left"], "left")
    tAnchor = np.ones(len(tTrack["t_ns"]), bool)
    tAnchor[::3] = False
    tRes["arm_rotations_anchor"] = V.armRotations(W.TAKE, tTrack["t_ns"], tTrack["c_right"], "right", anchorOk=tAnchor)
    # CLI argument defaults (argparse mains)
    from laundry_tracking import orbbec_candidate as build_candidate_v0
    from laundry_tracking.fusion import render_qa as render_fusion_qa, render_sources as render_traj_sources, trajectory_fusion as trajectory_fusion_v1
    from laundry_tracking import whole_demo_stitch as wdo
    tCli = {}
    tCli["wdo"] = _parsedArgs(lambda a: wdo.main(a), [W.TAKE, "left", "v2"])
    tCli["build_candidate"] = _parsedArgs(lambda a: build_candidate_v0.main(a), ["--takes", W.TAKE, "--derived-root", "d",
                                                                                 "--conf-dir", "c"])
    tCli["trajectory_fusion"] = _parsedArgs(lambda a: trajectory_fusion_v1.main(a), ["pack", "R,Cg"])
    tCli["render_fusion_qa"] = _parsedArgs(lambda a: render_fusion_qa.main(a), ["--pack", "p", "--var", "v", "--demos", "a:0",
                                                                                "--out", "o"])
    tCli["render_traj_sources"] = _parsedArgs(lambda a: render_traj_sources.main(a), ["--pack", "p", "--demo", "a:0", "--out",
                                                                                      "o.gif"])
    tRes["cli"] = tCli
    tOutPath.write_text(G.dumps(G.summarize(tRes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
