#!/usr/bin/env python3
"""11-02 flatten build-and-upload loop (dataset stage; was flat_build_upload.py). User "데이터 변환된거는 바로바로 올려줘": build each
finished flattening take as its own LeRobot set (0.5x, Cg = --label fused, continuous jaw = --grip-label cont, stitched masks,
GOLD + SILVER keys from every keys_fl1102*_gs.txt + keys_f7_gs.txt incl. the seed rescue) and rsync it to the RunPod volume
(staging pod) right away; the per-take sets are aggregated on the pod later.
v2 (user "ct gap있는거는 다 test로"): SILVER keys with a CT-gap flag (keys_flat1102_test_gap.txt) -> TEST split; train and test
roots are both uploaded. The 10-30 48-demo pass set (keys_f58_pass.txt, derived cgg_1030f) is built once first as
flat1030_F48_S05C
(11-02 v2: the v1 build was cut off at 34/48 episodes when its loop was stopped -> rebuilt from scratch).
Loop: every --poll-s, build every expected take that has keys and is not done yet (take 140013 waits for keys_fl1102r_gs.txt),
until all expected takes are done. Progress survives restarts in <work-dir>/flat_build_done.json.
Inputs / outputs in --work-dir: the key files above, <name>_split.json, build_<name>.log, flat_build_done.json.
usage: python -m laundry_dataset.builds.build_and_upload --host root@<pod-ip> --ssh-port <port> [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

from laundry_common import paths as egopaths

from .common import RSYNC, TAIL_CHARS, TAKE_DATE_LEN, UPLOAD_TAIL_CHARS, builtCounts, lerobotRoot, log, readKeys, runBuilder

EXPECT = ["20261102_" + tT for tT in ("121342 130406 131156 132540 133624 134434 135136 140013 140903 142422 144657 145344 "
                                      "151622 152215 152909 154442").split()]
WAIT_FOR_RESCUE = {"20261102_140013": "keys_fl1102r_gs.txt"}   # take -> key file that must exist before it is built
PASS_SET = "flat1030_F48"
PASS_SET_NAME = "flat1030_F48_S05C"
PASS_SET_DERIVED = "data/tag_rig_derived_cgg_1030f"
DERIVED = "data/tag_rig_derived_cgg_1102"
REMOTE_DIR = "/workspace/laundry_dit_training/data/lerobot/"
SSH_KEY = egopaths.HOME / ".runpod/ssh/runpodctl-ssh-key"
POLL_S = 60
SPEED = "0.5"
SUFFIX = "S05C"


class _Uploader:
    def __init__(self, host, port, key, remoteDir, rsync):
        self.mSsh = f"ssh -o StrictHostKeyChecking=no -i {key} -p {port}"
        self.mHost, self.mRemoteDir, self.mRsync = host, remoteDir, rsync

    def upload(self, d: Path) -> bool:
        tU = subprocess.run(f"{self.mRsync} -rlt --exclude '._*' -e '{self.mSsh}' {d} {self.mHost}:{self.mRemoteDir}", shell=True,
                            capture_output=True, text=True)
        if tU.returncode != 0:
            log(f"UPLOAD FAILED {d.name}: {tU.stderr[-UPLOAD_TAIL_CHARS:]}")
            return False
        log(f"uploaded {d.name}")
        return True


def _build(name, takes, derived, keys, test, work, uploader):
    """one set: train = keys without a CT-gap flag, test = the flagged ones -> dict(eps, frames, test) or None"""
    tTrain = [tK for tK in keys if tK not in test]
    tTest = [tK for tK in keys if tK in test]
    tSplitFile = work / f"{name}_split.json"
    tMethod = f"11-02 flatten {name}: Cg, 0.5x, continuous jaw; TEST = SILVER with a CT gap flag"
    tSplitFile.write_text(json.dumps({"method": tMethod,
                                      "episodes": {"train": tTrain, "val": [], "test": tTest}}))
    tArgs = ["--label", "fused", "--speed", SPEED, "--mask-mode", "stitched", "--ik-fail-mm", "30", "--render", "0",
             "--overwrite",
             "--grip-label", "cont", "--out", f"data/lerobot/{name}", "--takes", *[str(egopaths.TAG_RIG / tT) for tT in takes],
             "--derived", derived,
             "--demos", *keys, "--old-split", str(tSplitFile)]
    tR = runBuilder(tArgs, work / f"build_{name}.log")
    tOut = lerobotRoot(name, "train")
    tCounts = builtCounts(tOut)
    if tR.returncode != 0 or tCounts is None:
        log(f"BUILD FAILED {name}: {(tR.stdout + tR.stderr)[-TAIL_CHARS:]}")
        return None
    log(f"built {name}: {tCounts[0]} eps {tCounts[1]} frames")
    tOk = uploader.upload(tOut)
    tTestRoot = lerobotRoot(name, "test")
    if tTestRoot.exists():
        tOk = uploader.upload(tTestRoot) and tOk
    return dict(eps=tCounts[0], frames=tCounts[1], test=len(tTest)) if tOk else None


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("--host", required=True, help="ssh target of the staging pod, e.g. root@<ip>")
    tAp.add_argument("--ssh-port", required=True)
    tAp.add_argument("--ssh-key", default=str(SSH_KEY))
    tAp.add_argument("--remote-dir", default=REMOTE_DIR)
    tAp.add_argument("--work-dir", default=".", help="key files, split files, build logs, flat_build_done.json")
    tAp.add_argument("--rsync", default=RSYNC)
    tAp.add_argument("--poll-s", type=float, default=POLL_S)
    tA = tAp.parse_args(argv)
    tWork = Path(tA.work_dir)
    tUploader = _Uploader(tA.host, tA.ssh_port, tA.ssh_key, tA.remote_dir, tA.rsync)
    tDoneFile = tWork / "flat_build_done.json"
    tDone = json.loads(tDoneFile.read_text()) if tDoneFile.exists() else {}
    tTest = set(readKeys(tWork / "keys_flat1102_test_gap.txt"))
    if PASS_SET not in tDone:
        tKeys = readKeys(tWork / "keys_f58_pass.txt")
        tTakes = sorted({tK.split("/")[0] for tK in tKeys})
        tR = _build(PASS_SET_NAME, tTakes, PASS_SET_DERIVED, tKeys, tTest, tWork, tUploader)
        if tR:
            tDone[PASS_SET] = tR
            tDoneFile.write_text(json.dumps(tDone, indent=1))
    while True:
        tKeys = collections.defaultdict(list)
        for tF in glob.glob(str(tWork / "keys_fl1102*_gs.txt")) + [str(tWork / "keys_f7_gs.txt")]:
            for tK in readKeys(tF):
                tKeys[tK.split("/")[0]].append(tK)
        for tT in EXPECT:
            if tT in tDone or tT not in tKeys:
                continue
            if tT in WAIT_FOR_RESCUE and not (tWork / WAIT_FOR_RESCUE[tT]).exists():
                continue
            tR = _build(f"flat1102_{tT[TAKE_DATE_LEN:]}_{SUFFIX}", [tT], DERIVED, sorted(set(tKeys[tT])), tTest, tWork, tUploader)
            if tR:
                tDone[tT] = tR
                tDoneFile.write_text(json.dumps(tDone, indent=1))
        if all(tT in tDone for tT in EXPECT):
            break
        time.sleep(tA.poll_s)
    tEps = sum(tV["eps"] for tV in tDone.values())
    log(f"ALL_UPLOADED {len(tDone)} sets, {tEps} episodes, {sum(tV['frames'] for tV in tDone.values())} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
