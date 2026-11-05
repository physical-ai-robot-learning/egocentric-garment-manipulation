"""[2026-10-28] Run exactly N policy steps on the real robot from the current pose (user: "여기서 3스텝만 가보자"), through the SAME
runSession step mode as the UI ([한 스텝 실행]): the policy plans continuously, each step token sends ONE committed segment
(--commit steps), the next token is issued only after the previous one was consumed + --gap seconds; after N executed steps the run
stops (bridge /estop = hold the measured pose).  All safety checks of runSession stay on.
    DEPLOY_FAKE_WRIST=right python -m laundry_deploy.tools.run_n_steps --ckpt <dir> --n 3 [--commit 4 --speed 0.5]
Was laundry_deploy/run_n_steps.py.
"""

from __future__ import annotations
import argparse
import sys
import threading
import time

from ..core.errors import DeployError
from ..session.cli import buildParser as buildDeployParser
from ..session.rig import Rig
from ..session.run_loop import runSession


class StepControl:
    """Run control that hands out N step tokens, one per --gap seconds after the previous one was used. Was Ctl."""

    def __init__(self, n: int, gapS: float) -> None:
        self.mN, self.mGapS = n, gapS
        self.mLock = threading.Lock()
        self.mStepN = 0
        self.mStepsDone = 0
        self.mStopReq = None
        self.mSnap = None
        self.mStepMode = True
        self.mTLast = None

    def takeStep(self) -> bool:
        with self.mLock:
            if self.mStepN <= 0:
                return False
            self.mStepN -= 1
            self.mStepsDone += 1
            self.mTLast = time.monotonic()
            print(f"[n_steps] 스텝 {self.mStepsDone}/{self.mN} 실행", flush=True)
            return True

    def poll(self):
        with self.mLock:
            if self.mStopReq:
                return self.mStopReq
            if self.mStepsDone >= self.mN and self.mStepN == 0:
                if time.monotonic() - self.mTLast > self.mGapS:
                    self.mStopReq = f"{self.mN} 스텝 완료"
            elif self.mStepN == 0 and (self.mTLast is None or time.monotonic() - self.mTLast > self.mGapS):
                self.mStepN = 1
        return self.mStopReq

    def publish(self, snapshot) -> None:
        self.mSnap = snapshot


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--ckpt", required=True)
    tParser.add_argument("--policy", default="dit")
    tParser.add_argument("--n", type=int, default=3)
    tParser.add_argument("--commit", type=int, default=4)
    tParser.add_argument("--speed", type=float, default=0.5)
    tParser.add_argument("--gap", type=float, default=2.0)
    tParser.add_argument("--dit-steps", type=int, default=4)
    tParser.add_argument("--max-img-age-ms", type=float, default=600.0)
    tParser.add_argument("--max-plan-age-s", type=float, default=1.0)
    tParser.add_argument("--warmup", type=float, default=4.0, help="s of planning (preview only) before step 1")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRa = buildDeployParser().parse_args(["run"])
    tRa.policy = tArgs.policy
    tRa.ckpt = tArgs.ckpt
    tRa.commit = tArgs.commit
    tRa.speed = tArgs.speed
    tRa.slow_start_speed = tArgs.speed
    tRa.secs = 0.0
    tRa.dit_steps = tArgs.dit_steps
    tRa.go = True
    tRa.step_mode = True
    tRa.allow_start_pose = True
    tRa.max_img_age_ms = tArgs.max_img_age_ms  # step mode: the robot is still while waiting
    tRa.max_plan_age_s = tArgs.max_plan_age_s
    try:
        tRig = Rig(tRa).start()
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    tCtl = StepControl(tArgs.n, tArgs.gap)
    tCtl.mTLast = time.monotonic() + tArgs.warmup - tArgs.gap  # first step after the warm-up
    try:
        tSummary = runSession(tRa, tRig, tCtl)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        tRig.close()
        return 1
    print("[n_steps] 끝:", (tSummary or {}).get("stop_reason"), "| run", (tSummary or {}).get("run"), flush=True)
    tRig.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
