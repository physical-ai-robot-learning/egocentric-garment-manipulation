"""~5 Hz look-ahead check of the whole latest chunk, off the control path (deploy session): IK over every --preview-stride step
(seeded step to step, unbounded solver robotize_ik f0) + workspace / capsule collision / table on those future joint poses. Raises
WARN flags; with --preview-hold a predicted future collision or table hit becomes a HOLD for the control path. Never delays a
commit.
"""

from __future__ import annotations
import threading
import time

import numpy as np

from ..core.contract import CHUNK_ROWS, SIDES

IDLE_S = 0.01
MM_PER_M = 1e3


class Preview(threading.Thread):
    def __init__(self, arms: dict, collision, workspace: dict, args, tableZ: float) -> None:
        super().__init__(daemon=True)
        self.mArms, self.mCollision, self.mWorkspace, self.mArgs, self.mTableZ = arms, collision, workspace, args, tableZ
        self.mJob = None
        self.mLock = threading.Lock()
        self.mStop = False
        self.mResult = None
        self.mLog = []

    def submit(self, t: float, targets: dict, qSeed: dict) -> None:
        with self.mLock:
            self.mJob = (t, targets, qSeed)

    def _check(self, job, t0: float) -> dict:
        tT, tTargets, tSeed = job
        tA = self.mArgs
        tSteps = list(range(tA.preview_stride - 1, CHUNK_ROWS, tA.preview_stride))
        tFlags = []
        tQ = {tSide: tSeed[tSide].copy() for tSide in SIDES}
        tErr = {tSide: [] for tSide in SIDES}
        tClear, tTable = [], []
        tLevel = "HOLD" if tA.preview_hold else "WARN"
        for tK in tSteps:
            for tSide in SIDES:
                tQ[tSide], tPe, _ = self.mArms[tSide].mF0.ik(tQ[tSide], tTargets[tSide][tK])
                tErr[tSide].append(tPe * MM_PER_M)
                tP = tTargets[tSide][tK][:3, 3]
                if np.any(tP < self.mWorkspace[tSide][0]) or np.any(tP > self.mWorkspace[tSide][1]):
                    tFlags.append(f"WARN:future_workspace_{tSide[0]}_k{tK + 1}")
            tCheck = self.mCollision.check(tQ, tableZ=self.mTableZ, armClear=tA.arm_clear)
            tClear.append(tCheck["arm_clear_m"])
            tTable.append(tCheck["table_clear_m"])
            if not tCheck["ok_arm"]:
                tFlags.append(f"{tLevel}:future_collision_k{tK + 1}_{tCheck['pair']}")
            if not tCheck["ok_table"]:
                tFlags.append(f"{tLevel}:future_table_k{tK + 1}")
        tOk = {tSide: float(np.mean(np.array(tErr[tSide]) <= tA.max_ik_mm)) for tSide in SIDES}
        for tSide in SIDES:
            if tOk[tSide] < 1:
                tFlags.append(f"WARN:future_ik_{tSide[0]}_{max(tErr[tSide]):.0f}mm")
        return dict(
            t=tT,
            ik_ok=[tOk["left"], tOk["right"]],
            ik_max_mm=[max(tErr["left"]), max(tErr["right"])],
            min_clear=float(min(tClear)),
            min_table=float(min(tTable)),
            flags=tFlags,
            ms=(time.monotonic() - t0) * MM_PER_M,
        )

    def run(self) -> None:
        while not self.mStop:
            with self.mLock:
                tJob, self.mJob = self.mJob, None
            if tJob is None:
                time.sleep(IDLE_S)
                continue
            tT0 = time.monotonic()
            tResult = self._check(tJob, tT0)
            self.mResult = tResult
            self.mLog.append(tResult)
            time.sleep(max(0.0, 1.0 / self.mArgs.preview_hz - (time.monotonic() - tT0)))
