"""~5 Hz look-ahead check of the whole latest chunk (IK, workspace, capsule collision, table), off the control path.
"""
from __future__ import annotations
import threading
import time

import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core


# ---------------------------------------------------------------- run
# ---------------------------------------------------------------- 5 Hz preview (whole chunk)
class Preview(threading.Thread):
    """~5 Hz: IK over the WHOLE latest chunk (every --preview-stride step, seeded step to step, unbounded solver robotize_ik f0) +
    workspace / capsule collision / table on those future joint poses.  Raises WARN flags; with --preview-hold a predicted future
    collision or table hit becomes a HOLD (STOP) for the control path.  Off the control path (never delays a commit)."""
    def __init__(self, arms, col, WS, a, table_z):
        super().__init__(daemon=True); self.arms, self.col, self.WS, self.a, self.table_z = arms, col, WS, a, table_z
        self.job = None; self.lock = threading.Lock(); self.stop = False; self.res = None; self.log = []

    def submit(self, t, Tc, q_seed):
        with self.lock: self.job = (t, Tc, q_seed)

    def run(self):
        while not self.stop:
            with self.lock: job, self.job = self.job, None
            if job is None: time.sleep(0.01); continue
            t0 = time.monotonic(); t, Tc, qs = job; ks = list(range(self.a.preview_stride - 1, 32, self.a.preview_stride)); flags = []
            q = {s: qs[s].copy() for s in core.SIDES}; err = {s: [] for s in core.SIDES}; clr, tab = [], []
            for k in ks:
                for s in core.SIDES:
                    q[s], pe, re = self.arms[s].f0.ik(q[s], Tc[s][k]); err[s].append(pe * 1e3)
                    if np.any(Tc[s][k][:3, 3] < self.WS[s][0]) or np.any(Tc[s][k][:3, 3] > self.WS[s][1]): flags.append(f"WARN:future_workspace_{s[0]}_k{k + 1}")
                cc = self.col.check(q, table_z=self.table_z, arm_clear=self.a.arm_clear); clr.append(cc["arm_clear_m"]); tab.append(cc["table_clear_m"])
                if not cc["ok_arm"]: flags.append(f"{'HOLD' if self.a.preview_hold else 'WARN'}:future_collision_k{k + 1}_{cc['pair']}")
                if not cc["ok_table"]: flags.append(f"{'HOLD' if self.a.preview_hold else 'WARN'}:future_table_k{k + 1}")
            ok = {s: float(np.mean(np.array(err[s]) <= self.a.max_ik_mm)) for s in core.SIDES}
            for s in core.SIDES:
                if ok[s] < 1: flags.append(f"WARN:future_ik_{s[0]}_{max(err[s]):.0f}mm")
            r = dict(t=t, ik_ok=[ok["left"], ok["right"]], ik_max_mm=[max(err["left"]), max(err["right"])], min_clear=float(min(clr)), min_table=float(min(tab)),
                     flags=flags, ms=(time.monotonic() - t0) * 1e3)
            self.res = r; self.log.append(r)
            time.sleep(max(0.0, 1.0 / self.a.preview_hz - (time.monotonic() - t0)))
