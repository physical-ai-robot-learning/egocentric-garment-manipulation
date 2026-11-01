"""Robot bridge side: the gripper-opening guard for --go, the 25 Hz /observe poller, the 30 Hz command streamer on an indexed
joint timeline (merge / set), and the explicit return to READY.
"""
from __future__ import annotations
import threading
import time

import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.cameras import Ring
from garment_pipeline.eval_deploy.tagrig_deploy.paths import say


GRIP_OPEN_FRAC_MAX = {"left": 0.5, "right": 1.0}   # 10-27: the 10-23 bridge ran MIT_GRIP_OPEN_FRAC_LEFT=0.5; restarted without it -> left jaw over-opened, holder broke


def check_grip_limits(url):
    """--go refuses unless the bridge limits each jaw's opening to at most GRIP_OPEN_FRAC_MAX (bridge /status grip_open_frac)"""
    import requests
    try: got = requests.get(f"{url.rstrip('/')}/status", timeout=2).json().get("grip_open_frac") or {}
    except Exception as ex: raise SystemExit(f"bridge /status 읽기 실패 ({ex}) -- 그리퍼 열림 제한을 확인할 수 없어 GO 거부")
    bad = {s: got.get(s) for s, mx in GRIP_OPEN_FRAC_MAX.items() if got.get(s) is None or float(got[s]) > mx + 1e-6}
    if bad: raise SystemExit(f"그리퍼 열림 제한 불일치 {bad} (허용 최대 {GRIP_OPEN_FRAC_MAX}) -- bridge 를 MIT_GRIP_OPEN_FRAC_LEFT=0.5 로 다시 띄우세요. GO 거부")


class RobotPoller(threading.Thread):
    """/observe at ~25 Hz -> ring of (q, g_obs, estop); --assume-ready: FK(READY) forever (dry-run without a live robot)"""
    def __init__(self, br, qr, assume_ready, hz=25.0):
        super().__init__(daemon=True); self.br, self.qr, self.fake, self.hz = br, qr, assume_ready, hz; self.ring = Ring(20); self.stop = False; self.error = None; self.lat = []

    def read(self):
        if self.fake: return {s: self.qr[s].copy() for s in core.SIDES}, {s: None for s in core.SIDES}, False
        o = self.br.observe(); self.lat.append(o.latency_ms); return o.q, {s: o.g(s) for s in core.SIDES}, o.estop

    def run(self):
        while not self.stop:
            # 10-30: a bridge restart left self.error = "/observe: not connected" forever (never cleared after reads recovered) and
            # every later UI run stopped at once with "robot read: ..." -> clear it on the next good read
            try: self.ring.put(time.monotonic_ns(), self.read()); self.error = None
            except Exception as e: self.error = f"{type(e).__name__}: {e}"; self.ring.put(time.monotonic_ns(), (None, None, True))
            time.sleep(1.0 / self.hz)


# ---------------------------------------------------------------- 30 Hz command streamer
class Streamer(threading.Thread):
    """sends q interpolated along an INDEXED timeline of joint targets (tick i = t_base + i * DT) at --stream-hz (only with --go).
    [10-29] indy7 c4fba14 (bh_indy7_LeRobot inference/action_timeline.merge_chunk): a chunk row is keyed by the tick it was predicted
    for; merge() drops rows for ticks already driven (no pulling back along the covered path), blends rows for ticks still pending
    (crossfade, new weight rising) and appends the rest.  set() = restart the timeline from where the stream is now (recovery, step)."""
    def __init__(self, br, q0, g0, hz, dt=None):
        super().__init__(daemon=True); self.br, self.hz = br, hz; self.g = dict(g0); self.lock = threading.Lock(); self.stop = False; self.error = None
        self.dt = float(dt or core.DT)             # tick period (look-ahead: the policy's row period / speed)
        self.n = 0; self.last = None; self.t_base = time.monotonic(); self.tl = {0: {s: q0[s].copy() for s in core.SIDES}}; self.n_dry = 0

    def tick(self, t=None):
        return ((time.monotonic() if t is None else t) - self.t_base) / self.dt

    def executed(self, t=None):
        return int(np.floor(self.tick(t)))

    @property
    def last_idx(self):
        return max(self.tl)

    def _q_at(self, x):
        import bisect
        ks = sorted(self.tl); x = max(x, ks[0])
        if x >= ks[-1]: return self.tl[ks[-1]]
        j = bisect.bisect_right(ks, x) - 1; i0_, i1_ = ks[j], ks[j + 1]; f = (x - i0_) / (i1_ - i0_)
        return {s: self.tl[i0_][s] + f * (self.tl[i1_][s] - self.tl[i0_][s]) for s in core.SIDES}

    def anchor(self, t):
        """the tick index at time t and the COMMANDED q at that tick (the pose a chunk predicted from an observation at t starts from)"""
        with self.lock:
            i = self.executed(t); return i, {s: v.copy() for s, v in self._q_at(i).items()}

    def merge(self, i0, rows, g):
        """rows[j] = q for tick i0 + 1 + j -> (rows dropped as already driven, rows blended)"""
        with self.lock:
            ex = self.executed() + 1                # the tick being interpolated toward right now is left untouched
            new = {i0 + 1 + j: r for j, r in enumerate(rows) if i0 + 1 + j > ex}
            ov = sorted(i for i in new if i in self.tl)
            for m, i in enumerate(ov):
                w = (m + 1) / (len(ov) + 1); self.tl[i] = {s: (1 - w) * self.tl[i][s] + w * new[i][s] for s in core.SIDES}
            for i in new:
                if i not in self.tl: self.tl[i] = new[i]
            for i in [i for i in self.tl if i < ex - 1]: del self.tl[i]
            self.g = dict(g); return len(rows) - len(new), len(ov)

    def set(self, waypts, dur_each, g):
        """waypts: list of {side: q} (the committed steps); each reached after one tick (self.dt), from where the stream is now"""
        with self.lock:
            now = time.monotonic(); q = self.last if self.last is not None else self._q_at(self.tick(now))
            self.t_base = now; self.tl = {0: {s: q[s].copy() for s in core.SIDES}}
            for i, w in enumerate(waypts): self.tl[i + 1] = w
            self.g = dict(g)

    def _q(self):
        with self.lock:
            x = self.tick()
            if x > self.last_idx: self.n_dry += 1   # timeline ran dry: hold the last target (counted at stream rate)
            return self._q_at(x), self.g

    def run(self):
        nxt = time.monotonic()
        while not self.stop:
            q, g = self._q(); self.last = {s: q[s].copy() for s in core.SIDES}
            r = self.br.send(q, g); self.n += 1
            if not r.accepted: self.error = f"bridge reject: {r.error}"; return
            nxt += 1.0 / self.hz; time.sleep(max(0.0, nxt - time.monotonic()))


def go_ready(br, arms, qr):
    """explicit user request only: /clear_estop, link-gap check along the joint line, umi_replay_bi.ramp_both at 10 deg/s"""
    from visual_teleop.tools.umi_replay_bi import ramp_both
    o = br.observe(); q = o.q
    for u in np.linspace(0, 1, 30):
        qq = {s: q[s] + u * (qr[s] - q[s]) for s in core.SIDES}
        if core.link_gap(arms, qq) < 0.10: print("READY 경로에서 두 팔이 가까움 (< 100 mm) -- 손으로 옮기세요. 이동 안 함", flush=True); return
    r = br.s.post(br.base_url + "/clear_estop", timeout=2.0).json()
    if not r.get("ok", True): print("clear_estop 실패:", r); return
    g = {s: o.g(s) for s in core.SIDES}; say("READY 로 천천히 복귀"); ramp_both(br, q, qr, g, deg_s=10.0); br.estop(); say("READY 도착, 홀드")
