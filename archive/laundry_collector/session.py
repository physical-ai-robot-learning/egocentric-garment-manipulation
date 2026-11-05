"""Collector session: owns the cameras and runs the recording protocol (spec §6) as a Qt-free state machine.

  IDLE ──Start──▶ INIT (recording; hands on X_L / X_R, still for init_hold_s) ──auto──▶ MANIPULATION ──Stop──▶
  END (hands off, still for end_hold_s) ──auto──▶ REVIEW ──Save──▶ IDLE (kept, .complete)
                                                          └─Cancel──▶ IDLE (moved to _discarded/)
  Cancel during INIT / MANIPULATION / END stops and discards at once.

Every phase boundary goes into events.jsonl and metadata.phases with the host clock, so preprocessing can cut the
initialization window used for the world alignment. tick() must be called periodically (UI timer or headless loop).
"""
from __future__ import annotations

import logging
import time
from enum import Enum

import cv2
import numpy as np

from .config import AppConfig
from .devices import CameraSource, make_source, now_ns
from .episode import EpisodeStore, base_metadata
from .qa import build_report, report_text
from .recorder import EpisodeRecorder

log = logging.getLogger("laundry.session")


class State(str, Enum):
    IDLE = "IDLE"; INIT = "INIT"; MANIPULATION = "MANIPULATION"; END = "END"; REVIEW = "REVIEW"


PROMPT = {State.IDLE: "Place both hands at initialization markers, then Start.",
          State.INIT: "HOLD STILL on X_L / X_R",
          State.MANIPULATION: "Fold the T-shirt (thumb + index pinch). Stop when done.",
          State.END: "Hands off the cloth - hold still",
          State.REVIEW: "Review: set success / notes, then Save or Cancel."}

MOTION_STILL = 6.0       # mean |frame diff| (0..255, 160 px wide) above which a wrist counts as moving during INIT / END


class CollectorSession:
    def __init__(self, cfg: AppConfig, *, mock_drop_every: int = 0):
        self.cfg = cfg
        self.store = EpisodeStore(cfg)
        self.sources: dict[str, CameraSource] = {}
        self.state = State.IDLE
        self.rec: EpisodeRecorder | None = None
        self.ep = None; self.eid = None
        self.t: dict = {}
        self.phase_t0 = 0
        self.last_report: dict | None = None
        self.last_report_txt = ""
        self.summary = None
        self.motion: dict[str, list[float]] = {}
        self._prev_small: dict[str, np.ndarray] = {}
        self.message = ""
        self._mock_drop_every = mock_drop_every

    # ------------------------------------------------------------------ cameras
    def open(self) -> None:
        order = sorted(self.cfg.hardware.cameras, key=lambda c: c.backend != "orbbec")   # Orbbec first (USB bring-up order)
        for c in order:
            kw = dict(drop_every=self._mock_drop_every) if c.backend == "mock" else {}
            s = make_source(c, self.cfg.collector.buffer_s, **kw)
            try:
                s.open(); s.start()
            except Exception as exc:  # noqa: BLE001
                s.status.error = f"{type(exc).__name__}: {exc}"
                if c.required:
                    self.close()
                    raise RuntimeError(f"camera {c.name} failed to open: {exc}") from exc
            self.sources[c.name] = s

    def close(self) -> None:
        for s in self.sources.values():
            s.stop(); s.close()

    def devices(self) -> dict:
        return {n: dict(s.info, fps=s.cfg.fps, name=n) for n, s in self.sources.items()}

    def all_live(self) -> tuple[bool, list[str]]:
        bad = [n for n, s in self.sources.items() if not s.snapshot_status().connected]
        return not bad, bad

    # ------------------------------------------------------------------ protocol
    def _phase(self, st: State) -> None:
        now = now_ns()
        if self.t.get("phases") and self.t["phases"][-1]["end_host_ns"] is None:
            self.t["phases"][-1]["end_host_ns"] = now
        if st in (State.INIT, State.MANIPULATION, State.END):
            self.t["phases"].append(dict(phase={State.INIT: "initialization", State.MANIPULATION: "manipulation", State.END: "end"}[st],
                                         start_host_ns=now, end_host_ns=None))
            self.rec.event("phase_start", phase=self.t["phases"][-1]["phase"])
        self.state, self.phase_t0 = st, now

    def start(self) -> bool:
        if self.state != State.IDLE:
            return False
        ok, bad = self.all_live()
        if not ok:
            self.message = f"refused: camera not live: {bad}"; return False
        self.eid, self.ep = self.store.create(self.devices())
        self.rec = EpisodeRecorder(self.ep, self.sources, depth_writers=self.cfg.collector.depth_writers,
                                   sample_every=self.cfg.collector.qa_sample_every)
        self.rec.start()
        self.t = dict(start_ns=self.rec.t_start, start_wall=time.time(), phases=[])
        self.motion = {"initialization": [], "end": []}; self._prev_small = {}
        self._phase(State.INIT)
        self.message = f"episode {self.eid:06d} recording"
        return True

    def stop(self) -> bool:
        if self.state != State.MANIPULATION:
            return False
        if (now_ns() - self.phase_t0) / 1e9 < self.cfg.collector.protocol.min_manip_s:
            self.message = "manipulation too short -- Stop ignored"; return False
        self._phase(State.END)
        return True

    def _finish_recording(self) -> None:
        self._phase(State.REVIEW)
        self.summary = self.rec.stop()
        self.t.update(end_ns=self.rec.t_stop, end_wall=time.time())
        fps = {c.name: c.fps for c in self.cfg.hardware.cameras}
        self.last_report = build_report(self.ep, self.summary, fps)
        self.last_report["still_check"] = {k: dict(max_motion=round(max(v), 2) if v else None, threshold=MOTION_STILL) for k, v in self.motion.items()}
        for k, v in self.motion.items():
            if v and max(v) > MOTION_STILL:
                self.last_report["flags"].append(f"wrists moved during the {k} hold (motion {max(v):.1f} > {MOTION_STILL})")
                self.last_report["verdict"] = "CHECK"
        self.last_report_txt = report_text(self.last_report)

    def cancel(self) -> bool:
        if self.state in (State.INIT, State.MANIPULATION, State.END):
            self.t.setdefault("phases", []).append(dict(phase="cancelled", start_host_ns=now_ns(), end_host_ns=now_ns()))
            self._finish_recording()
        if self.state != State.REVIEW:
            return False
        return self._close_episode(keep=False, form={})

    def save(self, form: dict) -> bool:
        if self.state != State.REVIEW:
            return False
        if form.get("success") is None:
            self.message = "set Success / Failure before Save"; return False
        return self._close_episode(keep=True, form=form)

    def _close_episode(self, *, keep: bool, form: dict) -> bool:
        md = base_metadata(self.cfg, self.eid, form, self.t)
        md["qa_verdict"] = (self.last_report or {}).get("verdict")
        dst = self.store.finalize(self.ep, keep=keep, metadata=md, report=self.last_report, report_txt=self.last_report_txt,
                                  recorder_summary=self.summary)
        self.message = f"episode {self.eid:06d} {'saved' if keep else 'discarded'} -> {dst}"
        log.info(self.message)
        self.state, self.rec = State.IDLE, None
        return True

    def _wrist_motion(self) -> float:
        m = 0.0
        for n in ("left_wrist", "right_wrist"):
            f = self.sources[n].latest() if n in self.sources else None
            if f is None:
                continue
            small = cv2.cvtColor(cv2.resize(f.image, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if n in self._prev_small:
                m = max(m, float(np.abs(small - self._prev_small[n]).mean()))
            self._prev_small[n] = small
        return m

    def tick(self) -> None:
        el = (now_ns() - self.phase_t0) / 1e9
        p = self.cfg.collector.protocol
        if self.state == State.INIT:
            if el > 0.3:
                self.motion["initialization"].append(self._wrist_motion())
            if el >= p.init_hold_s:
                self._phase(State.MANIPULATION)
        elif self.state == State.END:
            if el > 0.3:
                self.motion["end"].append(self._wrist_motion())
            if el >= p.end_hold_s:
                self._finish_recording()

    def countdown(self) -> float:
        """Seconds left in the INIT / END hold; elapsed seconds in any other state."""
        el = (now_ns() - self.phase_t0) / 1e9
        p = self.cfg.collector.protocol
        if self.state == State.INIT:
            return p.init_hold_s - el
        if self.state == State.END:
            return p.end_hold_s - el
        return el
