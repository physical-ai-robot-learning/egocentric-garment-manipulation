"""Episode recorder: drains every camera source on one pump thread and writes, per stream,

  <stream>/rgb/rgb.mp4          H.264 (VideoStreamWriter, own encoder thread, never drops)
  <stream>/timestamps.csv       video_frame, frame_index, camera_timestamp_ns, host_timestamp_ns[, depth_index, depth_timestamp_ns, depth_file]
  global/depth/NNNNNN.png       uint16 depth aligned to global color (written by a thread pool, not on the pump thread)

plus phase / device events (events.jsonl, host clock) and per-frame quality samples for the QA report.
Pre-roll is never recorded: buffers are drained at START.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from handumi_collector.collector.video_writer import VideoStreamWriter

from .devices import CameraSource, Frame, now_ns


def image_quality(img: np.ndarray) -> dict:
    g = cv2.cvtColor(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return dict(sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()), brightness=float(g.mean()),
                saturated=float((g >= 250).mean()), dark=float((g <= 5).mean()))


class StreamWriter:
    def __init__(self, ep: Path, src: CameraSource, pool: ThreadPoolExecutor | None):
        c = src.cfg
        self.name, self.depth = c.name, bool(c.depth)
        d = ep / c.name; (d / "rgb").mkdir(parents=True, exist_ok=True)
        self.video = VideoStreamWriter(d / "rgb" / "rgb.mp4", width=c.width, height=c.height, fps=c.fps, codec=c.codec,
                                       bitrate_kbps=c.bitrate_kbps)
        self._f = open(d / "timestamps.csv", "w", newline="")
        self.csv = csv.writer(self._f)
        cols = ["video_frame", "frame_index", "camera_timestamp_ns", "host_timestamp_ns"]
        if self.depth:
            (d / "depth").mkdir(exist_ok=True); self.depth_dir = d / "depth"
            cols += ["depth_index", "depth_timestamp_ns", "depth_file"]
        self.csv.writerow(cols)
        self.pool = pool
        self.depth_futures = []
        self.n = 0
        self.first_host = self.last_host = None
        self.quality: list[dict] = []
        self.depth_valid: list[float] = []

    def put(self, f: Frame, sample_every: int) -> None:
        vf = self.video.put(f.image)
        row = [vf, f.frame_index, f.camera_ts_ns if f.camera_ts_ns is not None else "", f.host_ts_ns]
        if self.depth:
            name = ""
            if f.depth is not None:
                name = f"{vf:06d}.png"
                self.depth_futures.append(self.pool.submit(cv2.imwrite, str(self.depth_dir / name), f.depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
            row += [f.depth_index if f.depth_index is not None else "", f.depth_ts_ns if f.depth_ts_ns is not None else "", name]
        self.csv.writerow(row)
        if self.n % sample_every == 0:
            self.quality.append(dict(video_frame=vf, **image_quality(f.image)))
            if self.depth and f.depth is not None:
                self.depth_valid.append(float((f.depth > 0).mean()))
        self.first_host = self.first_host or f.host_ts_ns; self.last_host = f.host_ts_ns
        self.n += 1

    def close(self) -> dict:
        self._f.close()
        self.video.close()
        failed = sum(1 for fu in self.depth_futures if fu.result() is not True)
        return dict(frames=self.n, video_frames=self.video.frames_written, codec=self.video.codec_used, encoder_error=self.video.error,
                    depth_png_failed=failed if self.depth else None)


class EpisodeRecorder:
    def __init__(self, ep_dir: Path, sources: dict[str, CameraSource], *, depth_writers: int = 3, sample_every: int = 15):
        self.ep = Path(ep_dir)
        self.sources = sources
        self.sample_every = sample_every
        self.pool = ThreadPoolExecutor(depth_writers, thread_name_prefix="depth-png")
        self.events_f = open(self.ep / "events.jsonl", "w")
        self.writers: dict[str, StreamWriter] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.overflow0 = {}
        self.t_start = self.t_stop = None

    def event(self, kind: str, **kw) -> None:
        self.events_f.write(json.dumps(dict(host_ts_ns=now_ns(), wall_time=time.time(), event=kind, **kw)) + "\n"); self.events_f.flush()

    def start(self) -> None:
        for name, s in self.sources.items():
            s.drain()                                    # discard pre-roll
            self.overflow0[name] = s.overflow()
            self.writers[name] = StreamWriter(self.ep, s, self.pool)
        self.t_start = now_ns()
        self.event("recording_start")
        self._thread = threading.Thread(target=self._pump, name="episode-pump", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            self._drain_once()
            time.sleep(0.02)
        self._drain_once()

    def _drain_once(self) -> None:
        for name, s in self.sources.items():
            for f in s.drain():
                self.writers[name].put(f, self.sample_every)

    def stop(self) -> dict:
        self.t_stop = now_ns()
        self._stop.set()
        if self._thread:
            self._thread.join(10)
        self.event("recording_stop")
        out = {}
        for name, w in self.writers.items():
            out[name] = w.close()
            out[name]["buffer_overflow"] = self.sources[name].overflow() - self.overflow0[name]
            out[name]["quality_samples"] = w.quality
            out[name]["depth_valid_samples"] = w.depth_valid
        self.pool.shutdown(wait=True)
        self.events_f.close()
        return out
