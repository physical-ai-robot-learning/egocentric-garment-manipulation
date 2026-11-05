"""Camera sources for the laundry collector. One capture thread per camera; every frame carries THREE time/index keys:

  frame_index        the camera's own frame counter when it has one (Orbbec color frame number), else a host counter
  camera_ts_ns       device timestamp (Orbbec color frame timestamp; AVFoundation presentation time via CAP_PROP_POS_MSEC
                     when the backend reports one), None when the device gives nothing
  host_ts_ns         time.monotonic_ns() right after the grab returned (the collector's common clock)

The old handumi capture numbered frames with a host counter only, so dropped frames were invisible. Here a drop is counted
from the device frame number when there is one, and from the device / host frame interval otherwise (> 1.5 / fps).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CameraCfg

log = logging.getLogger("laundry.devices")
STALL_S = 1.5


def now_ns() -> int:
    return time.monotonic_ns()


@dataclass
class Frame:
    stream: str
    frame_index: int
    camera_ts_ns: int | None
    host_ts_ns: int
    image: np.ndarray                    # BGR uint8
    depth: np.ndarray | None = None      # uint16, aligned to color (global only)
    depth_ts_ns: int | None = None       # depth frame device timestamp (global only)
    depth_index: int | None = None


@dataclass
class CamStatus:
    name: str
    connected: bool = False
    fps: float = 0.0
    age_ms: float | None = None
    frames: int = 0
    drops: int = 0                        # device-counter gaps + interval gaps since open
    read_failures: int = 0
    error: str | None = None
    detail: dict = field(default_factory=dict)


class DropCounter:
    """Counts missing frames from the device frame number (exact) or from the frame interval (> 1.5 nominal periods)."""

    def __init__(self, fps: float) -> None:
        self.period_ns = 1e9 / fps
        self.last_idx: int | None = None
        self.last_ts: int | None = None
        self.drops = 0

    def update(self, idx: int | None, ts_ns: int, *, device_index: bool) -> int:
        lost = 0
        if device_index and idx is not None and self.last_idx is not None:
            lost = max(0, idx - self.last_idx - 1)
        elif self.last_ts is not None:
            gap = ts_ns - self.last_ts
            if gap > 1.5 * self.period_ns:
                lost = int(round(gap / self.period_ns)) - 1
        self.last_idx, self.last_ts = idx, ts_ns
        self.drops += lost
        return lost


class CameraSource:
    """Base: a daemon thread calls _grab() and publishes frames into a bounded deque the recorder drains."""

    device_index = False   # True when frame_index is the device's own counter

    def __init__(self, cfg: CameraCfg, buffer_s: float = 4.0) -> None:
        self.cfg = cfg
        self.status = CamStatus(cfg.name)
        self._buf: deque[Frame] = deque(maxlen=max(8, int(buffer_s * cfg.fps)))
        self._overflow = 0
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rate: deque[int] = deque(maxlen=60)
        self._drops = DropCounter(cfg.fps)
        self.info: dict = {}                # identity / intrinsics / settings, saved into the episode

    # subclasses: open(), close(), _grab() -> Frame | None
    def open(self) -> None: ...
    def close(self) -> None: ...
    def _grab(self) -> Frame | None: raise NotImplementedError

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cfg.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                f = self._grab()
            except Exception as exc:  # noqa: BLE001 -- a camera error must surface in the UI, not kill the thread
                self.status.error = f"{type(exc).__name__}: {exc}"; time.sleep(0.05); continue
            if f is None:
                self.status.read_failures += 1; time.sleep(0.002); continue
            ts = f.camera_ts_ns if f.camera_ts_ns is not None else f.host_ts_ns
            self._drops.update(f.frame_index, ts, device_index=self.device_index)
            with self._lock:
                if len(self._buf) == self._buf.maxlen:
                    self._overflow += 1
                self._buf.append(f); self._latest = f
            self._rate.append(f.host_ts_ns)
            self.status.frames += 1

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def drain(self) -> list[Frame]:
        with self._lock:
            out = list(self._buf); self._buf.clear()
        return out

    def overflow(self) -> int:
        return self._overflow

    def snapshot_status(self) -> CamStatus:
        s = self.status
        if len(self._rate) >= 2:
            s.fps = (len(self._rate) - 1) / ((self._rate[-1] - self._rate[0]) / 1e9)
        s.age_ms = (now_ns() - self._latest.host_ts_ns) / 1e6 if self._latest else None
        s.drops = self._drops.drops
        s.connected = self._latest is not None and (s.age_ms or 0) < STALL_S * 1000
        return s


# ----------------------------------------------------------------------------------------------------- Orbbec RGB-D
class OrbbecRgbdSource(CameraSource):
    """Fixed global RGB-D. Color MJPG + depth, depth hardware-aligned to color (AlignFilter), device frame numbers and
    timestamps kept for both. Mode taken from the config (the old OrbbecCapture hard-coded 848x480@30)."""

    device_index = True

    def open(self) -> None:
        from pyorbbecsdk import AlignFilter, Config, OBFormat, OBSensorType, OBStreamType, Pipeline
        c = self.cfg
        pipe = Pipeline(); oc = Config()
        dw, dh = c.depth_width or c.width, c.depth_height or c.height
        dprof = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(dw, dh, OBFormat.UNKNOWN_FORMAT, c.fps)
        cprof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(c.width, c.height, OBFormat.MJPG, c.fps)
        oc.enable_stream(dprof); oc.enable_stream(cprof)
        try:
            oc.set_frame_aggregate_output_mode(__import__("pyorbbecsdk").OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:  # noqa: BLE001 -- older SDKs: framesets may then lack depth; _grab() treats that as a failure
            pass
        pipe.start(oc)
        self._pipe, self._align = pipe, AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        intr = cprof.get_intrinsic()
        try:
            dist = cprof.get_distortion(); dist = dict(k1=dist.k1, k2=dist.k2, k3=dist.k3, k4=dist.k4, k5=dist.k5, k6=dist.k6, p1=dist.p1, p2=dist.p2)
        except Exception:  # noqa: BLE001
            dist = None
        dev = pipe.get_device().get_device_info()
        self.info = dict(backend="orbbec", model=dev.get_name(), serial=dev.get_serial_number(),
                         firmware=getattr(dev, "get_firmware_version", lambda: None)(),
                         color=dict(width=c.width, height=c.height, fps=c.fps, format="MJPG"),
                         depth=dict(width=dw, height=dh, fps=c.fps, aligned_to="color"),
                         intrinsics=dict(model="pinhole", fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, width=intr.width, height=intr.height,
                                         distortion=dist))
        if c.serial and c.strict_identity and self.info["serial"] != c.serial:
            raise RuntimeError(f"global: Orbbec serial {self.info['serial']} != expected {c.serial}")
        self.depth_scale = None
        for _ in range(40):                      # warm-up: first framesets are often incomplete
            if self._grab() is not None:
                break
            time.sleep(0.05)
        log.info("orbbec open %s", self.info)

    def close(self) -> None:
        try:
            self._pipe.stop()
        except Exception:  # noqa: BLE001
            pass

    def _grab(self) -> Frame | None:
        fs = self._pipe.wait_for_frames(500)
        host = now_ns()
        if fs is None:
            return None
        al = self._align.process(fs)
        al = al.as_frame_set() if al is not None and hasattr(al, "as_frame_set") else al
        if al is None:
            return None
        cf, df = al.get_color_frame(), al.get_depth_frame()
        if cf is None or df is None:
            return None
        img = cv2.imdecode(np.frombuffer(cf.get_data(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        depth = np.frombuffer(df.get_data(), np.uint16).reshape(df.get_height(), df.get_width()).copy()
        if self.depth_scale is None:
            self.depth_scale = float(df.get_depth_scale()); self.info["depth_scale_mm_per_unit"] = self.depth_scale
        return Frame(self.cfg.name, int(cf.get_index()), int(cf.get_timestamp_us()) * 1000, host, img,
                     depth=depth, depth_ts_ns=int(df.get_timestamp_us()) * 1000, depth_index=int(df.get_index()))


# ----------------------------------------------------------------------------------------------------- UVC (wrists)
def resolve_uvc(cfg: CameraCfg) -> tuple[int, dict]:
    """Product name -> the OpenCV index that camera REALLY opens on.

    The ffmpeg/AVFoundation listing order is NOT OpenCV's index order (2026-10-08: the listing said index 1 = the Arducam,
    OpenCV index 1 was the MacBook camera, and a bring-up take recorded the operator's face). So the index is MEASURED:
    handumi probe_index_map darkens the camera by name through uvc-util and watches which index goes dark. An explicit
    cfg.index is still proven after open (UvcSource.open). Unproven + strict_identity -> refuse.
    """
    from handumi_collector.config import CameraCfg as HCfg
    from handumi_collector.devices.camera import list_video_devices, probe_index_map, verify_identity
    listing = list_video_devices()
    h = HCfg(name=cfg.name, role="policy_obs", backend="uvc", match_name=cfg.match_name, ordinal=cfg.ordinal, index=cfg.index,
             serial=cfg.serial, strict_identity=False)
    ident = verify_identity(h, listing) if cfg.match_name else {}
    hits = [n for n in listing if cfg.match_name and cfg.match_name.lower() in n.lower()]
    if cfg.strict_identity and len(hits) != 1:
        raise RuntimeError(f"{cfg.name}: {len(hits)} devices named like {cfg.match_name!r} in {listing} -- need exactly one")
    if cfg.index is not None:
        return int(cfg.index), dict(ident, listing=listing, index_source="config")
    m = probe_index_map([hits[0]]) if hits else {}
    if hits and hits[0] in m:
        return int(m[hits[0]]), dict(ident, listing=listing, index_source="probe_index_map (uvc-util darkening)")
    raise RuntimeError(f"{cfg.name}: could not prove which OpenCV index is {cfg.match_name!r} (probe map {m}) -- refusing")


class UvcSource(CameraSource):
    def open(self) -> None:
        c = self.cfg
        idx, ident = resolve_uvc(c)
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height); cap.set(cv2.CAP_PROP_FPS, c.fps)
        if not cap.isOpened():
            raise RuntimeError(f"{c.name}: cannot open OpenCV index {idx}")
        ok, img = cap.read()
        if not ok or img is None:
            raise RuntimeError(f"{c.name}: no frame from index {idx}")
        if (img.shape[1], img.shape[0]) != (c.width, c.height):
            cap.release()
            raise RuntimeError(f"{c.name}: camera delivers {img.shape[1]}x{img.shape[0]}, profile wants {c.width}x{c.height}")
        self._cap, self._n = cap, 0
        # second, independent proof on the stream actually opened at full resolution
        from handumi_collector.devices.camera import prove_binding
        def sample():
            vals = []
            for _ in range(4):
                ok, im = cap.read()
                if ok and im is not None:
                    vals.append(float(cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY).mean()))
            return sum(vals) / len(vals) if vals else None
        proof = prove_binding(ident["listing_name"], sample) if ident.get("listing_name") else dict(proved=None, reason="no name")
        # 2026-10-08: with skin filling the frame the picture got BRIGHTER (60 -> 153) when the named camera's brightness was swung.
        # Any large response to a control sent BY NAME proves the binding; another camera's stream would not move at all.
        b, d = proof.get("baseline"), proof.get("darkened")
        if proof.get("proved") is False and b and d and abs(d - b) / max(b, d) > 0.15:
            proof = dict(proof, proved=True, reason=f"responded {b:.0f}->{d:.0f} (direction reversed, scene/AE dependent)")
        if proof.get("proved") is False or (proof.get("proved") is None and c.strict_identity):
            cap.release()
            raise RuntimeError(f"{c.name}: identity not proven on index {idx}: {proof}")
        # the proof swings brightness / exposure mode and restores them; auto-exposure needs ~1 s to settle again
        # (bring-up 2026-10-08: brightness 175 -> 105 over the first second of the take). Wait it out before going live.
        t_end = time.time() + 2.0
        while time.time() < t_end:
            cap.grab()
        self.info = dict(backend="uvc", opencv_index=idx, identity=ident, binding_proof=proof, width=c.width, height=c.height,
                         fps=c.fps, fourcc=c.fourcc)

    def close(self) -> None:
        self._cap.release()

    def _grab(self) -> Frame | None:
        if not self._cap.grab():
            return None
        host = now_ns()
        ok, img = self._cap.retrieve()
        if not ok or img is None:
            return None
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)            # AVFoundation presentation time when available
        cam = int(ms * 1e6) if ms and ms > 0 and math.isfinite(ms) else None
        f = Frame(self.cfg.name, self._n, cam, host, img); self._n += 1
        return f


# ----------------------------------------------------------------------------------------------------- mock
class MockSource(CameraSource):
    """Synthetic camera at the configured fps: moving square, frame number burnt in, synthetic depth for the global stream.
    `drop_every` > 0 skips that many device frame numbers periodically (exercises drop detection)."""

    device_index = True

    def __init__(self, cfg: CameraCfg, buffer_s: float = 4.0, drop_every: int = 0) -> None:
        super().__init__(cfg, buffer_s)
        self.drop_every = drop_every

    def open(self) -> None:
        self._n = 0; self._t0 = now_ns(); self._next = self._t0
        self.info = dict(backend="mock", width=self.cfg.width, height=self.cfg.height, fps=self.cfg.fps,
                         intrinsics=dict(model="pinhole", fx=self.cfg.width * 0.8, fy=self.cfg.width * 0.8, cx=self.cfg.width / 2,
                                         cy=self.cfg.height / 2, width=self.cfg.width, height=self.cfg.height, distortion=None))
        if self.cfg.depth:
            self.info["depth_scale_mm_per_unit"] = 1.0

    def _grab(self) -> Frame | None:
        period = int(1e9 / self.cfg.fps)
        self._next += period
        d = self._next - now_ns()
        if d > 0:
            time.sleep(d / 1e9)
        n = self._n; self._n += 1
        if self.drop_every and n and n % self.drop_every == 0:
            self._n += 1                                     # one device frame lost
        w, h = self.cfg.width, self.cfg.height
        img = np.full((h, w, 3), 40, np.uint8)
        x = int((n * 3) % max(1, w - 40)); cv2.rectangle(img, (x, h // 3), (x + 40, h // 3 + 40), (60, 200, 60), -1)
        cv2.putText(img, f"{self.cfg.name} {n}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        depth = None
        if self.cfg.depth:
            depth = np.full((h, w), 900, np.uint16); depth[: h // 10] = 0          # 10 % invalid rows
        ts = self._t0 + n * period
        return Frame(self.cfg.name, n, ts, now_ns(), img, depth=depth, depth_ts_ns=ts if depth is not None else None,
                     depth_index=n if depth is not None else None)


def make_source(cfg: CameraCfg, buffer_s: float, **kw) -> CameraSource:
    if cfg.backend == "mock":
        return MockSource(cfg, buffer_s, **kw)
    return {"orbbec": OrbbecRgbdSource, "uvc": UvcSource}[cfg.backend](cfg, buffer_s)
