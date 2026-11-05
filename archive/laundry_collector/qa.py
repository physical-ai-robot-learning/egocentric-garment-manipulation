"""Per-episode synchronization + data-quality report (spec §9, §10), computed from what is ON DISK (timestamps.csv, mp4,
depth PNGs), not from the recorder's in-memory counters, so it can be re-run on any raw episode:

  python -m laundry_collector.qa <episode_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STREAMS

# engineering thresholds for the pilot (revisit after the 10-episode pilot)
TH = dict(fps_rel_tol=0.05, max_drop_frac=0.01, max_skew_ms=40.0, min_depth_valid=0.6, dark_mean=35, bright_mean=220,
          max_saturated=0.05, min_sharpness=40.0)


def _video_frames(mp4: Path) -> int:
    import av
    with av.open(str(mp4)) as c:
        return sum(1 for _ in c.decode(video=0))


def stream_stats(df: pd.DataFrame, fps_nominal: float) -> dict:
    host = df.host_timestamp_ns.to_numpy(np.int64)
    cam = pd.to_numeric(df.camera_timestamp_ns, errors="coerce").to_numpy()
    use_cam = np.isfinite(cam).all() and len(cam) > 1
    t = cam if use_cam else host.astype(float)
    dt = np.diff(t) / 1e6 if len(t) > 1 else np.array([])
    period = 1000.0 / fps_nominal
    idx = df.frame_index.to_numpy(np.int64)
    idx_gaps = int(np.clip(np.diff(idx) - 1, 0, None).sum()) if len(idx) > 1 else 0
    interval_gaps = int(np.clip(np.round(dt / period) - 1, 0, None)[dt > 1.5 * period].sum()) if len(dt) else 0
    dur = (host[-1] - host[0]) / 1e9 if len(host) > 1 else 0.0
    return dict(frames=int(len(df)), duration_s=round(dur, 3), fps_measured=round((len(df) - 1) / dur, 2) if dur else None,
                fps_nominal=fps_nominal, clock="camera" if use_cam else "host",
                dropped_by_index=idx_gaps, dropped_by_interval=interval_gaps,
                interval_ms=dict(p50=round(float(np.median(dt)), 2), p95=round(float(np.percentile(dt, 95)), 2),
                                 max=round(float(dt.max()), 2)) if len(dt) else None,
                host_minus_camera_ms=dict(p50=round(float(np.median(host - cam) / 1e6), 2),
                                          spread=round(float(np.ptp(host - cam) / 1e6), 2)) if use_cam else None)


def skew_ms(ref_host: np.ndarray, other_host: np.ndarray) -> dict:
    """For every reference frame, |host time - nearest other-stream frame|: what nearest-neighbour sync would give."""
    j = np.clip(np.searchsorted(other_host, ref_host), 1, len(other_host) - 1)
    d = np.minimum(np.abs(other_host[j] - ref_host), np.abs(other_host[j - 1] - ref_host)) / 1e6
    return dict(p50=round(float(np.median(d)), 2), p95=round(float(np.percentile(d, 95)), 2), max=round(float(d.max()), 2))


def build_report(ep: Path, recorder_summary: dict | None = None, fps: dict | None = None) -> dict:
    ep = Path(ep)
    devices = json.loads((ep / "calibration" / "devices.json").read_text()) if (ep / "calibration" / "devices.json").exists() else {}
    rep: dict = dict(episode=ep.name, streams={}, skew_ms={}, flags=[])
    dfs = {}
    for s in STREAMS:
        f = ep / s / "timestamps.csv"
        if not f.exists():
            rep["flags"].append(f"{s}: no timestamps.csv"); continue
        df = pd.read_csv(f); dfs[s] = df
        nominal = float((fps or {}).get(s) or devices.get(s, {}).get("fps") or 30)
        st = stream_stats(df, nominal) if len(df) else dict(frames=0)
        mp4 = ep / s / "rgb" / "rgb.mp4"
        st["mp4_frames"] = _video_frames(mp4) if mp4.exists() else 0
        if st["mp4_frames"] != st["frames"]:
            rep["flags"].append(f"{s}: mp4 has {st['mp4_frames']} frames, timestamps.csv {st['frames']}")
        if s == "global" and "depth_file" in df:
            names = df.depth_file.dropna().astype(str)
            missing = int(sum(1 for n in names if n and not (ep / s / "depth" / n).exists()))
            st["depth_frames"] = int((names != "").sum()); st["depth_missing_files"] = missing + int((names == "").sum())
            if st["depth_missing_files"]:
                rep["flags"].append(f"global: {st['depth_missing_files']} frames without a depth PNG")
        q = (recorder_summary or {}).get(s, {})
        if q.get("quality_samples"):
            qs = pd.DataFrame(q["quality_samples"])
            st["sharpness_mean"] = round(float(qs.sharpness.mean()), 1)
            st["brightness_mean"] = round(float(qs.brightness.mean()), 1)
            st["saturated_mean"] = round(float(qs.saturated.mean()), 4)
            st["blurry_frac"] = round(float((qs.sharpness < TH["min_sharpness"]).mean()), 3)
        if q.get("depth_valid_samples"):
            dv = np.asarray(q["depth_valid_samples"])
            st["depth_valid_ratio"] = dict(mean=round(float(dv.mean()), 3), min=round(float(dv.min()), 3))
            st["depth_missing_ratio"] = round(1 - float(dv.mean()), 3)
        for k in ("buffer_overflow", "encoder_error", "depth_png_failed"):
            if q.get(k):
                rep["flags"].append(f"{s}: {k} = {q[k]}")
        rep["streams"][s] = st
        # flags
        if st.get("frames", 0) and st.get("fps_measured") and abs(st["fps_measured"] / nominal - 1) > TH["fps_rel_tol"]:
            rep["flags"].append(f"{s}: fps {st['fps_measured']} vs nominal {nominal}")
        drops = max(st.get("dropped_by_index", 0), st.get("dropped_by_interval", 0))
        if st.get("frames") and drops / (st["frames"] + drops) > TH["max_drop_frac"]:
            rep["flags"].append(f"{s}: {drops} dropped frames")
        if st.get("brightness_mean") is not None and not (TH["dark_mean"] <= st["brightness_mean"] <= TH["bright_mean"]):
            rep["flags"].append(f"{s}: exposure, mean brightness {st['brightness_mean']}")
        if st.get("saturated_mean", 0) > TH["max_saturated"]:
            rep["flags"].append(f"{s}: {st['saturated_mean']:.1%} saturated pixels")
        if st.get("blurry_frac", 0) > 0.2:
            rep["flags"].append(f"{s}: {st['blurry_frac']:.0%} of sampled frames blurry")
        if st.get("depth_valid_ratio") and st["depth_valid_ratio"]["mean"] < TH["min_depth_valid"]:
            rep["flags"].append(f"global: depth valid {st['depth_valid_ratio']['mean']:.0%}")
    if "global" in dfs:
        g = dfs["global"].host_timestamp_ns.to_numpy(np.int64)
        for s in ("left_wrist", "right_wrist"):
            if s in dfs and len(dfs[s]) > 1 and len(g):
                rep["skew_ms"][f"global-{s.split('_')[0]}"] = sk = skew_ms(g, dfs[s].host_timestamp_ns.to_numpy(np.int64))
                if sk["max"] > TH["max_skew_ms"]:
                    rep["flags"].append(f"global-{s}: max skew {sk['max']} ms")
    rep["verdict"] = "PASS" if not rep["flags"] else "CHECK"
    return rep


def report_text(rep: dict) -> str:
    L = [f"episode {rep['episode']}  verdict {rep['verdict']}", ""]
    for s, st in rep["streams"].items():
        L.append(f"{s:12s} fps {st.get('fps_measured')} (nominal {st.get('fps_nominal')}, {st.get('clock')} clock)  frames {st.get('frames')}  "
                 f"dropped {max(st.get('dropped_by_index', 0), st.get('dropped_by_interval', 0))}"
                 + (f"  sharpness {st['sharpness_mean']}" if "sharpness_mean" in st else "")
                 + (f"  depth valid {st['depth_valid_ratio']['mean']:.1%}" if "depth_valid_ratio" in st else ""))
    L.append("")
    for k, v in rep["skew_ms"].items():
        L.append(f"max timestamp skew {k}: {v['max']} ms (p95 {v['p95']})")
    L += ["", "flags:"] + [f"  - {f}" for f in rep["flags"]] if rep["flags"] else ["", "flags: none"]
    return "\n".join(L)


if __name__ == "__main__":
    ep = Path(sys.argv[1])
    summ = json.loads((ep / "recorder_summary.json").read_text()) if (ep / "recorder_summary.json").exists() else None
    r = build_report(ep, summ)
    print(report_text(r))
