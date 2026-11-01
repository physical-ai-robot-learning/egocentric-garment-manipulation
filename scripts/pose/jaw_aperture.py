"""Jaw aperture from jaw-tip keypoints (2026-10-22, replaces the binary-only jaw label for takes recorded after the camera re-angle).
Per wrist frame: YOLO-pose (configs/calibration/jaw_kpt_<hand>_v002.pt) -> tips A/B -> distance d (px, in the jaw crop) ->
  * confidence = detection score; frames with conf < --min-conf are treated as missing and d is interpolated from neighbours
  * d median-filtered over ~0.2 s
  * g = clip((d - d_closed) / (d_open - d_closed), 0, 1)            (0 = closed, 1 = open; per-hand calibration json)
  * state: hysteresis on d around the per-hand threshold (+-band px); RIGHT hand: inside the band the ResNet classifier (umi_jaw
    infer, v003) decides, because the right-hand threshold sits close to the closed-on-cloth distances
Validated on take 20261022_105200 (96 hand-labelled frames): left 48/48, right 46/48 (binary v003 classifier: 37/48 and 47/48).
Writes into <wrist dir>/umi_jaw.npz: keeps the classifier output as p_open_cls / state_cls, sets state, g_cont (= aperture g),
g_is_aperture=1, d_px, kpt_conf, valid (0 = interpolated). Needs ultralytics >= 8.4 (run with ~/finger_distill/.venv/bin/python).
    ~/finger_distill/.venv/bin/python scripts/pose/jaw_aperture.py datasets/tag_rig/<take>/wrist        (and .../wrist_left)
    ... jaw_aperture.py --rescore <wrist dir>      # calibration changed: recompute g/state from the saved distances
    NOTE v002: label rounds 1/2 used different tip conventions, so --calibrate on the merged labels is wrong; the v002 json
    thresholds were set by hand (see its note). Do not rerun --calibrate on combined_v002.
    ~/finger_distill/.venv/bin/python scripts/pose/jaw_aperture.py --calibrate datasets/jaw_kpt/20261022_105200   # (re)build the calibration"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts" / "pose")); sys.path.insert(0, str(ROOT / "src"))
from jaw_kpt_label import crop              # noqa: E402  same crop as the labels
CAL = ROOT / "configs/calibration/jaw_kpt_calib_v002.json"   # 10-22: v002 models (white + dark cloth, new camera settings)


def otsu(x: np.ndarray) -> float:
    h, e = np.histogram(x, bins=60); c = (e[:-1] + e[1:]) / 2; best = (0.0, float(np.median(x)))
    for k in range(1, len(h)):
        w0, w1 = h[:k].sum(), h[k:].sum()
        if w0 and w1:
            v = w0 * w1 * ((h[:k] * c[:k]).sum() / w0 - (h[k:] * c[k:]).sum() / w1) ** 2
            if v > best[0]: best = (v, float(e[k]))
    return best[1]


def calibrate(D: Path) -> None:
    cal = {}
    for hand in ("left", "right"):
        L = json.loads((D / hand / "labels.json").read_text())["labels"]
        d = np.array([np.linalg.norm(np.subtract(*l["tips"])) for l in L if l["visible"] and l["tips"]]); thr = otsu(d)
        cal[hand] = dict(threshold_px=thr, d_closed_px=float(np.median(d[d < thr])), d_open_px=float(np.percentile(d[d >= thr], 90)), band_px=15.0,
                         model=f"configs/calibration/jaw_kpt_{hand}_v002.pt", n_labels=int(len(d)))
    cal["source"] = str(D); cal["note"] = "10-22 Otsu threshold on hand-labelled tip distances; distances in px of the jaw crop (jaw_kpt_label.CROP)"
    CAL.write_text(json.dumps(cal, indent=1)); print(json.dumps(cal, indent=1))


def run(a) -> None:
    from ultralytics import YOLO
    sys.path.insert(0, str(ROOT / "src")); from visual_teleop.tools.umi_jaw import events
    d = Path(a.dir); meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}; hand = meta.get("hand", "right")
    c = json.loads(CAL.read_text())[hand]; t = np.load(d / "t_ns.npy")
    old = dict(np.load(d / "umi_jaw.npz")) if (d / "umi_jaw.npz").exists() else {}
    if "p_open_cls" in old: old["p_open"] = old["p_open_cls"]; old["state"] = old.get("state_cls", old.get("state"))   # rerun: classifier, not our last output
    if a.rescore and "d_px" in old and str(old.get("model_kpt_dpx", "")) == Path(c["model"]).name:                 # new calibration only: reuse the saved (already filtered) distances, skip YOLO
        Dm = old["d_px"].astype(float); conf = old["kpt_conf"]; valid = old["valid"].astype(bool); D = None
    else:
        Dm = None; net = YOLO(str(ROOT / c["model"])); D = np.full(len(t), np.nan); conf = np.zeros(len(t))
    for k in (range(len(t)) if D is not None else []):
        im = cv2.imread(str(d / f"wrist_{k:05d}.jpg"))
        if im is None: continue
        r = net.predict(crop(im), verbose=False, device=a.device)[0]
        if r.keypoints is None or len(r.boxes) == 0: continue
        j = int(np.argmax(r.boxes.conf.cpu().numpy())); kp = r.keypoints.xy[j].cpu().numpy(); conf[k] = float(r.boxes.conf[j])
        D[k] = float(np.linalg.norm(kp[0] - kp[1]))
        if k % 500 == 0: print(f"[개구] {d.name} {k}/{len(t)}", flush=True)
    if Dm is None:
        valid = np.isfinite(D) & (conf >= a.min_conf)
        if valid.sum() < 2: sys.exit("[개구] 검출된 프레임이 거의 없음")
        Di = np.interp(np.arange(len(t)), np.where(valid)[0], D[valid])
        kk = max(1, int(round(0.2 * (len(t) - 1) / max((t[-1] - t[0]) / 1e9, 1e-9)))) | 1
        Dm = np.array([np.median(Di[max(0, i - kk // 2):i + kk // 2 + 1]) for i in range(len(t))])
    g = np.clip((Dm - c["d_closed_px"]) / (c["d_open_px"] - c["d_closed_px"]), 0, 1).astype(np.float32)
    thr, band = c["threshold_px"], c["band_px"]
    # pseudo-probability for the existing hysteresis/event code: 1 above thr+band, 0 below thr-band, linear inside
    p = np.clip((Dm - (thr - band)) / (2 * band), 0, 1)
    if hand == "right" and "p_open" in old:        # inside the band defer to the classifier (closed-on-cloth sits near the threshold)
        pc = np.interp(t, old["t_ns"], old["p_open"]); inside = np.abs(Dm - thr) < band; p = np.where(inside, pc, p)
    st, ev = events(t, p, 0.6, 0.4, a.min_dwell)
    out = dict(old); out.update(t_ns=t, state=st, g_cont=g, g_is_aperture=np.int8(1), d_px=Dm.astype(np.float32), kpt_conf=conf.astype(np.float32),
                               valid=valid.astype(np.int8), hand=hand, model_kpt=Path(c["model"]).name,
                               model_kpt_dpx=(old["model_kpt_dpx"] if D is None else Path(c["model"]).name), ev_t=np.array([e[0] for e in ev], np.int64), ev_kind=np.array([e[1] for e in ev]))
    if "p_open" in old: out["p_open_cls"] = old["p_open"]; out["state_cls"] = old.get("state")
    out["p_open"] = p.astype(np.float32)
    np.savez(d / "umi_jaw.npz", **{k: v for k, v in out.items() if v is not None})
    print(f"[개구] {d.name} ({hand}): 프레임 {len(t)}, 검출 {valid.mean() * 100:.1f}%, 열림 비율 {st.mean():.2f}, 사건 {len(ev)}개, g 중앙값 {np.median(g):.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-dwell", type=float, default=0.2); ap.add_argument("--device", default="mps")
    ap.add_argument("--rescore", action="store_true", help="reuse saved d_px (after a calibration change), no YOLO"); a = ap.parse_args()
    calibrate(Path(a.dir)) if a.calibrate else run(a)


if __name__ == "__main__":
    main()
