#!/usr/bin/env python3
"""[2026-10-23] Wrist-camera domain gap, UMI (training) vs robot (deploy), quantified -- UMI-L <-> robot-L, UMI-R <-> robot-R.
UMI side  = raw take frames (wrist/, wrist_left/ 1280x720 JPEG, = what the converter resized) from the first 1.5 s of every training
            demo ('자유:' start = the typical cloth start state), arm-assigned exactly like the converter (source_meta assignments).
Robot side= live robot wrist cameras (deploy/act_tagrig_cams.json locations, same 1920x1080 -> 1280x720 path + fixed exposure as deploy).
Per side: resolution / capture path, nominal intrinsics + FOV (same Arducam KB4 model), brightness (grey mean / p1 / p50 / p99, per
channel), jaw mask (dark pixels connected to the bottom edge, centre band) -> area %, bbox, centroid, width at 90 % height, axis angle,
cloth mask (beige/brown HSV) -> area %, centroid; orientation check = jaw at bottom centre, axis ~vertical (no rotation / flip).
A deterministic framing fix (scale about the centre + shift, from the jaw bboxes) is PROPOSED when the jaws differ by > 5 % scale
or > 15 px, applied to the robot frames and re-measured (before/after); it is written to act_tagrig_cams.json as wrist_align_proposed
and used only with `act_tagrig_deploy.py run --wrist-align` after the user copies it to wrist_align.
Output: Desktop act_wrist_domain_gap_<t>.jpg (paired, masks overlaid), _hist.png, reports/act_deploy/domain_gap_<t>.json
usage: .venv-mac/bin/python deploy/act_tagrig_domain_gap.py [--frames 20] [--umi-per-demo 3]"""
import argparse, json, pathlib, sys, time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: E402
from egogarment import paths as egopaths  # noqa: E402
from garment_pipeline.eval_deploy.act_tagrig_deploy import CAMS_CFG, DESKTOP, OUT, WristCam, recorder_running, umi_location_warning  # noqa: E402

TAG = egopaths.TAG_RIG
FISHEYE = {"left": egopaths.DATA_CALIB / "fisheye_left_v002.yaml",
           "right": egopaths.DATA_CALIB / "fisheye_right_v002.yaml"}


def jaw_mask(bgr):
    """dark (grey < 70) pixels connected to the bottom edge inside the centre 40 % band, lower 70 % of the image (a black start tag
    touching the jaw can leak in -> medians over many frames)"""
    h, w = bgr.shape[:2]; g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); m = (g < 70).astype(np.uint8)
    m[: int(0.30 * h)] = 0; m[:, : int(0.3 * w)] = 0; m[:, int(0.7 * w):] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if y + hh >= h - 3 and a > 0.002 * h * w: keep[lab == i] = 1
    return keep.astype(bool)


def cloth_mask(bgr, jaw):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (H >= 5) & (H <= 25) & (S >= 35) & (S <= 170) & (V >= 50) & (V <= 220) & ~jaw
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool); return m


def measure(bgr):
    h, w = bgr.shape[:2]; g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); jm = jaw_mask(bgr); cm = cloth_mask(bgr, jm)
    out = dict(size=[w, h], grey=dict(mean=float(g.mean()), p1=float(np.percentile(g, 1)), p50=float(np.median(g)), p99=float(np.percentile(g, 99))),
               bgr_mean=[float(x) for x in bgr.reshape(-1, 3).mean(0)], contrast_std=float(g.std()),
               jaw_area_pct=float(jm.mean() * 100), cloth_area_pct=float(cm.mean() * 100))
    if jm.any():
        ys, xs = np.nonzero(jm); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); row = int(y0 + 0.9 * (y1 - y0)); r_ = np.nonzero(jm[row])[0]
        c = np.cov(np.c_[xs, ys].T); ev, evec = np.linalg.eigh(c); ax = evec[:, -1]; ang = float(np.degrees(np.arctan2(ax[0], ax[1]))); ang = (ang + 90) % 180 - 90
        out.update(jaw_bbox=[int(x0), int(y0), int(x1), int(y1)], jaw_centroid=[float(xs.mean()), float(ys.mean())], jaw_top_y=int(y0),
                   jaw_width_at90=int(r_.max() - r_.min()) if len(r_) else 0, jaw_axis_deg_from_vertical=ang)
    if cm.any():
        ys, xs = np.nonzero(cm); out["cloth_centroid"] = [float(xs.mean()), float(ys.mean())]
    return out, jm, cm


def agg(ms):
    keys = ["jaw_area_pct", "cloth_area_pct", "contrast_std", "jaw_top_y", "jaw_width_at90", "jaw_axis_deg_from_vertical"]
    r = {k: round(float(np.median([m[k] for m in ms if k in m])), 2) for k in keys if any(k in m for m in ms)}
    r["grey"] = {k: round(float(np.median([m["grey"][k] for m in ms])), 1) for k in ("mean", "p1", "p50", "p99")}
    r["bgr_mean"] = np.round(np.median([m["bgr_mean"] for m in ms], 0), 1).tolist()
    for k in ("jaw_centroid", "cloth_centroid"):
        v = [m[k] for m in ms if k in m]
        if v: r[k] = np.round(np.median(v, 0), 1).tolist()
    bb = [m["jaw_bbox"] for m in ms if "jaw_bbox" in m]
    if bb: r["jaw_bbox"] = np.round(np.median(bb, 0)).astype(int).tolist()
    r["n"] = len(ms); return r


def overlay(bgr, jm, cm, label):
    v = bgr.copy(); v[jm] = (0.5 * v[jm] + [0, 0, 127]).astype(np.uint8); v[cm] = (0.5 * v[cm] + [127, 0, 0]).astype(np.uint8)
    cv2.rectangle(v, (0, 0), (v.shape[1], 50), (0, 0, 0), -1); cv2.putText(v, label, (10, 36), 0, 1.1, (0, 255, 255), 2); return v


def umi_frames(n_per_demo, start_only=True):
    meta = json.loads((core.LDT / "data/lerobot/tagrig_laundry_v1_fused_train/source_meta.json").read_text()); out = {"left": [], "right": []}
    for take, asg in meta["assignments"].items():
        td = TAG / take; m = json.loads((td / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
        for side in ("left", "right"):
            cd = td / asg["cam_of_arm"][side]; t = np.load(cd / "t_ns.npy")
            for p in [p for p in m["protocol"] if p["text"].startswith("자유:")]:
                ts = t0 + int(p["t0"] * 1e9); te = ts + int(p["dur"] * 1e9)
                sel = np.flatnonzero((t >= ts) & (t <= (ts + 1.5e9 if start_only else te)))
                for i in sel[:: max(1, len(sel) // n_per_demo)][:n_per_demo]:
                    f = cd / f"wrist_{i:05d}.jpg"
                    if f.exists(): out[side].append(f)
    return out


def fov(side):
    import yaml
    c = yaml.safe_load(FISHEYE[side].read_text()); K = np.array(c["K"]); D = np.array(c["D"]); W, H = c["image_size"]
    def theta(rd):           # invert KB4 theta_d = th (1 + k1 th^2 + ...)
        th = rd
        for _ in range(30): th = rd / (1 + D[0] * th ** 2 + D[1] * th ** 4 + D[2] * th ** 6 + D[3] * th ** 8)
        return th
    hf = np.degrees(theta((W - K[0, 2]) / K[0, 0]) + theta(K[0, 2] / K[0, 0])); vf = np.degrees(theta((H - K[1, 2]) / K[1, 1]) + theta(K[1, 2] / K[1, 1]))
    return dict(file=FISHEYE[side].name, fx_1920=round(float(K[0, 0]), 1), hfov_deg=round(float(hf), 1), vfov_deg=round(float(vf), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20); ap.add_argument("--umi-per-demo", type=int, default=3); ap.add_argument("--exposure", type=int, default=2)
    ap.add_argument("--whole-demo", action="store_true", help="UMI frames from the whole demos (default: first 1.5 s = start state)"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if recorder_running(): raise SystemExit(f"녹화 프로세스가 카메라를 쓰는 중: {recorder_running()}")
    cfg = json.loads(CAMS_CFG.read_text()); umi_location_warning(cfg); ts = time.strftime("%H%M%S") + (f"_{a.tag}" if a.tag else "")
    U = umi_frames(a.umi_per_demo, start_only=not a.whole_demo); rep = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), cams={s: cfg[s] for s in core.SIDES}, sides={})
    cams = {s: WristCam(s, cfg[s]["location"], cfg[s]["index"], a.exposure, prep=False) for s in core.SIDES}
    for c in cams.values(): c.start()
    time.sleep(1.5); R = {s: [] for s in core.SIDES}
    for _ in range(a.frames):
        for s in core.SIDES: R[s].append(cv2.resize(cams[s].raw, (1280, 720), interpolation=cv2.INTER_AREA))
        time.sleep(0.1)
    for c in cams.values(): c.stop = True
    rows, hists = [], {}
    for s in core.SIDES:
        um = [cv2.imread(str(f)) for f in U[s]]; um = [x for x in um if x is not None]
        mu = [measure(x)[0] for x in um]; mr = [measure(x)[0] for x in R[s]]; au, ar = agg(mu), agg(mr)
        # proposed framing: scale = UMI jaw bbox height / robot jaw bbox height (jaws touch the bottom edge), shift = centroid x
        prop = None
        if "jaw_bbox" in au and "jaw_bbox" in ar:
            hu, hr = 720 - au["jaw_bbox"][1], 720 - ar["jaw_bbox"][1]; sc = hu / max(hr, 1)
            dx = au["jaw_centroid"][0] - (640 + sc * (ar["jaw_centroid"][0] - 640))
            # scale about the centre moves the bottom edge to 360 + sc*360; shift it back down so the jaw still touches the bottom
            dy = 720 - (360 + sc * 360)
            prop = dict(scale=round(float(sc), 3), dx=round(float(dx), 1), dy=round(float(dy), 1), rot180=False, hflip=False)
            if abs(sc - 1) <= 0.05 and abs(dx) <= 15: prop["needed"] = False
            elif 0.8 <= sc <= 1.25: prop["needed"] = True
            else: prop.update(needed=False, not_fixable=f"jaw scale ratio {sc:.2f}: a zoom this large would cut the fisheye FOV -> jaw / camera mount geometry differs, not a framing offset")
        after = agg([measure(core.apply_align(x, prop))[0] for x in R[s]]) if prop and prop["needed"] else None
        rep["sides"][s] = dict(umi=au, robot=ar, robot_after_align=after, proposed_align=prop, umi_files=len(um),
                               capture=dict(umi="Arducam 0x0c45:0x0261, 1920x1080 -> 1280x720 INTER_AREA, JPEG q92 (umi_jaw record), fixed exposure 2 + IMAGE_CTRL",
                                            robot="same model / same path (deploy WristCam + core.prep_wrist), fixed exposure %d + IMAGE_CTRL" % a.exposure),
                               intrinsics=dict(umi_nominal=fov(s), robot="not calibrated (same Arducam fisheye model; assume nominal)"),
                               orientation=dict(umi_jaw_axis_deg=au.get("jaw_axis_deg_from_vertical"), robot_jaw_axis_deg=ar.get("jaw_axis_deg_from_vertical"),
                                                verdict="no rotation / flip needed" if abs(ar.get("jaw_axis_deg_from_vertical", 90) - au.get("jaw_axis_deg_from_vertical", 0)) < 15
                                                and abs(ar.get("jaw_centroid", [0])[0] - 640) < 200 else "CHECK"),
                               left_right=dict(robot_cloth_centroid_x=ar.get("cloth_centroid", [None])[0],
                                               expect=("cloth right of centre (x > 640) for the LEFT arm" if s == "left" else "cloth left of centre (x < 640) for the RIGHT arm")))
        # paired image: best-matching UMI frame (closest cloth area) vs the last robot frame
        # typical UMI frame: jaw extent near the UMI median, brightness near the robot's (the cloth-area match picked washed-out frames)
        k = int(np.argmin([abs(m.get("jaw_top_y", 0) - au.get("jaw_top_y", 0)) / 50 + abs(m["grey"]["mean"] - ar["grey"]["mean"]) / 20 for m in mu])) if mu else 0
        _, jm, cm = measure(um[k]); a1 = overlay(um[k], jm, cm, f"UMI {s.upper()} (train) {U[s][k].parent.parent.name}/{U[s][k].parent.name}")
        _, jm, cm = measure(R[s][-1]); a2 = overlay(R[s][-1], jm, cm, f"ROBOT {s.upper()} live {cfg[s]['location']}")
        row = [a1, a2]
        if after: x = core.apply_align(R[s][-1], prop); _, jm, cm = measure(x); row.append(overlay(x, jm, cm, f"ROBOT {s.upper()} + proposed align"))
        rows.append(np.hstack([cv2.resize(r_, (640, 360)) for r_ in row] + ([np.zeros((360, 640, 3), np.uint8)] if len(row) == 2 else [])))
        hists[s] = (np.concatenate([cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256]).ravel()[None] for x in um]).mean(0),
                    np.concatenate([cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256]).ravel()[None] for x in R[s]]).mean(0))
    pj = DESKTOP / f"act_wrist_domain_gap_{ts}.jpg"; cv2.imwrite(str(pj), np.vstack(rows))
    pl = []
    for s in core.SIDES:
        hu, hr = hists[s]; x = np.arange(64) * 4
        pl.append(core.cv_plot([(x, hu / hu.sum(), (200, 120, 0), "UMI (train starts)"), (x, hr / hr.sum(), (0, 120, 230), "robot (live)")],
                               (0, float(max((hu / hu.sum()).max(), (hr / hr.sum()).max()))), f"{s} wrist grey histogram", (640, 260), xlabel="grey"))
    ph = DESKTOP / f"act_wrist_domain_gap_{ts}_hist.png"; cv2.imwrite(str(ph), np.hstack(pl))
    cfg["wrist_align_proposed"] = {s: rep["sides"][s]["proposed_align"] for s in core.SIDES}; CAMS_CFG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    rep.update(paired=str(pj), hist=str(ph)); OUT.mkdir(parents=True, exist_ok=True); pr = OUT / f"domain_gap_{ts}.json"; pr.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False)); print("저장", pj, ph, pr)


if __name__ == "__main__":
    main()
