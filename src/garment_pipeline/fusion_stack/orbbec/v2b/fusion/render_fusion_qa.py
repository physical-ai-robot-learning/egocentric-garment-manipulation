"""10-24 fusion QA video per demo (both arms) from stitch/qa/<var>/ runs + the CoTracker pack (writes only to --out).
Left: Orbbec RGB (Orbbec frame rate, idle-trim window) with per arm  production TCP (h-1, filled circle; L green / R cyan),
hidden-tag estimate (the run in which that frame's 4 s block is hidden, orange ring), tag GT (white +, >= 1 tag), raw CoTracker
tag-point (small yellow dot), and a red border flash while |hidden estimate - tag| > 10 cm.
Below: left / right wrist cameras (nearest frame).
Right: per arm x/y/z (production solid, hidden estimate orange, tag dots), |hidden estimate - tag| [mm] with the 100 mm line,
CoTracker gaps shaded red, hidden blocks of the current run not shaded (every frame is hidden in exactly one run); moving cursor.
    python render_fusion_qa.py --pack <ctpack> --var fus_Cg_s --demos TAKE:demo ... --out DIR [--tag-csv fusion_metrics_<tag>.csv]"""
import argparse, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
from garment_pipeline.fusion_stack import core as C  # noqa
from garment_pipeline.fusion_stack import camcal  # noqa
FD = C.FD
T_C_B = np.array(json.loads((FD.EGO / "configs/calibration/robot_cam_v001.json").read_text())["T_C_B"])
RAW_VAR = "fus_R_s"
COL = {"left": (60, 220, 60), "right": (230, 200, 40)}; ARMS = ("left", "right")


def interp(t, p, tq):
    out = np.stack([np.interp(tq, t, p[:, a]) for a in range(3)], 1); out[(tq < t[0]) | (tq > t[-1])] = np.nan; return out


def proj(p, I):
    if not np.isfinite(p[0]): return None
    x = T_C_B[:3, :3] @ p + T_C_B[:3, 3]
    if x[2] <= 0.05: return None
    return int(I["fx"] * x[0] / x[2] + I["cx"]), int(I["fy"] * x[1] / x[2] + I["cy"])


def load(var, take, arm, demo, off):
    f = ROOT / f"stitch/qa/{var}/{take}_{arm}_d{demo:02d}_h{off}.npz"
    if not f.exists(): return None
    z = np.load(f); return z if len(z["t_ns"]) >= 2 else None          # an arm with no valid sample (e.g. gated to 0 %) -> no run


def render(pack, var, take, demo, out, info):
    global T_C_B; T_C_B = camcal.T_C_B(take)                          # 10-26: per-take camera calibration
    tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]; W = {w["demo"]: w for w in FD.trims()[take]}[demo]
    S = {}
    for arm in ARMS:
        D = FD.load_arm(take, arm, with_slam=False); t = D["t"]; k = np.flatnonzero((t >= W["w0"]) & (t <= W["w1"])); tt = t[k]
        prod = load(var, take, arm, demo, -1); off_of = ((tt - t[0]) // int(4e9)) % 3; hid = np.full((len(k), 3), np.nan)
        for off in (0, 1, 2):
            r = load(var, take, arm, demo, off)
            if r is not None: m = off_of == off; hid[m] = interp(r["t_ns"], r["p_B"], tt[m])
        z = np.load(Path(pack) / f"{take}_d{demo:02d}" / f"solve2_{arm}.npz"); tct = z["t"].astype(np.int64); okc = z["conf"] > 0
        rr = load(RAW_VAR, take, arm, demo, -1)
        S[arm] = dict(k=k, tt=tt, prod=interp(prod["t_ns"], prod["p_B"], tt) if prod is not None else np.full((len(k), 3), np.nan), hid=hid,
                      tag=np.where((D["ntags"][k] >= 1)[:, None], D["p_tag"][k], np.nan), ct=interp(rr["t_ns"], rr["p_B"], tt) if rr is not None else None,
                      ctgap=~np.interp(tt, tct, okc.astype(float)).round().astype(bool) | (tt < tct[0]) | (tt > tct[-1]))
        wd = tk / D["cam"]; wt = np.load(wd / "t_ns.npy").astype(np.int64); S[arm]["wrist"] = (wd, np.clip(np.searchsorted(wt, tt), 0, len(wt) - 1))
        S[arm]["err"] = np.linalg.norm(S[arm]["hid"] - S[arm]["tag"], axis=1) * 1000
    k = S["left"]["k"]; tt = S["left"]["tt"]; ts = (tt - tt[0]) / 1e9
    # static plot
    fig, ax = plt.subplots(4, 2, figsize=(8.5, 7.2), dpi=100, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.2]))
    for c, arm in enumerate(ARMS):
        s = S[arm]; gap = s["ctgap"]
        for r_, lab in enumerate("xyz"):
            a = ax[r_, c]; a.plot(ts, s["prod"][:, r_], "g-" if arm == "left" else "c-", lw=1.2); a.plot(ts, s["hid"][:, r_], color="orange", lw=0.9)
            a.plot(ts, s["tag"][:, r_], "k.", ms=1.5); a.set_ylabel(f"{arm[0].upper()} {lab} [m]", fontsize=7); a.tick_params(labelsize=6)
            a.fill_between(ts, 0, 1, where=gap, transform=a.get_xaxis_transform(), color="red", alpha=0.18, lw=0)
        a = ax[3, c]; a.plot(ts, s["err"], color="orange", lw=0.9); a.axhline(100, color="r", lw=0.8, ls="--"); a.set_ylim(0, max(150, np.nanmax(np.r_[s["err"], 0]) * 1.05))
        a.fill_between(ts, 0, 1, where=gap, transform=a.get_xaxis_transform(), color="red", alpha=0.18, lw=0); a.set_ylabel("|hid - tag| mm", fontsize=7); a.tick_params(labelsize=6); a.set_xlabel("s", fontsize=7)
        ax[0, c].set_title(f"{arm}: " + info.get(arm, ""), fontsize=7)
    fig.suptitle(f"{take} d{demo:02d}  {var}   (solid = production, orange = hidden-tag estimate, dots = tag, red = CoTracker gap)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.canvas.draw(); plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    x0 = ax[3, 0].get_window_extent(); x1 = ax[3, 1].get_window_extent(); H_ = plot.shape[0]; plt.close(fig)
    px = lambda e, s_: int(e.x0 + (s_ - ts[0]) / (ts[-1] - ts[0] + 1e-9) * (e.x1 - e.x0))
    h_img = 480; sc = h_img / plot.shape[0] if plot.shape[0] > 0 else 1; plot = cv2.resize(plot, (int(plot.shape[1] * 480 * 1.5 / plot.shape[0]), int(480 * 1.5))); sc = plot.shape[0] / H_
    out.parent.mkdir(parents=True, exist_ok=True); Wv = 848 + plot.shape[1]; Hv = max(plot.shape[0], 480 + 40); Hv = max(Hv, 720); Wv += Wv % 2; Hv += Hv % 2
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wv}x{Hv}", "-r", "15", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    for j, kk in enumerate(k):
        img = cv2.imread(str(tk / f"color_{kk:05d}.jpg")); canvas = np.zeros((Hv, Wv, 3), np.uint8)
        if img is None: img = np.zeros((480, 848, 3), np.uint8)
        bad = False
        for arm in ARMS:
            s = S[arm]; c_ = COL[arm]
            if s["ct"] is not None and (q := proj(s["ct"][j], I)): cv2.circle(img, q, 3, (0, 230, 255), -1)
            if q := proj(s["prod"][j], I): cv2.circle(img, q, 7, c_, -1)
            if q := proj(s["hid"][j], I): cv2.circle(img, q, 11, (0, 140, 255), 2)
            if q := proj(s["tag"][j], I): cv2.drawMarker(img, q, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            e = s["err"][j]; bad |= bool(np.isfinite(e) and e > 100)
            cv2.putText(img, f"{arm[0].upper()} err {e:5.0f} mm" if np.isfinite(e) else f"{arm[0].upper()} err   -", (10 if arm == "left" else 640, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_, 1)
            if s["ctgap"][j]: cv2.putText(img, "CT GAP", (10 if arm == "left" else 640, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if bad: cv2.rectangle(img, (0, 0), (847, 479), (0, 0, 255), 6)
        cv2.putText(img, f"{take} d{demo:02d}  t={ts[j]:5.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        canvas[:480, :848] = img; p2 = plot.copy()
        for c, arm in enumerate(ARMS):                          # wrist cameras below (L | R)
            wd, wi = S[arm]["wrist"]; wimg = cv2.imread(str(wd / f"wrist_{wi[j]:05d}.jpg"))
            if wimg is not None:
                canvas[482:720, 424 * c:424 * c + 423] = cv2.resize(wimg, (423, 238)); cv2.putText(canvas, f"{arm} wrist", (424 * c + 6, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[arm], 1)
        for e in (x0, x1):
            xx = int(px(e, ts[j]) * sc); cv2.line(p2, (xx, 0), (xx, p2.shape[0] - 1), (0, 0, 200), 1)
        canvas[:p2.shape[0], 848:848 + p2.shape[1]] = p2
        cv2.putText(canvas, "fill = production  orange ring = hidden-tag estimate  white + = tag  yellow = raw CoTracker  red border = err > 10 cm",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close(); ff.wait()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--var", required=True); ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tag-csv", default=""); ap.add_argument("--prefix", default=""); ap.add_argument("--raw-var", default="fus_R_s"); ap.add_argument("--quality-csv", default=""); ap.add_argument("--start", type=int, default=1); a_ = ap.parse_args()
    global RAW_VAR; RAW_VAR = a_.raw_var
    import pandas as pd
    df = pd.read_csv(ROOT / "stitch/qa" / a_.tag_csv) if a_.tag_csv else None
    qd = pd.read_csv(a_.quality_csv, dtype={"take": str}) if a_.quality_csv else None
    for i, td in enumerate(a_.demos):
        take, demo = td.split(":"); demo = int(demo); info = {}
        if qd is not None:
            r = qd[(qd["take"] == take) & (qd["demo"] == demo)]
            if len(r):
                r = r.iloc[0]
                for arm in ARMS:
                    L = arm[0].upper(); info[arm] = (f"[{r['class']}] cov {r[L + '_cov']:.2f} | gap {r[L + '_gap_s']:.1f}s | far10 {r[L + '_far10']:.2f} | "
                                                    f"burst {r[L + '_burst_s']:.1f}s | swap {r[L + '_swap_s']:.1f}s")
        if df is not None:
            for arm in ARMS:
                r = df[(df["var"] == a_.var) & (df["take"].astype(str) == take) & (df["demo"] == demo) & (df["arm"] == arm)]
                if len(r): r = r.iloc[0]; info[arm] = f"e32 p90 {r.e32_p90:.0f} | abs p90 {r.abs_p90:.0f} | far10 {r.far10:.2f} | burst {r.burst_s:.1f}s | gap {r.gap_s:.1f}s"
        name = f"{a_.prefix}{i + a_.start:02d}_{take}_d{demo:02d}.mp4" if a_.prefix else f"{take}_d{demo:02d}.mp4"
        render(a_.pack, a_.var, take, demo, Path(a_.out) / name, info); print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
