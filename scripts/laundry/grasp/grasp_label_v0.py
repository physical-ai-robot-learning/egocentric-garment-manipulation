#!/usr/bin/env python3
"""[2026-11-05] Grasp success / failure labeler v0 (rule baseline, NOT validated -- needs the user's gold check before any dataset use).

Per close event of each HandUMI jaw (wrist/umi_jaw.npz ev_t), decide held / empty / unknown from that hand's wrist camera:
  the jaw servo is torque-off position-only (no load/current), and the closed position does not separate cloth from nothing
  (right jaw closes to 5263-5279 ticks every time, left varies ~300 ticks with grip force) -> vision cue:
  while the jaw is closed AND the hand moves (wrist-cam ego flow), the jaw-tip disk (minus the fixed jaw pixels) shows
    held  = cloth stays there and moves WITH the camera (dark fraction high, tip flow << ego flow)
    empty = the tip disk sees the table / background (dark fraction low)
    unknown = hand did not move while closed, or the cues disagree (never forced)
Frame labels on the jaw stream: open / closing (first CLOSING_S after a close event) / held / empty / unknown.
Dark = gray < DARK (the 11-03/11-04 shirts are black on a white table -- colored shirts need a different cloth cue).

usage: grasp_label_v0.py OUT_DIR TAKE/demoNN [...]      -> OUT_DIR/<take>_dNN_<counts>.mp4 + events.csv + labels/<take>_dNN.npz
"""
import csv, json, pathlib, subprocess, sys
import cv2, numpy as np

TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
TRIM = pathlib.Path(__file__).resolve().parent.parent / "reports/idle_trim.csv"
CAMS = (("L", "wrist_left"), ("R", "wrist"))
W, H = 320, 180                # analysis resolution
DARK, CLOTH_HELD, CLOTH_EMPTY, RATIO_HELD = 90, 0.5, 0.2, 0.5
EGO_MOVE_PX, MIN_MOVING, CLOSING_S, SAMPLE_HZ = 1.0, 3, 0.3, 10.0
TEX_GRAD, TEX_MIN_FRAC, EDGE_S = 40.0, 0.05, 0.5   # textured-pixel gradient; min textured share of the disk; rest-at-window-edge margin
COL = {"open": (200, 200, 200), "closing": (0, 200, 255), "held": (60, 200, 60), "empty": (40, 40, 230), "unknown": (200, 120, 200)}
LBL = ("open", "closing", "held", "empty", "unknown")


def demo_window(take, d):
    m = json.loads((TR / take / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
    p = [p for p in m["protocol"] if p["text"].startswith("자유:")][d]
    tr = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(TRIM))}.get((take, d))
    a = t0 + int(p["t0"] * 1e9)
    if tr: return a + int(float(tr["keep_start_s"]) * 1e9), a + int(float(tr["keep_end_s"]) * 1e9)
    return a, a + int(p["dur"] * 1e9)


def gray(cam_dir, i):
    im = cv2.imread(str(cam_dir / f"wrist_{i:05d}.jpg"))
    return None if im is None else cv2.cvtColor(cv2.resize(im, (W, H)), cv2.COLOR_BGR2GRAY)


def jaw_mask_and_tip(cam_dir, tw, J):
    """fixed jaw pixels = dark in >= 90 % of closed frames over the take; tip = mean of the topmost 10 % of those near the center"""
    closed = np.flatnonzero(J["state"] == 0)
    if len(closed) < 20: closed = np.arange(len(J["t_ns"]))
    idx = np.unique(np.searchsorted(tw, J["t_ns"][closed[np.linspace(0, len(closed) - 1, 200).astype(int)]]).clip(0, len(tw) - 1))
    G = np.stack([g for g in (gray(cam_dir, i) for i in idx) if g is not None])
    mask = (G < DARK).mean(0) >= 0.9
    ys, xs = np.nonzero(mask[:, int(0.35 * W):int(0.65 * W)]); xs = xs + int(0.35 * W)
    if len(ys) < 30: return mask, (int(0.49 * W), int(0.62 * H))
    top = ys <= np.percentile(ys, 10)
    return mask, (int(xs[top].mean()), int(ys[top].mean()))


def event_features(cam_dir, tw, t_a, t_b, mask, tip, r):
    yy, xx = np.mgrid[:H, :W]; disk = ((xx - tip[0]) ** 2 + (yy - tip[1]) ** 2 <= r * r) & ~mask
    rest = ~mask & ~disk
    ts = np.arange(t_a, t_b, 1e9 / SAMPLE_HZ); idx = np.unique(np.searchsorted(tw, ts).clip(0, len(tw) - 1))
    prev, rows = None, []
    for i in idx:
        g = gray(cam_dir, i)
        if g is None: continue
        if prev is not None:
            f = np.linalg.norm(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2)
            ego = float(np.median(f[rest])); gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)); tex = disk & (gm > TEX_GRAD)
            # flow on textured disk pixels only: uniform black cloth gives ~0 Farneback flow whether it is held or sliding under the tips
            rat = float(np.median(f[tex])) / max(ego, 1e-3) if tex.sum() >= TEX_MIN_FRAC * disk.sum() else np.nan
            rows.append((ego, rat, float((g[disk] < DARK).mean())))
        prev = g
    R = np.array(rows).reshape(-1, 3); mv = R[R[:, 0] > EGO_MOVE_PX]
    f = dict(n_samples=len(R), n_moving=len(mv), ego_med=round(float(np.median(R[:, 0])), 2) if len(R) else 0.0,
             cloth=round(float(np.median(mv[:, 2])), 2) if len(mv) else None,
             ratio=round(float(np.nanmedian(mv[:, 1])), 2) if len(mv) and np.isfinite(mv[:, 1]).sum() >= MIN_MOVING else None)
    if len(mv) < MIN_MOVING: lab, why = "unknown", "hand still while closed"
    elif f["cloth"] < CLOTH_EMPTY: lab, why = "empty", "tips see table/background"
    elif f["ratio"] is None: lab, why = "unknown", "no texture at the tips (uniform cloth)"
    elif f["cloth"] >= CLOTH_HELD and f["ratio"] < RATIO_HELD: lab, why = "held", "cloth at tips moves with the jaw"
    elif f["ratio"] >= 0.8: lab, why = "empty", "cloth slides under the tips"
    else: lab, why = "unknown", "cues disagree"
    return lab, why, f


def label_hand(take, cam, t0, t1):
    cam_dir = TR / take / cam; J = np.load(cam_dir / "umi_jaw.npz"); tw = np.load(cam_dir / "t_ns.npy")
    mask, tip = jaw_mask_and_tip(cam_dir, tw, J); r = int(0.09 * W)
    ev = list(zip(J["ev_t"].tolist(), J["ev_kind"].tolist())); tj = J["t_ns"]
    lab = np.where(J["state"] == 0, 4, 0).astype(np.int8)           # closed -> unknown until an event decides it
    events = []
    for k, (te, kind) in enumerate(ev):
        if kind != "close": continue
        tn = next((tb for tb, kb in ev[k + 1:] if kb == "open"), int(tj[-1]))
        if tn < t0 or te > t1: continue
        a_, b_ = max(te + CLOSING_S * 1e9, t0), min(tn, t1)
        l, why, f = event_features(cam_dir, tw, a_, b_, mask, tip, r)
        rest = te < t0 + EDGE_S * 1e9 or tn > t1 - EDGE_S * 1e9    # jaw already closed at demo start / still closed at the end = resting, not a grasp attempt
        seg = (tj >= te) & (tj < tn); lab[seg] = LBL.index(l); lab[seg & (tj < te + CLOSING_S * 1e9)] = 1
        events.append(dict(t_close_ns=te, t_open_ns=tn, closed_s=round((tn - te) / 1e9, 2), label=l, why=why, rest=int(rest), **f))
    return dict(tj=tj, lab=lab, events=events, mask=mask, tip=tip, r=r, tw=tw, cam_dir=cam_dir)


def render(out_mp4, take, d, t0, t1, hands):
    tw_ref = hands["R"]["tw"]; ts = np.arange(t0, t1, 1e9 / 15.0)      # 15 fps real time
    tw_, th_ = 480, 270; bar = 70
    tmp = out_mp4.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (2 * tw_, th_ + bar))
    # timelines (per hand) precomputed at 1 px per output column
    tl = {}
    for h, D in hands.items():
        tc = np.linspace(t0, t1, 2 * tw_); li = D["lab"][np.searchsorted(D["tj"], tc).clip(0, len(D["tj"]) - 1)]
        tl[h] = np.array([COL[LBL[x]] for x in li], np.uint8)
    for t in ts:
        panes = []
        for h in ("L", "R"):
            D = hands[h]; i = int(np.searchsorted(D["tw"], t).clip(0, len(D["tw"]) - 1))
            im = cv2.imread(str(D["cam_dir"] / f"wrist_{i:05d}.jpg")); im = np.zeros((th_, tw_, 3), np.uint8) if im is None else cv2.resize(im, (tw_, th_))
            s = tw_ / W; cv2.circle(im, (int(D["tip"][0] * s), int(D["tip"][1] * s)), int(D["r"] * s), (255, 255, 0), 1)
            j = int(np.searchsorted(D["tj"], t).clip(0, len(D["tj"]) - 1)); name = LBL[D["lab"][j]]
            evn = [n for n, e in enumerate(D["events"]) if e["t_close_ns"] <= t < e["t_open_ns"]]
            txt = f"{h}  {name.upper()}" + (f"  #{h}{evn[0] + 1}" + (" (rest)" if D["events"][evn[0]]["rest"] else "") if evn else "")
            cv2.rectangle(im, (0, 0), (tw_, 34), COL[name], -1); cv2.putText(im, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            panes.append(im)
        fr = np.vstack([np.hstack(panes), np.full((bar, 2 * tw_, 3), 30, np.uint8)])
        for row, h in enumerate(("L", "R")):
            y0 = th_ + 8 + row * 30; fr[y0:y0 + 22] = tl[h][None]; cv2.putText(fr, h, (4, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        x = int((t - t0) / (t1 - t0) * (2 * tw_ - 1)); cv2.line(fr, (x, th_), (x, th_ + bar), (255, 255, 255), 2)
        cv2.putText(fr, f"{take} d{d + 1:02d}  {(t - t0) / 1e9:5.1f}s", (2 * tw_ - 330, th_ + bar - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(fr)
    vw.release()
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(out_mp4)], check=True)
    tmp.unlink()


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rows = []
    for key in sys.argv[2:]:
        take, dn = key.split("/"); d = int(dn[4:]); t0, t1 = demo_window(take, d)
        hands = {h: label_hand(take, cam, t0, t1) for h, cam in CAMS}
        n = {k: sum(e["label"] == k and not e["rest"] for D in hands.values() for e in D["events"]) for k in ("held", "empty", "unknown")}
        name = f"{take}_d{d + 1:02d}_H{n['held']}E{n['empty']}U{n['unknown']}"
        render(out / f"{name}.mp4", take, d, t0, t1, hands)
        np.savez_compressed(out / "labels" / f"{take}_d{d + 1:02d}.npz", **{f"{h}_t_ns": D["tj"] for h, D in hands.items()},
                            **{f"{h}_label": D["lab"] for h, D in hands.items()}, label_names=np.array(LBL))
        for h, D in hands.items():
            for k, e in enumerate(D["events"]):
                rows.append(dict(video=name, event=f"{h}{k + 1}", t_close_s=round((e["t_close_ns"] - t0) / 1e9, 1), **{x: e[x] for x in e if not x.endswith("_ns")}, user_ok=""))
        print(f"{name}  " + "  ".join(f"{h}:" + ",".join(e["label"][0] for e in D["events"]) for h, D in hands.items()), flush=True)
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
