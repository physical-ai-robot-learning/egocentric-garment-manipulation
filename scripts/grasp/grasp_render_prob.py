#!/usr/bin/env python3
"""[2026-11-05] Causal frame-level p_grasp(t) videos (color = probability, not classes).  11-05 user: score OPEN frames too (every tick).
CNN = the v1 per-frame ResNet-18 recipe trained on ALL v2 gold (data/grasp_labels_v2/gold_v2.csv, v2 jaw segmentation open>0.30/closed<0.15).
p_grasp(t) (PG_HZ, default 15 Hz; window PG_WIN_S, default 0.2 s since the user review) uses ONLY frames <= t: valid while the v2 jaw is closed and t >= close + 0.3 s; p = mean per-frame P(grasp) of the jaw-ROI
crops in [max(close + 0.3, t - 0.6), t].  Otherwise valid = 0 (shown gray = open, blue = closed but not yet valid).
Color: P(grasp) 0 red -> 0.5 yellow -> 1 green.  Saves OUT/<demo>_pgrasp.npz (t_ns, p, valid per hand) + mp4.
usage: grasp_render_prob.py OUT_DIR DEMO_KEY [...]     (DEMO_KEY = take_dNN, 1-indexed as in the video names)"""
import csv, pathlib, subprocess, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True); DEMOS = sys.argv[2:]
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_label_v0 as G, grasp_cnn_v1 as C, grasp_v2_events as E2  # noqa: E402
import json, os
# 11-05 user (after the confident-wrong clips): EMPTY-gold P>0.9 = grasped-then-slipped -> "window size를 줄이면" -> 0.2 s window at 15 Hz
HZ, WARM_S, WIN_S = float(os.environ.get("PG_HZ", 15)), 0.3, float(os.environ.get("PG_WIN_S", 0.2))
OVR = json.loads((H / "laundry_dit_training/data/grasp_labels_v2/gold_overrides_v2.json").read_text())   # slip -> excluded, loose -> not held


def crop_fn(take, cam):
    cd = G.TR / take / cam; tw = np.load(cd / "t_ns.npy"); tip = G.jaw_mask_and_tip(cd, tw, np.load(cd / "umi_jaw.npz"))[1]
    cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H; x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464))
    def f(t):
        i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1)); im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg"))
        im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
        return cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224))
    return f, tw, cd, tip


def train_all():
    gold = list(csv.DictReader(open(H / "laundry_dit_training/data/grasp_labels_v2/gold_v2.csv"))); X, y, fns = [], [], {}
    for g in gold:
        ov = OVR.get(g["id"])
        if ov == "slip": continue                    # held THEN released inside one close: mixed, no single event label
        if ov == "loose": g = dict(g, gold="empty")  # arranging the cloth, not a secure grasp
        k = (g["take"], g["cam"])
        if k not in fns: fns[k] = crop_fn(*k)[0]
        te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
        X.append(np.stack([fns[k](te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9))) for dt in C.DTS])); y.append(int(g["gold"] == "empty"))
        # 11-05 reviewer: p_grasp = P(cloth securely held | recent frames) -> the classifier itself must output ~0 on OPEN frames (no rule
        # override): 4 frames of the open approach before this close (te - 0.6 .. -0.15 s) as "not held" (= label 1 for the P(empty) head)
        X.append(np.stack([fns[k](te - dt * 1e9) for dt in (0.6, 0.45, 0.3, 0.15)])); y.append(1)
    m = C.train_fold(np.stack(X), np.array(y)); print(f"CNN trained on {len(y)} samples = v2 gold closes + one open-approach negative each ({sum(y)} not-held)", flush=True); return m


def color(p, valid, closed):
    p = float(np.clip(p, 0, 1)); r = 255 if p < 0.5 else int(255 * (1 - p) * 2); g = int(255 * p * 2) if p < 0.5 else 255
    return (0, g, r)


def p_series(m, take, cam, ts):
    f, tw, cd, tip = crop_fn(take, cam); tj, st, ev = E2.jaw_v2(take, cam)
    closed = st[np.searchsorted(tj, ts).clip(0, len(tj) - 1)] == 0
    # 11-05 user: "열려있는 동안도 판정해줘" -> every tick is scored (open too); the causal window restarts at each jaw state change
    chg = np.array([t for t, k in ev]); last_chg = np.array([chg[chg <= t].max() if (chg <= t).any() else ts[0] for t in ts])
    fp = np.zeros(len(ts))
    for b in range(0, len(ts), 64): fp[b:b + 64] = 1 - C.predict(m, np.stack([f(t) for t in ts[b:b + 64]])[:, None])
    p = np.array([fp[(ts >= max(last_chg[i], ts[i] - WIN_S * 1e9)) & (ts <= ts[i])].mean() for i in range(len(ts))])   # causal window mean
    return dict(p=p, valid=np.ones(len(ts), bool), closed=closed, tw=tw, cd=cd, tip=tip)


def wrong_clips(m):
    """the 25 user-reviewed clips (~/Desktop/grasp_confident_wrong) again, now with the causal P(grasp)(t) strip"""
    gold = {g["id"]: g for g in csv.DictReader(open(H / "laundry_dit_training/data/grasp_labels_v2/gold_v2.csv"))}
    for eid, kind in OVR.items():
        if eid.startswith("_"): continue
        g = gold[eid]; tj, st, ev = E2.jaw_v2(g["take"], g["cam"]); te, tn = int(g["t_close_ns"]), int(g["t_open_ns"])
        po = max([t for t, k in ev if k == "open" and t < te], default=te - 10**10); nc = min([t for t, k in ev if k == "close" and t > te], default=tn + 10**10)
        a = max(te - 1e9, po + 0.15e9); b = min(tn + 1e9, nc - 0.15e9, a + 7e9); ts = np.arange(a, b, 1e9 / HZ).astype(np.int64)
        D = p_series(m, g["take"], g["cam"], ts); name = f"{kind.upper()}__{eid}"; tmp = OUT / f"{name}.raw.mp4"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 430)); strip = np.array([color(x, 1, 1) for x in D["p"][np.linspace(0, len(ts) - 1, 640).astype(int)]], np.uint8)
        for t in np.arange(a, b, 1e9 / 15):
            k = int(np.searchsorted(ts, t).clip(0, len(ts) - 1)); i = int(np.searchsorted(D["tw"], t).clip(0, len(D["tw"]) - 1))
            im = cv2.imread(str(D["cd"] / f"wrist_{i:05d}.jpg")); im = np.zeros((360, 640, 3), np.uint8) if im is None else cv2.resize(im, (640, 360))
            bar = np.full((70, 640, 3), 30, np.uint8); bar[4:26] = strip[None]; x = int((t - a) / (b - a) * 639); cv2.line(bar, (x, 0), (x, 30), (255, 255, 255), 2)
            cl = te <= t < tn; cv2.rectangle(bar, (0, 32), (639, 38), (0, 160, 255), -1) if cl else None
            cv2.putText(bar, f"{kind}  P(grasp) {D['p'][k]:.2f}  {'JAW CLOSED' if cl else 'jaw open'}  {(t - a) / 1e9:4.1f}s", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(np.vstack([im, bar]))
        vw.release()
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(OUT / f"{name}.mp4")], check=True); tmp.unlink()
        print(name, f"P(grasp) closed max {D['p'][(ts >= te) & (ts < tn)].max():.2f} end {D['p'][(ts >= te) & (ts < tn)][-1]:.2f}", flush=True)


def main():
    m = train_all()
    if DEMOS == ["--wrong"]: return wrong_clips(m)
    for dk in DEMOS:
        take = "_".join(dk.split("_")[:2]); d = int(dk.split("_")[2][1:]) - 1; t0, t1 = G.demo_window(take, d)
        ts = np.arange(t0, t1, 1e9 / HZ).astype(np.int64); hands = {}
        for h, cam in G.CAMS: hands[h] = p_series(m, take, cam, ts)
        np.savez(OUT / f"{dk}_pgrasp.npz", t_ns=ts, **{f"{h}_p": D["p"] for h, D in hands.items()}, **{f"{h}_valid": D["valid"].astype(np.int8) for h, D in hands.items()})
        render(OUT / f"{dk}_pgrasp.mp4", dk, ts, t0, t1, hands); print(dk, {h: f"valid {D['valid'].mean():.0%} mean p {D['p'][D['valid']].mean():.2f}" if D["valid"].any() else "-" for h, D in hands.items()}, flush=True)


def render(out, dk, ts, t0, t1, hands):
    W_, H_, bar = 480, 270, 70; tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (2 * W_, H_ + bar))
    tl = {}
    for h, D in hands.items():
        col = np.linspace(0, len(ts) - 1, 2 * W_).astype(int); tl[h] = np.array([color(D["p"][i], D["valid"][i], D["closed"][i]) for i in col], np.uint8)
    for t in np.arange(t0, t1, 1e9 / 15):
        k = int(np.searchsorted(ts, t).clip(0, len(ts) - 1)); panes = []
        for h in ("L", "R"):
            D = hands[h]; i = int(np.searchsorted(D["tw"], t).clip(0, len(D["tw"]) - 1)); im = cv2.imread(str(D["cd"] / f"wrist_{i:05d}.jpg"))
            im = np.zeros((H_, W_, 3), np.uint8) if im is None else cv2.resize(im, (W_, H_)); s = W_ / G.W
            c = color(D["p"][k], D["valid"][k], D["closed"][k]); cv2.rectangle(im, (0, 0), (W_, 34), c, -1)
            txt = f"{h}  P(grasp) {D['p'][k]:.2f}   jaw {'CLOSED' if D['closed'][k] else 'open'}"
            cv2.putText(im, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.circle(im, (int(D["tip"][0] * s), int(D["tip"][1] * s)), int(0.09 * G.W * s), (255, 255, 0), 1); panes.append(im)
        fr = np.vstack([np.hstack(panes), np.full((bar, 2 * W_, 3), 30, np.uint8)])
        for row, h in enumerate(("L", "R")):
            y0 = H_ + 8 + row * 30; fr[y0:y0 + 22] = tl[h][None]; cv2.putText(fr, h, (4, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        x = int((t - t0) / (t1 - t0) * (2 * W_ - 1)); cv2.line(fr, (x, H_), (x, H_ + bar), (255, 255, 255), 2)
        cv2.putText(fr, f"{dk}  {(t - t0) / 1e9:5.1f}s", (2 * W_ - 300, H_ + bar - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(fr)
    vw.release()
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(out)], check=True); tmp.unlink()


if __name__ == "__main__":
    main()
