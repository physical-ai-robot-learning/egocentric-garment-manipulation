#!/usr/bin/env python3
"""[2026-11-05] FROZEN p_grasp generator "grasp-v2-W0.4" for the FLAT250-PGRASP run.
1. classifier: jaw-ROI ResNet-18 (grasp_render_prob.train_all: v2 gold closes x4 frames + open-approach negatives, slip excluded,
   loose = not held; user overrides gold_overrides_v2.json) trained on ALL gold -> OUT/model.pt + model_meta.json
2. per demo (take/demoNN of the 11-03 + 11-04 (+ 11-05) S03G train/val/test sets): causal p_grasp(t) at 15 Hz over the demo window
   (demo start .. end, + 1 s margins), W = 0.4 s window restarted at each v2 jaw state change (open > 0.30 / closed < 0.15),
   scored on open AND closed frames -> OUT/demos/<take>_demoNN.npz (t_ns, L_p, R_p; L = wrist_left cam, R = wrist cam)
NOTE: gold events are in the classifier's training set, so p on those events is in-sample (slightly overconfident).
usage: grasp_pgrasp_gen.py OUT_DIR"""
import json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, torch
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); (OUT / "demos").mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(OUT), "--none"]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_cnn_v1 as C, grasp_label_v0 as G, grasp_render_prob as R, grasp_v2_events as E2  # noqa: E402
VERSION, HZ, W, MARGIN_S = "grasp-v2-W0.4", 15.0, 0.4, 1.0
LR = H / "laundry_dit_training/data/lerobot"


def demo_keys():
    keys = []
    for sp in ("train", "val", "test"):
        f = LR / f"flat1103_1104_S03G_{sp}/meta/split.json"
        if f.exists(): keys += json.loads(f.read_text())["episodes"][sp]
    if len(keys) < 300:     # the aggregated train set lives on the cluster only -> the per-take split docs (scratchpad) list the same keys
        S = pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad")
        for f in sorted(S.glob("flat1103n_*_S03G_split.json")) + sorted(S.glob("flat1104_*_S03G_split.json")) + sorted(S.glob("flat1105_*_S03G_split.json")):
            for sp, ks in json.loads(f.read_text())["episodes"].items():
                if sp in ("train", "val", "test"): keys += ks
    return sorted(set(keys))


def main():
    mp = OUT / "model.pt"
    if mp.exists():
        import torchvision
        m = torchvision.models.resnet18(); m.fc = torch.nn.Linear(512, 1); m.load_state_dict(torch.load(mp)); m = m.to(C.DEV).eval()
    else:
        m = R.train_all(); torch.save(m.state_dict(), mp)
        (OUT / "model_meta.json").write_text(json.dumps(dict(version=VERSION, created=time.strftime("%Y-%m-%d %H:%M"), arch="torchvision resnet18, fc -> 1 logit = P(not held)",
            input="jaw-ROI 256 px crop (tip from grasp_label_v0.jaw_mask_and_tip), top at tip_y - 153 px, RGB 224, ImageNet norm",
            train="data/grasp_labels_v2/gold_v2.csv + gold_overrides_v2.json (slip excluded, loose = not held) + open-approach negatives; EPOCHS 8",
            rule=f"p_grasp = 1 - P(not held); causal mean over [max(last v2 jaw change, t - {W}), t] at {HZ:.0f} Hz; v2 jaw hysteresis open>0.30 closed<0.15",
            human_reference="data/grasp_labels_v2/ts_cv/human_reference_W0.4.json"), indent=1))
    keys = demo_keys(); print(f"{VERSION}: {len(keys)} demos", flush=True)
    pool = ThreadPoolExecutor(6)
    for n, k in enumerate(keys):
        take, dn = k.split("/"); d = int(dn[4:]); f_out = OUT / "demos" / f"{take}_{dn}.npz"
        if f_out.exists(): continue
        t0, t1 = G.demo_window(take, d); ts = np.arange(t0 - MARGIN_S * 1e9, t1 + MARGIN_S * 1e9, 1e9 / HZ).astype(np.int64); out = dict(t_ns=ts)
        for h, cam in G.CAMS:
            f, tw, cd, tip = R.crop_fn(take, cam); tj, st, ev = E2.jaw_v2(take, cam)
            crops = np.stack(list(pool.map(f, ts)))
            fp = np.concatenate([1 - C.predict(m, crops[b:b + 128][:, None]) for b in range(0, len(ts), 128)])
            chg = np.array([t for t, _ in ev]); last = np.array([chg[chg <= t].max() if (chg <= t).any() else ts[0] for t in ts])
            out[f"{h}_p"] = np.array([fp[(ts >= max(last[i], ts[i] - W * 1e9)) & (ts <= ts[i])].mean() for i in range(len(ts))], np.float32)
            out[f"{h}_closed"] = (st[np.searchsorted(tj, ts).clip(0, len(tj) - 1)] == 0).astype(np.int8)
        np.savez(f_out, version=VERSION, **out)
        if n % 10 == 0: print(f"  {n}/{len(keys)} {k}  L mean {out['L_p'].mean():.2f}  R mean {out['R_p'].mean():.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
