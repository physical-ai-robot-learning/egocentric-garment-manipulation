"""Train a jaw-tip keypoint model per hand from scripts/pose/jaw_kpt_label.py labels (2026-10-22). One object per image ("gripper")
with 2 keypoints (A = image-left jaw tip, B = image-right jaw tip); the box spans both jaws down to the bottom edge.
Split by frame index (every 7th frame -> val) so near-duplicate neighbours do not leak much. Reports val tip error in px.
    .venv/bin/python scripts/pose/jaw_kpt_train.py datasets/jaw_kpt/20261022_105200 --hand left [--epochs 120 --model yolov8n-pose.pt]
-> datasets/jaw_kpt/<take>/<hand>/yolo/ (dataset), runs in datasets/jaw_kpt/<take>/<hand>/train/, best weights copied to
   configs/calibration/jaw_kpt_<hand>_v001.pt"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def export(D: Path, hand: str) -> Path:
    L = json.loads((D / hand / "labels.json").read_text())["labels"]; Y = D / hand / "yolo"
    if Y.exists(): shutil.rmtree(Y)
    for sp in ("train", "val"): (Y / "images" / sp).mkdir(parents=True); (Y / "labels" / sp).mkdir(parents=True)
    for k, l in enumerate(L):
        sp = "val" if k % 7 == 3 else "train"; im = cv2.imread(str(D / hand / "frames" / l["file"])); h, w = im.shape[:2]
        shutil.copy(D / hand / "frames" / l["file"], Y / "images" / sp / l["file"])
        if not l["visible"] or not l["tips"]:
            (Y / "labels" / sp / (Path(l["file"]).stem + ".txt")).write_text(""); continue
        (xa, ya), (xb, yb) = l["tips"]; x0 = max(0, min(xa, xb) - 0.12 * w); x1 = min(w, max(xa, xb) + 0.12 * w); y0 = max(0, min(ya, yb) - 0.05 * h); y1 = h
        box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
        kp = [xa / w, ya / h, 2, xb / w, yb / h, 2]
        (Y / "labels" / sp / (Path(l["file"]).stem + ".txt")).write_text("0 " + " ".join(f"{v:.6f}" for v in box + kp) + "\n")
    (Y / "data.yaml").write_text(f"path: {Y}\ntrain: images/train\nval: images/val\nnames:\n  0: gripper\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n")
    return Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--hand", required=True, choices=["left", "right"]); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--model", default="yolov8n-pose.pt"); ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--tag", default="v001"); a = ap.parse_args()
    from ultralytics import YOLO
    D = Path(a.dir).resolve(); Y = export(D, a.hand)
    m = YOLO(a.model)
    m.train(data=str(Y / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=16, device="mps", project=str(D / a.hand / "train"), name="run",
            exist_ok=True, fliplr=0.5, degrees=8, translate=0.05, scale=0.2, mosaic=0.0, hsv_v=0.4, verbose=False, plots=False)
    best = D / a.hand / "train/run/weights/best.pt"; out = ROOT / f"configs/calibration/jaw_kpt_{a.hand}_{a.tag}.pt"; shutil.copy(best, out)
    # val tip error in px (original crop resolution)
    net = YOLO(str(out)); L = json.loads((D / a.hand / "labels.json").read_text())["labels"]; err = []; miss = 0
    for k, l in enumerate(L):
        if k % 7 != 3 or not l["visible"]: continue
        r = net.predict(str(D / a.hand / "frames" / l["file"]), verbose=False, device="mps")[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0: miss += 1; continue
        j = int(np.argmax(r.boxes.conf.cpu().numpy())); kp = r.keypoints.xy[j].cpu().numpy()
        gt = np.array(l["tips"]); kp = kp[np.argsort(kp[:, 0])]
        err += list(np.linalg.norm(kp - gt, axis=1))
    err = np.array(err); print(f"[jaw kpt {a.hand}] val tips {len(err)} | error median {np.median(err):.1f} px, p90 {np.percentile(err, 90):.1f} px | no detection {miss} -> {out}")


if __name__ == "__main__":
    main()
