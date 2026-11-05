#!/usr/bin/env python3
"""[2026-11-05] Grasp held / empty v1-CNN: ImageNet ResNet-18 fine-tuned on jaw-ROI crops (the v1 tip finder, 256 px -> 224) of the user's
gold events; 4 frames per event at close + {0.3, 0.5, 0.7, 0.9} s (clipped inside the closed interval), event prob = mean over frames.
5-fold GROUPED BY TAKE CV.  Then a fold-internal stack with the v1 motor features (log closed duration etc.).
usage: grasp_cnn_v1.py OUT_DIR"""
import pathlib, sys
import cv2, numpy as np, torch, torchvision
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
OUT = pathlib.Path(sys.argv[1]).expanduser(); sys.argv = sys.argv[:2]
import grasp_label_v0 as G, grasp_label_v1_temporal as V  # noqa: E402

DTS = (0.3, 0.5, 0.7, 0.9); DEV = "mps"; EPOCHS = 8; SEED = 0
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def crops():
    f = OUT / "cnn_crops.npz"
    if f.exists(): z = np.load(f, allow_pickle=True); return z["X"], z["y"], z["take"], z["ids"]
    E = V.gold(); tips = {}; X, y, tk, ids = [], [], [], []
    for e in E:
        take, cam, J, te, tn = V.event_times(e); cd = G.TR / take / cam; tw = np.load(cd / "t_ns.npy")
        if (take, cam) not in tips: tips[(take, cam)] = G.jaw_mask_and_tip(cd, tw, J)[1]
        tip = tips[(take, cam)]; cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H
        x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464)); fr = []
        for dt in DTS:
            t = te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)); i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
            im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg")); im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
            fr.append(cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224)))
        X.append(np.stack(fr)); y.append(int(e["gold"] == "empty")); tk.append(take); ids.append(e["id"])
    X, y, tk, ids = np.stack(X), np.array(y), np.array(tk), np.array(ids); np.savez(f, X=X, y=y, take=tk, ids=ids); return X, y, tk, ids


def to_t(a):   # (n, 224, 224, 3) uint8 -> normalized NCHW
    return torch.tensor(np.ascontiguousarray(((a.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)))


def train_fold(Xtr, ytr):
    torch.manual_seed(SEED); m = torchvision.models.resnet18(); m.load_state_dict(torch.load(pathlib.Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    m.fc = torch.nn.Linear(512, 1); m = m.to(DEV)
    imgs = Xtr.reshape(-1, 224, 224, 3); lab = np.repeat(ytr, Xtr.shape[1]).astype(np.float32)
    pw = torch.tensor((1 - lab.mean()) / max(lab.mean(), 1e-3), device=DEV); opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * int(np.ceil(len(imgs) / 32))); rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        m.train()
        for b in np.array_split(rng.permutation(len(imgs)), int(np.ceil(len(imgs) / 32))):
            a = imgs[b].copy()
            if rng.random() < 0.5: a = a[:, :, ::-1]                       # horizontal flip
            x = to_t(a).to(DEV); x = x * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.2, 0.2))   # brightness/contrast jitter
            l = torch.nn.functional.binary_cross_entropy_with_logits(m(x).squeeze(1), torch.tensor(lab[b], device=DEV), pos_weight=pw)
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval(); p = torch.sigmoid(m(to_t(X.reshape(-1, 224, 224, 3)).to(DEV)).squeeze(1)).cpu().numpy()
    return p.reshape(len(X), -1).mean(1)


def main():
    X, y, tk, ids = crops(); takes = np.unique(tk); rng = np.random.default_rng(1105); rng.shuffle(takes)
    folds = np.array_split(takes, 5); p = np.zeros(len(y))
    for k, ft in enumerate(folds):
        te = np.isin(tk, ft); m = train_fold(X[~te], y[~te]); p[te] = predict(m, X[te])
        print(f"fold {k}: {te.sum()} events ({y[te].sum()} empty)", flush=True)
    np.savez(OUT / "cnn_cv.npz", p=p, y=y, take=tk, ids=ids)
    # stack with the motor model: average of probabilities (no extra fitting)
    Z = dict(np.load(OUT / "v1_feats.npz", allow_pickle=True)); pos = {i: n for n, i in enumerate(Z["ids"])}
    Zs = dict(y=np.asarray(Z["y"]), take=np.asarray(Z["take"]), mot=Z["mot"], clip=Z["clip"]); pm_all = V.cv_pred(Zs, "mot", lam=10.0)
    pm = np.array([pm_all[pos[i]] for i in ids])
    for name, q in (("cnn", p), ("cnn+motor avg", (p + pm) / 2)):
        for lo, hi in ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)):
            o, cov = V.f1s(y, q, lo, hi); print(f"{name:14s} band[{lo},{hi}] coverage {cov:.2f}  empty P/R/F1 {o['empty']}  held P/R/F1 {o['held']}", flush=True)


if __name__ == "__main__":
    main()
