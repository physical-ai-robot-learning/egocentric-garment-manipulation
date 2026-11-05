#!/usr/bin/env python3
"""[2026-11-05] Grasp event segmentation v2: tighter jaw hysteresis on the motor g_cont (0 closed .. 1 open) for the GRASP pipeline only
(umi_jaw.npz and the training jaw labels are NOT touched).  v1 events came from umi_jaw.npz state (open > 0.55, closed < 0.35): 202 of
3554 closed runs hid a partial re-open to ~0.48 for ~1 s.  v2: open > HI = 0.30, closed < LO = 0.15, dwell 0.1 s (jaw_from_motor.hysteresis).
Steps: all v2 events of the 343 flatten demos -> before/after stats -> human gold remapped by CLOSE TIMESTAMP (|dt| <= 0.3 s) ->
the v1 CNN recipe re-trained on v2 crops, same take-grouped 5 folds -> P/R/F1 next to v1.
usage: grasp_v2_events.py OUT_DIR"""
import collections, csv, json, pathlib, sys
import cv2, numpy as np
H = pathlib.Path.home(); OUT = pathlib.Path(sys.argv[1]).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(H / "Desktop/grasp_label_v1")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(H / "ego_collector/scripts"))
import grasp_label_v0 as G, grasp_label_v1_temporal as V, grasp_cnn_v1 as C  # noqa: E402
from jaw_from_motor import hysteresis  # noqa: E402
HI, LO, DWELL, MATCH_S = 0.30, 0.15, 0.1, 0.3
V1CSV = H / "laundry_dit_training/data/grasp_labels_v1/grasp_events_v1.csv"
log = lambda s: (print(s, flush=True), open(OUT / "report.txt", "a").write(s + "\n"))


def demo_keys():
    S = pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad")
    keys = set()
    for f in sorted(S.glob("flat1103n_*_S03G_split.json")) + sorted(S.glob("flat1104_*_S03G_split.json")):
        for sp, ks in json.loads(f.read_text())["episodes"].items():
            if sp != "reject": keys |= set(ks)
    return sorted(keys)


_jaw = {}
def jaw_v2(take, cam):
    if (take, cam) not in _jaw:
        J = np.load(G.TR / take / cam / "umi_jaw.npz"); st, ev = hysteresis(J["t_ns"], J["g_cont"], HI, LO, DWELL); _jaw[(take, cam)] = (J["t_ns"], st, ev)
    return _jaw[(take, cam)]


def events_v2():
    rows = []
    for k in demo_keys():
        take, dn = k.split("/"); d = int(dn[4:]); t0, t1 = G.demo_window(take, d)
        for h, cam in G.CAMS:
            tj, st, ev = jaw_v2(take, cam); n = 0
            for i, (te, kind) in enumerate(ev):
                if kind != "close": continue
                tn = next((tb for tb, kb in ev[i + 1:] if kb == "open"), int(tj[-1]))
                if tn < t0 or te > t1: continue
                n += 1
                rows.append(dict(id=f"{take}_d{d + 1:02d}_{h}{n}", demo=k, take=take, cam=cam, hand=h, event=f"{h}{n}", t_close_ns=te, t_open_ns=tn,
                                 closed_s=round((tn - te) / 1e9, 2), rest=int(te < t0 + G.EDGE_S * 1e9 or tn > t1 - G.EDGE_S * 1e9)))
    return rows


def main():
    (OUT / "report.txt").write_text("")
    new = events_v2(); old = list(csv.DictReader(open(V1CSV)))
    for r in old: r["t_close_ns"], r["t_open_ns"], r["rest"] = int(r["t_close_ns"]), int(r["t_open_ns"]), int(r["rest"])
    by = collections.defaultdict(list)
    for r in new: by[(r["demo"], r["hand"])].append(r)
    # each v1 event -> the v2 closes inside its closed interval (split if > 1)
    nsplit = 0
    for r in old:
        inside = [x for x in by[(r["demo"], r["hand"])] if r["t_close_ns"] - MATCH_S * 1e9 <= x["t_close_ns"] < r["t_open_ns"]]
        r["n_v2"] = len(inside); nsplit += len(inside) > 1
    on, nn = [r for r in old if not r["rest"]], [r for r in new if not r["rest"]]
    log(f"=== segmentation v1 (open>0.55 / closed<0.35) vs v2 (open>{HI} / closed<{LO}), 343 demos ===")
    log(f"events total      v1 {len(old)}  v2 {len(new)}   non-rest v1 {len(on)}  v2 {len(nn)}")
    log(f"events per demo   v1 {len(on) / 343:.2f}  v2 {len(nn) / 343:.2f}  (non-rest)")
    log(f"v1 events split by a partial re-open: {nsplit}  (-> {sum(r['n_v2'] for r in old if r['n_v2'] > 1)} v2 events); v1 events with no v2 close: {sum(r['n_v2'] == 0 for r in old)}")
    cd = np.array([r["closed_s"] for r in nn]); co = np.array([float(r["closed_s"]) for r in on])
    log(f"closed duration s median/p90   v1 {np.median(co):.2f}/{np.quantile(co, .9):.2f}   v2 {np.median(cd):.2f}/{np.quantile(cd, .9):.2f}")
    # gold remap by close timestamp
    gold = V.gold(); ev2 = {f"{r['video'].rsplit('_', 1)[0]}_{r['event']}": r["label"] for r in csv.DictReader(open(H / "Desktop/grasp_gold_r2/events.csv"))}
    oldid = {f"{r['demo'].split('/')[0]}_d{int(r['demo'].split('/')[1][4:]) + 1:02d}_{r['event']}": r for r in old}
    G2, miss = [], 0
    for g in gold:
        o = oldid.get(g["id"])
        if o is None: miss += 1; continue
        cand = [x for x in by[(o["demo"], o["hand"])] if abs(x["t_close_ns"] - o["t_close_ns"]) <= MATCH_S * 1e9]
        if not cand: miss += 1; continue
        x = min(cand, key=lambda x: abs(x["t_close_ns"] - o["t_close_ns"]))
        G2.append(dict(x, gold=g["gold"], v1_id=g["id"], split=int(o["n_v2"] > 1), random=int(ev2.get(g["id"]) == "unknown"),
                       dt_ms=round((x["t_close_ns"] - o["t_close_ns"]) / 1e6)))
    log(f"gold held/empty {len(gold)} -> remapped {len(G2)} ({len(G2) / len(gold):.1%}), unmapped {miss}; of remapped, in a split v1 event {sum(g['split'] for g in G2)}")
    log(f"close-time shift v1->v2 ms median {np.median([g['dt_ms'] for g in G2]):.0f}  p90 {np.quantile([abs(g['dt_ms']) for g in G2], .9):.0f}")
    for nm, s in (("all", G2), ("random", [g for g in G2 if g["random"]])):
        c = collections.Counter(g["gold"] for g in s); log(f"gold {nm:6s}: held {c['held']}  empty {c['empty']}  (empty {c['empty'] / max(len(s), 1):.1%})")
    with open(OUT / "events_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(new[0])); w.writeheader(); w.writerows(new)
    with open(OUT / "gold_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(G2[0])); w.writeheader(); w.writerows(G2)
    # crops on the v2 intervals + CNN CV with the v1 fold assignment
    tips, X = {}, []
    for g in G2:
        cdir = G.TR / g["take"] / g["cam"]; tw = np.load(cdir / "t_ns.npy")
        if (g["take"], g["cam"]) not in tips: tips[(g["take"], g["cam"])] = G.jaw_mask_and_tip(cdir, tw, np.load(cdir / "umi_jaw.npz"))[1]
        tip = tips[(g["take"], g["cam"])]; cx, cy = tip[0] * 1280 // G.W, tip[1] * 720 // G.H
        x0 = int(np.clip(cx - 128, 0, 1024)); y0 = int(np.clip(cy - 153, 0, 464)); te, tn = g["t_close_ns"], g["t_open_ns"]; fr = []
        for dt in C.DTS:
            t = te + min(dt * 1e9, max(tn - te - 0.05e9, 0.1e9)); i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1))
            im = cv2.imread(str(cdir / f"wrist_{i:05d}.jpg")); im = np.zeros((720, 1280, 3), np.uint8) if im is None else im
            fr.append(cv2.resize(cv2.cvtColor(im[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_BGR2RGB), (224, 224)))
        X.append(np.stack(fr))
    X = np.stack(X); y = np.array([g["gold"] == "empty" for g in G2], int); tk = np.array([g["take"] for g in G2])
    takes = np.unique(tk); rng = np.random.default_rng(1105); rng.shuffle(takes); p = np.zeros(len(y))
    for ft in np.array_split(takes, 5):
        te_ = np.isin(tk, ft); m = C.train_fold(X[~te_], y[~te_]); p[te_] = C.predict(m, X[te_])
    np.savez(OUT / "cnn_cv_v2.npz", p=p, y=y, take=tk, ids=np.array([g["id"] for g in G2]), v1_ids=np.array([g["v1_id"] for g in G2]),
             random=np.array([g["random"] for g in G2]), split=np.array([g["split"] for g in G2]))
    z1 = np.load(H / "Desktop/grasp_label_v1/cnn_cv.npz", allow_pickle=True); p1 = dict(zip(z1["ids"].tolist(), z1["p"].tolist()))
    vid = [g["v1_id"] for g in G2]; q1 = np.array([p1[i] for i in vid]); rnd = np.array([g["random"] for g in G2], bool)
    log("=== CNN take-grouped 5-fold CV, same remapped gold events (p = P(empty)) ===")
    for nm, s in (("all gold", np.ones(len(y), bool)), ("random subset", rnd)):
        for tag, q in (("v1 crops", q1), ("v2 crops", p)):
            o, cov = V.f1s(y[s], q[s], 0.5, 0.5)
            pg = 1 - q[s]; hh = pg > 0.9; ee = pg < 0.2
            log(f"{nm:13s} {tag}: @0.5 held P/R/F1 {o['held']}  empty P/R/F1 {o['empty']}  | rule 0.9/0.2: held {hh.sum()} prec {1 - (hh & (y[s] == 1)).sum() / max(hh.sum(), 1):.3f}"
                f"  empty {ee.sum()} prec {1 - (ee & (y[s] == 0)).sum() / max(ee.sum(), 1):.3f}  empty-recall {(ee & (y[s] == 1)).sum() / max((y[s] == 1).sum(), 1):.2f}  unknown {(~hh & ~ee).mean():.0%}")


if __name__ == "__main__":
    main()
