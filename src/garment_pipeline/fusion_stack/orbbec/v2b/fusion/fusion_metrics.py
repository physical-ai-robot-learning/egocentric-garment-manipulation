"""10-24 per-arm fusion ablation table from stitch/qa/<var> runs + stitch/qa/eval_<tag>.csv (qa_eval hidden-tag e32).
  raw_cov     CoTracker conf>0 fraction of the trimmed window (Orbbec frames);  cont_cov = window fraction inside the run's time span
  sup_cov     nodes within 0.25 s of any used observation (production run h-1; raw var = raw_cov);  gap_s = longest unsupported span
  e32         n-weighted mean over demos of the qa_eval hidden-block e32 median / p90 [mm]
  abs         hidden-block absolute |p - p_tag| (>= 2 tags) median / p90 [mm];  far10 = rate > 10 cm;  burst_s = longest > 10 cm run
  gapbr       gap-bridge error: hidden >= 1-tag frames inside CoTracker gaps >= 0.2 s (+-0.15 s, seed start excluded), |e - demo median e| [mm] max / med
    python fusion_metrics.py <ctpack> <eval csv tag> VAR1,VAR2,..."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
from garment_pipeline.fusion_stack import core as C  # noqa
from garment_pipeline.fusion_stack.stitch.wdo import hidden_mask  # noqa
FD = C.FD


def runs_s(b, t):
    x = np.r_[0, b.astype(int), 0]; s, e = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    return [(t[j - 1] - t[i]) / 1e9 + 0.067 for i, j in zip(s, e)]


def interp(t, p, tq): return np.stack([np.interp(tq, t, p[:, a]) for a in range(3)], 1)


def main(pack, tag, variants):
    ev = pd.read_csv(ROOT / f"stitch/qa/eval_{tag}.csv"); rows = []
    for d in sorted(Path(pack).iterdir()):
        if not (d / "solve2_left.npz").exists(): continue
        meta = json.loads((d / "meta.json").read_text()); take, demo = meta["take"], int(meta["demo"]); W = {w["demo"]: w for w in FD.trims()[take]}[demo]
        for arm in ("left", "right"):
            D = FD.load_arm(take, arm, with_slam=False); t = D["t"]; inw = (t >= W["w0"]) & (t <= W["w1"])
            z = np.load(d / f"solve2_{arm}.npz"); tct = z["t"].astype(np.int64); ok = z["conf"] > 0
            okw = ok[(tct >= W["w0"]) & (tct <= W["w1"])]; tw = tct[(tct >= W["w0"]) & (tct <= W["w1"])]
            gaps = runs_s(~okw, tw); raw_cov = float(okw.mean())
            gstart = [tw[i] for i in np.flatnonzero(np.diff(np.r_[1, (~okw).astype(int)]) == -1)]  # rejoin times (first ok after a gap)
            gl = runs_s(~okw, tw); rej_t = [g for g, L in zip(gstart, gl) if L >= 0.2] if len(gl) == len(gstart) else gstart
            for var in variants:
                f = ROOT / f"stitch/qa/{var}/{take}_{arm}_d{demo:02d}_h-1.npz"
                if not f.exists(): continue
                r = np.load(f); tr, pr = r["t_ns"], r["p_B"]; tn = np.arange(W["w0"], W["w1"], int(5e7))
                cont = float(((tn >= tr[0]) & (tn <= tr[-1])).mean())
                if "info" in r:
                    m_ = (tr >= W["w0"]) & (tr <= W["w1"]); sup = r["info"][m_]; sup_cov = float(sup.mean()); sgap = max(runs_s(~sup, tr[m_]) or [0.0])
                else: sup_cov, sgap = raw_cov, max(gaps or [0.0])
                rej = []                                   # gap-bridge error: hidden >= 1-tag frames inside CT gaps >= 0.2 s (+-0.15 s), bias-removed
                ing = np.zeros(len(t), bool); x_ = np.r_[0, (~okw).astype(int), 0]
                for i_, j_ in zip(np.flatnonzero(np.diff(x_) == 1), np.flatnonzero(np.diff(x_) == -1)):
                    if (tw[j_ - 1] - tw[i_]) / 1e9 + 0.067 >= 0.2 and tw[i_] > tct[np.flatnonzero(ok)[0]]: ing |= (t >= tw[i_] - 1.5e8) & (t <= tw[j_ - 1] + 1.5e8)
                for off in (0, 1, 2):
                    f3 = ROOT / f"stitch/qa/{var}/{take}_{arm}_d{demo:02d}_h{off}.npz"
                    if not f3.exists(): continue
                    r3 = np.load(f3); hm = hidden_mask(t, off) & inw & (D["ntags"] >= 1) & (t >= r3["t_ns"][0]) & (t <= r3["t_ns"][-1])
                    if not (hm & ing).any(): continue
                    ev_ = interp(r3["t_ns"], r3["p_B"], t[hm]) - D["p_tag"][hm]; b_ = np.median(ev_[~ing[hm]], 0) if (~ing[hm]).sum() > 10 else 0
                    rej += list(np.linalg.norm(ev_[ing[hm]] - b_, axis=1) * 1000)
                A = []; far_b = []
                for off in (0, 1, 2):
                    f2 = ROOT / f"stitch/qa/{var}/{take}_{arm}_d{demo:02d}_h{off}.npz"
                    if not f2.exists(): continue
                    r2 = np.load(f2); hid = hidden_mask(t, off) & inw & (D["ntags"] >= 2)
                    if not hid.any(): continue
                    e = np.linalg.norm(interp(r2["t_ns"], r2["p_B"], t[hid]) - D["p_tag"][hid], axis=1); A.append(e); bad = np.zeros(len(t), bool); bad[np.flatnonzero(hid)[e > 0.10]] = True
                    far_b += runs_s(bad[hid], t[hid])
                A = np.concatenate(A) if A else np.zeros(0)
                e_ = ev[(ev["var"] == var) & (ev["arm"] == arm) & (ev["demo"] == demo) & (ev["take"].astype(str) == take)]
                rows.append(dict(var=var, take=take, arm=arm, demo=demo, raw_cov=raw_cov, cont_cov=cont, sup_cov=sup_cov, gap_s=sgap,
                                 e32_med=float(e_["e32_med"].iloc[0]) if len(e_) else np.nan, e32_p90=float(e_["e32_p90"].iloc[0]) if len(e_) else np.nan, n=int(e_["n"].iloc[0]) if len(e_) else 0,
                                 abs_med=np.median(A) * 1000 if len(A) else np.nan, abs_p90=np.percentile(A, 90) * 1000 if len(A) else np.nan, far10=float((A > 0.1).mean()) if len(A) else np.nan, burst_s=max(far_b or [0.0]), rejoin=max(rej or [np.nan]), gapbr_med=np.median(rej) if rej else np.nan, n_gap=len(rej), n_abs=len(A)))
    df = pd.DataFrame(rows); df.to_csv(ROOT / f"stitch/qa/fusion_metrics_{tag}.csv", index=False)
    def wavg(x, w):
        m = np.isfinite(x); return np.average(x[m], weights=w[m]) if m.any() else np.nan
    def agg(g):
        w = g["n"].clip(lower=1); wa = g["n_abs"].clip(lower=1)
        return pd.Series(dict(raw_cov=g["raw_cov"].mean(), cont_cov=g["cont_cov"].mean(), sup_cov=g["sup_cov"].mean(), gap_s=g["gap_s"].max(),
                              e32_med=wavg(g["e32_med"], w), e32_p90=wavg(g["e32_p90"], w),
                              abs_med=wavg(g["abs_med"], wa), abs_p90=wavg(g["abs_p90"], wa), far10=wavg(g["far10"], wa),
                              burst_s=g["burst_s"].max(), gapbr_max=g["rejoin"].max(), gapbr_med=g["gapbr_med"].median(), n_gap=g["n_gap"].sum()))
    out = df.groupby(["arm", "var"], sort=False).apply(agg).reset_index()
    pd.set_option("display.width", 220); print(out.round(3).to_string(index=False)); return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
