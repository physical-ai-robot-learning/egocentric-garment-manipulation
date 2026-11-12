"""11-11: q_<pack>.csv -> split json for a tag-free take (GOLD gap<0.25 train, goldgap val/test alternating, SILVER silver, REJECT reject); prints the demo keys"""
import json, sys, pandas as pd
qcsv, out = sys.argv[1], sys.argv[2]
q = pd.read_csv(qcsv, dtype={"take": str}); eps = {"train": [], "val": [], "test": [], "silver": [], "reject": []}; gg = []
for _, r in q.iterrows():
    k = f"{r['take']}/demo{int(r['demo']):02d}"; mg = max(r["L_gap_s"], r["R_gap_s"])
    if r["class"] == "GOLD":
        if mg < 0.25: eps["train"].append(k)
        else: gg.append(k)
    else: eps["silver" if r["class"] == "SILVER" else "reject"].append(k)
for i, k in enumerate(gg): eps["val" if i % 2 == 0 else "test"].append(k)
json.dump({"method": "11-11 tag-free take: GOLD gap<0.25 train, goldgap val/test alternating, SILVER silver, REJECT reject", "episodes": eps}, open(out, "w"), indent=1)
print(" ".join(sorted(k for v in eps.values() for k in v)))
print({s: len(v) for s, v in eps.items()}, file=sys.stderr)
