"""10-27 tighter idle trim for the flattening takes (user: drop the waiting start and the still end): inside the current
reports/idle_trim.csv window, keep [first moving - MARGIN, last moving + MARGIN], moving = max over both arms of the 0.3 s smoothed
fused TCP speed (stitch/qa/fus_Cg_g) > V_MOVE. Only tightens (never widens); backup idle_trim.csv.bak_tight_<time> first.
    python tighten_trim.py <idle_audit.csv from idle_audit.py> [MARGIN=0.3]"""
import csv, shutil, sys, time
from pathlib import Path
CSV = Path.home() / "laundry_dit_training/reports/idle_trim.csv"; MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
aud = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(sys.argv[1]))}
shutil.copy2(CSV, CSV.with_suffix(f".csv.bak_tight_{time.strftime('%m%d_%H%M%S')}"))
rows = list(csv.DictReader(open(CSV))); fields = list(rows[0].keys()); n = 0
for r in rows:
    a = aud.get((r["take"], int(r["demo"])))
    if not a or a["head_s"] in ("", "nan"): continue
    cut_h = max(0.0, float(a["head_s"]) - MARGIN); cut_t = max(0.0, float(a["tail_s"]) - MARGIN)
    if cut_h < 0.05 and cut_t < 0.05: continue
    ks, ke, dl = float(r["keep_start_s"]) + cut_h, float(r["keep_end_s"]) - cut_t, float(r["demo_len_s"])
    r.update(keep_start_s=round(ks, 2), keep_end_s=round(ke, 2), cut_head_s=round(ks, 1), cut_tail_s=round(dl - ke, 1), kept_s=round(ke - ks, 1)); n += 1
with open(CSV, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("tightened", n, "demos")
