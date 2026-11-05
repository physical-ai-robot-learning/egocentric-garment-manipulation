#!/usr/bin/env python3
"""Check every number in the technical report (report/report_en.tex) against the repository file it comes from.

The report adds no new numbers: each number it prints was taken from one of the project's documents (README.md,
docs/pipeline.md, docs/history.md, docs/results.md, docs/details.md, docs/status_2026-11-0*.md or the archived
report v0.1). This script extracts every number from the body of report/report_en.tex and asserts that it appears, as
a whole number token, in the source file that SOURCES maps it to. A number that is in the report but not in SOURCES fails ("unmapped"); a
mapping that is no longer used fails too ("stale"), so the map stays in step with the text.

What is not a claim and is skipped: the preamble and the bibliography, LaTeX labels / references / citations, file
paths and code names (\\code, \\texttt, \\url), figure file names and sizes, table column specifications, and digits
glued to letters (product and dataset names such as CoTracker3, STS3215, P20, p95). The TikZ figure is layout; it is
checked to print no numbers at all.

Further checks: the Korean report prints no number the English one does not (it is a translation), HandUMI is
attributed to BrikHMP18 in both, and the report states that no success rate is reported and that the pinching trials were not counted.

Run from the repo root:  python analysis/verify_claims.py [--md out.md]     (standard library only)
Exit code 1 if any check fails.
"""
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
README, README_KO = "README.md", "README_ko.md"
RESULTS, DETAILS, DETAILS_KO = "docs/results.md", "docs/details.md", "docs/details_ko.md"
STATUS = "docs/status_2026-11-01.md"
STATUS2 = "docs/status_2026-11-02.md"
STATUS3 = "docs/status_2026-11-05.md"
V01 = "report/archive/report_v0.1_en.tex"
# 2026-11-05: the long parts of the README moved to docs/ (text unchanged); numbers that left the README point there
PIPELINE, HISTORY = "docs/pipeline.md", "docs/history.md"

# number as printed in the report -> the repository file it was taken from
SOURCES = {
    # dates, versions, counts used throughout
    "2026": README, "1": STATUS, "8": V01, "13": README, "16": RESULTS, "20": README, "21": V01, "22": README,
    "23": README, "25": V01, "26": README, "27": README, "28": README, "29": README, "30": STATUS,
    "0": V01, "0.1": V01, "2": README, "3": V01, "4": RESULTS, "5": V01, "6": README, "7": README, "9": STATUS,
    "10": README, "12": V01, "14": RESULTS, "15": PIPELINE, "17": V01, "18": V01, "19": RESULTS,
    # hardware
    "48": README, "336": README, "38": RESULTS, "4.0": README, "1280": V01, "720": V01, "100": PIPELINE, "113": README, "5090": PIPELINE, "4090": STATUS,
    # world frame and calibration
    "32": README, "96": DETAILS, "105": DETAILS, "70": PIPELINE, "89": PIPELINE, "8.4": PIPELINE, "0.98": RESULTS,
    "1.3": RESULTS, "225": PIPELINE, "166": PIPELINE, "243": PIPELINE, "194": PIPELINE, "1.8": RESULTS, "4.8": RESULTS,
    "400": README, "401": README,
    # pose
    "0.10": V01, "0.2": V01, "40": DETAILS, "42": RESULTS, "49": DETAILS, "77": DETAILS, "2.0": V01,
    "60": STATUS, "73": RESULTS, "1000": RESULTS, "0.07": RESULTS, "0.16": RESULTS, "0.25": RESULTS,
    "0.43": RESULTS, "24": V01, "35": STATUS,
    # jaw labels
    "98": PIPELINE, "80": HISTORY,
    # replay track
    "0.06": V01, "50": STATUS, "320": V01, "480": V01, "120": V01, "0.35": README, "33": STATUS, "94": README,
    "1.5": RESULTS, "0.9": RESULTS, "1.1": RESULTS, "5.5": RESULTS, "11.6": RESULTS, "14.5": RESULTS,
    "0.98": RESULTS, "1.04": RESULTS, "0.3": RESULTS, "1.6": RESULTS, "13.7": RESULTS, "14.6": RESULTS,
    "1.7": RESULTS, "1.4": RESULTS, "3.4": RESULTS, "0.5": RESULTS, "37": RESULTS, "79": RESULTS, "95": RESULTS,
    "9.3": RESULTS, "20.6": RESULTS, "6.3": RESULTS, "11.5": RESULTS, "10.3": RESULTS, "17.2": RESULTS,
    "17.4": RESULTS, "29.0": RESULTS, "30.3": RESULTS, "8.1": RESULTS, "12.3": RESULTS, "12.7": RESULTS,
    "12.6": RESULTS, "17.0": V01, "12.1": RESULTS, "14.5": RESULTS, "15.5": RESULTS, "8.6": RESULTS,
    "22.6": RESULTS, "51": RESULTS, "34": RESULTS, "61": RESULTS, "180": V01, "11": README,
    # training data
    "318": RESULTS, "300": RESULTS, "72": DETAILS, "92": DETAILS, "82": DETAILS, "54": DETAILS, "0.8": V01,
    "0.95": V01, "0.05": V01, "0.80": V01, "0.20": V01, "139": RESULTS, "133": RESULTS, "281": README,
    "123": RESULTS, "53": RESULTS, "58": RESULTS, "20.3": V01, "27.8": V01, "12.9": V01, "7.6": V01, "14.8": V01,
    "0.4": STATUS, "4.1": V01, "5.9": V01, "2.3": V01, "3.2": V01, "2.9": V01, "3.5": V01, "3.1": V01,
    "4.9": STATUS, "0.7": V01, "0.6": V01, "44": README, "65": README, "0.93": README, "127": README,
    # policies
    # pinching (status 2026-11-01)
    "74": STATUS, "56": STATUS, "147": STATUS, "197": STATUS, "144": STATUS, "62160": STATUS, "55": STATUS, "46": STATUS, "4.7": STATUS, "31": HISTORY, "6207": STATUS, "19.89": STATUS, "19.13": STATUS,
    "17.1": STATUS, "17.7": STATUS, "150": STATUS, "500": STATUS, "2.6": STATUS,
    # pinching robot runs (status 2026-11-02)
    "17.29": STATUS2, "18.89": STATUS2,
}

# 11-05: flattening after the pinching test (report section sec:flat)
SOURCES.update({
    "250": STATUS3,
    "149": STATUS3,
    "101": STATUS3,
    "0.025": STATUS3,
    "0.08": STATUS3,
    "62": STATUS3,
    "60.89": STATUS3,
    "90": STATUS3,
    "64.36": STATUS3,
    "71.67": STATUS3,
    "72.27": STATUS3,
    "102.36": STATUS3,
    "101.78": STATUS3,
    "82.81": STATUS3,
    "82.42": STATUS3,
    "60.74": STATUS3,
    "59.85": STATUS3,
    "57.64": STATUS3,
    "56.95": STATUS3,
    "53.56": STATUS3,
    "54.16": STATUS3,
    "52.56": STATUS3,
    "52.32": STATUS3,
    "444217": STATUS3,
    "39": STATUS3,
    "1024": STATUS3,
    "343": STATUS3,
    "148": STATUS3,
    "585": STATUS3,
    "543": STATUS3,
    "372": STATUS3,
    "171": STATUS3,
    "0.90": STATUS3,
    "0.78": STATUS3,
    "0.99": STATUS3,
    "0.03": STATUS3,
    "301": STATUS3,
    "45": STATUS3,
    "68.79": STATUS3,
    "68.43": STATUS3,
})

def norm_source(text):
    t = re.sub(r"(\d)\\,(\d{3})(?!\d)", r"\1\2", text)          # 1\,524 -> 1524
    t = re.sub(r"(\d)\{,\}(\d{3})(?!\d)", r"\1\2", t)            # 1{,}524
    t = re.sub(r"(\d),(\d{3})(?![\d])", r"\1\2", t)              # 1,524
    return t

def body_of(tex):
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = tex[tex.index("\\begin{document}"):]
    if "\\begin{thebibliography}" in tex:
        tex = tex[:tex.index("\\begin{thebibliography}")]
    return tex

NUM = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?![0-9])")

def numbers_in(tex):
    """Return (numbers in the text, numbers printed inside TikZ figures)."""
    b = norm_source(body_of(tex))                                  # 62,160 -> 62160 (thousands separators)
    tikz = "".join(re.findall(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", b, re.S))
    b = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", b, flags=re.S)
    for cmd in ("label", "ref", "cite", "code", "texttt", "url", "includegraphics", "bibitem", "hypersetup"):
        b = re.sub(r"\\" + cmd + r"(\[[^\]]*\])?\{[^{}]*\}", " ", b)
    b = re.sub(r"\\begin\{(tabular|table\*?|figure\*?)\}(\[[^\]]*\])?(\{(?:[^{}]|\{[^{}]*\})*\})?", " ", b)
    b = re.sub(r"\\\\\[[^\]]*\]", " ", b)                        # \\[1mm]
    b = re.sub(r"width=[0-9.]+\\(textwidth|columnwidth)", " ", b)
    b = b.replace("\\\\", " ")                                     # LaTeX line breaks
    b = re.sub(r"\\[A-Za-z]+", " ", b)                           # remaining command names (\mm, \times, ...)
    # printed text of the TikZ figure = the {...} after each node name "(name)"; styles and coordinates are layout
    t = " ".join(re.findall(r"\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", tikz))
    t = re.sub(r"\\[A-Za-z]+", " ", t.replace("\\\\", " "))
    return NUM.findall(b), NUM.findall(t)

rows = []
def check(section, claim, ok, source):
    rows.append((section, claim, source, "PASS" if ok else "FAIL"))

en = (R / "report/report_en.tex").read_text()
ko = (R / "report/report_ko.tex").read_text()
nums, tikz_nums = numbers_in(en)
cache = {}
for n in dict.fromkeys(nums):
    src = SOURCES.get(n)
    if src is None:
        check("numbers", f"{n}: unmapped (no source)", False, "-")
        continue
    if src not in cache:
        cache[src] = norm_source((R / src).read_text())
    hit = re.search(r"(?<![0-9.])" + re.escape(n) + r"(?![0-9])(?!\.\d)", cache[src]) is not None
    check("numbers", f"{n} appears in its source", hit, src)
for n in sorted(set(SOURCES) - set(nums)):
    check("map", f"{n}: mapped but not in the report (stale entry)", False, SOURCES[n])
check("figure", f"TikZ figure prints no numbers (found: {sorted(set(tikz_nums))})", not tikz_nums, "report/report_en.tex")

ko_nums, ko_tikz = numbers_in(ko)
extra = sorted(set(ko_nums + ko_tikz) - set(nums))
check("translation", f"Korean report prints no number the English one lacks (extra: {extra})", not extra, "report/report_ko.tex")

flat = " ".join(en.split())
check("attribution", "EN: HandUMI is not our design (BrikHMP18)", "HandUMI is not our design." in flat and "BrikHMP18" in flat, "report/report_en.tex")
check("attribution", "KO: HandUMI attributed to BrikHMP18", "BrikHMP18" in ko and "우리가 설계한 것이 아니다" in ko, "report/report_ko.tex")
check("status", "EN: states that no success rate is reported", "No success rate is reported" in flat, "report/report_en.tex")
check("status", "EN: pinching robot grasps stated as an observation from trials that were not counted", "trials were not counted" in flat and "first grasps observed" in flat.lower(), "report/report_en.tex")

nf = sum(r[-1] == "FAIL" for r in rows)
if "--md" in sys.argv:
    md = ["| section | claim | source | status |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | `{c}` | {d} |" for a, b, c, d in rows]
    md.append(f"\n{len(rows)} claims checked, {nf} failed.")
    Path(sys.argv[sys.argv.index("--md") + 1]).write_text("\n".join(md) + "\n")
for r in rows:
    if r[-1] == "FAIL":
        print("FAIL:", r)
print(f"{len(rows)} claims checked, {nf} failed.")
sys.exit(1 if nf else 0)
