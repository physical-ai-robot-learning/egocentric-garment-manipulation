# report

The technical report of the project.

- `report_en.pdf` — English, status of 26 November 2026 (Section XIII, 6–26 November: deploy latency matching, X-VLA
  fine-tune and its offline comparison with the DiT on the same held-out data, tag-free collection, the robot runs of
  16–18 November, the wrist left/right probe, the wrist-primary flow policy, the robot table offset, the 34-dimensional-state DiT,
  the repaired X-VLA, the robot tests of 23 November, where the DiT grasps, the high-resolution wrist DiT, the scripted grasp test and the data restart of 26 November; Section XII,
  flattening after the pinching test, the FLAT-PGRASP policies and the robot runs of 7 November as an interim
  observation).
- `report_ko.pdf` — Korean, status of 2 November 2026 (not updated since).
- `archive/` — Technical Report v0.1 (27 October, "Egocentric Laundry Folding"), v0 (21 October) and the short technical
  note, as PDF.

**Checking the numbers.** Every number in the English report is mapped to the document it was taken from (the project
README, `docs/pipeline.md`, `docs/history.md`, `docs/results.md`, `docs/details.md` and the dated status notes in
`docs/`) and checked by a script: 319 claims, 0 failed on 26 November. The LaTeX source and the checking script are part
of the non-public code.

**Paths named in the reports.** The current reports name the code paths of the layout of 5 November 2026
(`src/laundry_*`, `scripts/laundry/`, `configs/laundry/`); the archived reports were written in the project's former
working repository, `egocentric-laundry-folding`, and name its paths of that time. These paths refer to the code, which
is not part of this public version.
