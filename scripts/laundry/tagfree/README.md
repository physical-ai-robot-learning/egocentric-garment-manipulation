# tagfree — jaw-tip poses without fiducial markers (2026-11-11/12)

**Question.** The policy acts in relative motion with a task-anchor state, so its labels do not need absolute marker
poses. Can the pose chain run with no AprilTags at all, if each demonstration starts from a known rest?

**Method.** Both hand units rest on black L tapes at the start of every demonstration. The start pose comes from the
tapes and the resting units in the fixed camera's depth (`v2b/tagfree_seed.py`), the heading from the wrist IMU
(Mahony filter) anchored at that rest (`v2b/tagfree_yaw.py`); the rest of the chain is unchanged — point tracking,
rigid point solver, gates, fusion, labels (`v2b/cotrack/`, `v2b/tagfree_prep.py`, `v2b/tagfree_derive.py`).

**Checks.**
- `chain/tagfree_vs_tag.py` — a tagged take processed as if it had no tags, compared with its own tag measurements:
  jaw-tip error left 17 mm, right 9–14 mm, constant over the demonstration (after two seeding errors were fixed).
- `chain/a0_tagfree_test.py` — first marker-free label test on the tagged flattening data (heading from motion vs tags).
- `chain/cam_drift_icp.py`, `chain/l_tape_check.py` — camera motion between takes from the static depth scene; tape
  positions as a drift check (used after the 11-12 camera re-aim).
- `chain/render_tagfree.py`, `chain/tagfree_split.py` — QA video per demonstration; GOLD / SILVER / REJECT → splits.

These files are a snapshot of the working code of 12 November, not yet folded into `src/laundry_tracking`; they import
the original working tree and are here to document the method, not to be run as they are.
Example output: [media/tagfree_20261112_135910_d02.gif](../../../media/tagfree_20261112_135910_d02.gif). Numbers:
[docs/status_2026-11-12.md](../../../docs/status_2026-11-12.md) §4.
