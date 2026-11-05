"""Flatten-task dataset build scripts of 11-02 .. 11-04 (dataset stage, around tag_rig_unsplit): CT-gap and idle audits, the
GOLD / SILVER / REJECT split builder, per-take builds with a spread holdout, the polling build-and-upload loop and the fusion QA
video renderer. Each module is a CLI (`python -m laundry_dataset.builds.<module> --help`); session inputs (key lists, quality
tables, packs) are explicit arguments, outputs go to --work-dir.
"""
