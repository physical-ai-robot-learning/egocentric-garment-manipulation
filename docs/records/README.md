# records

> Paths in this record are those of its date; see [docs/refactor_2026-11.md](../refactor_2026-11.md) for the current ones.

Data records kept with the code that produced or used them.

| folder | what | used by |
|---|---|---|
| `jaw_kpt_labels_20261022_105200/` | jaw-tip keypoint labels (left / right) of the 10-22 take, made with `scripts/pose/jaw_kpt_label.py` (draft + browser labelling) | `scripts/pose/jaw_kpt_train.py` (YOLO-pose models `configs/calibration/jaw_kpt_*_v00N.pt`) |

`jaw_kpt_labels_20261022_105200/` moved from `data/` in the 11-01 layout refactor.
