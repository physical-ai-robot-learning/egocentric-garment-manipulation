# grasp

Grasp held / empty labels for the HandUMI jaw close events, and the causal grasp probability p_grasp(t) that the
FLAT250-PGRASP run puts into the masked jaw slots of the observation state
(`garment_pipeline.training.tagrig_common.PGRASP`).

**Status (2026-11-05):** research scripts as they ran on the recording Mac (MPS; torch, torchvision, cv2; the CLIP
variants also need `transformers`). They import each other from this folder, rewrite `sys.argv` before importing
`grasp_label_v0` (which parses arguments at import) and use working folders under `~/Desktop/grasp_*` and the training
workspace. Numbers: [docs/status_2026-11-05.md](../../docs/status_2026-11-05.md) §5.

## Why
The jaw servo is read with torque off, position only: no load or current, and the closed position does not separate
cloth from nothing. A closed jaw does not say whether the cloth is held; the cue has to come from the wrist camera.

## Pipeline (in the order it was built)

| step | script | what |
|---|---|---|
| rule baseline | `grasp_label_v0.py` | per close event: dark share of the jaw-tip disk + tip flow vs. ego flow → held / empty / unknown; helpers the later scripts import (demo window, jaw mask, tip finder) |
| gold | `grasp_gold_ui.py` | labeling page blind to any prediction (keys 1 held, 2 empty, 3 uncertain) → `gold_events.csv` |
| v1 | `grasp_clf_v1.py`, `grasp_label_v1_temporal.py`, `grasp_cnn_v1.py` | CLIP + logistic regression, temporal CLIP + servo features, ResNet-18 on a jaw-tip crop (kept); take-grouped CV |
| next rounds | `grasp_mine_r3.py`, `grasp_apply_v1.py`, `grasp_render_v1.py` | pick likely empties + the uncertain band for labeling, apply v1, review videos |
| v2 events | `grasp_v2_events.py` | tighter jaw hysteresis for this pipeline only (open > 0.30, closed < 0.15, dwell 0.1 s); gold remapped by close time; the training jaw labels are not changed |
| p(t) | `grasp_render_prob.py`, `grasp_ts_cv.py`, `grasp_ts_cv2.py`, `grasp_wrong_clips.py` | causal per-frame p_grasp(t) videos, window-length and crop-size CV, clips of confident CV errors |
| frozen | `grasp_pgrasp_gen.py` | generator `grasp-v2-W0.4`: ResNet-18 on all gold → per-demo p_grasp of both hands at 15 Hz, mean over the last 0.4 s, restarted at each jaw state change |
| side file | `grasp_pgrasp_sidefile.py` | `<set>/meta/pgrasp_<version>.npy` (frames × 2) for an aggregated LeRobot set without rebuilding it; checks per episode that frame 0 sits at the demo start |

The frozen generator is trained on all gold events, so its output on those events is in-sample.
