# grasp

Grasp held / empty labels for the HandUMI jaw close events, and the causal grasp probability `p_grasp(t)` that the
FLAT-PGRASP run feeds into the masked jaw slots of the state (`laundry_training.env_options.PGRASP`).

**Status (2026-11-05):** research scripts, copied as they ran from `~/laundry_dit_training/datasets/`. They are **not yet
in OA style**: they edit `sys.path`, import each other as top-level modules, rewrite `sys.argv` before importing
`grasp_label_v0` (which parses its arguments at import) and hard-code working folders under `~/Desktop/grasp_*` and
`~/laundry_dit_training/data/grasp_labels_v*`. Run them from this folder with the Mac MPS environment (torch, torchvision,
cv2; `grasp_clf_v1` / `grasp_label_v1_temporal` also need `transformers` for CLIP). Convert to a `src/laundry_grasp`
package once the recipe stops changing.

## Why
The jaw servo is read torque-off, position only: no load or current, and the closed position does not separate cloth from
nothing (the right jaw closes to 5263–5279 ticks every time). So "jaw closed" does not say whether the cloth is held. The
cue has to come from the wrist camera.

## Pipeline (in the order it was built)

| step | script | what |
|---|---|---|
| v0 rule | `grasp_label_v0.py` | per close event: jaw-tip disk dark share + tip flow vs ego flow → held / empty / unknown; the shared helpers (demo window, jaw mask / tip finder) the later scripts import |
| gold | `grasp_gold_ui.py` | blind labeling page (127.0.0.1:PORT; keys 1 held, 2 empty, 3 uncertain) → `gold_events.csv` |
| v1 | `grasp_clf_v1.py`, `grasp_label_v1_temporal.py`, `grasp_cnn_v1.py` | CLIP + logistic regression, temporal CLIP + motor features, jaw-ROI ResNet-18 (chosen); take-grouped CV |
| mining | `grasp_mine_r3.py`, `grasp_apply_v1.py`, `grasp_render_v1.py` | next labeling round (likely empties + uncertain band), apply v1 with the 0.9 / 0.2 rule, review videos |
| v2 events | `grasp_v2_events.py` | tighter jaw hysteresis for this pipeline only (open > 0.30, closed < 0.15, dwell 0.1 s; v1 hid 148 partial re-opens); gold remapped by close time; `umi_jaw.npz` and the training jaw labels are **not** changed |
| p(t) | `grasp_render_prob.py`, `grasp_ts_cv.py`, `grasp_ts_cv2.py`, `grasp_wrong_clips.py` | causal frame-level p_grasp(t) videos (open frames scored too), window sweep CV, ROI-size CV, clips of confident CV errors |
| frozen | `grasp_pgrasp_gen.py` | **`grasp-v2-W0.4`**: ResNet-18 on all gold (+ `gold_overrides_v2.json`; slip excluded, loose = not held, open-approach negatives) → per-demo `t_ns, L_p, R_p` at 15 Hz, mean over `[max(last v2 jaw change, t − 0.4 s), t]` |
| side file | `grasp_pgrasp_sidefile.py` | `<set>/meta/pgrasp_<version>.npy` (total_frames × 2, [L, R]) for an aggregated LeRobot set without rebuilding it; checks per episode that frame 0 sits at the demo window start; zero-order hold on the wrist source time |

## Numbers (11-05)
- Gold: 585 events over four rounds, 543 remapped to v2 events (held 372, empty 171).
- Event CNN, take-grouped 5-fold, v2 crops: held P/R/F1 0.90/0.90/0.90, empty 0.79/0.77/0.78.
- Causal p_grasp(t), W = 0.4 s: held median 0.99, 12 % of held events dip below 0.2 at least once; empty median 0.03, 17 %
  touch 0.9 at least once; open approach median 0.00.
- Caveat: the frozen model is trained on all gold, so p on gold events is in-sample.

Outputs (not in the repository): `~/laundry_dit_training/data/grasp_labels_v2/` (`gold_v2.csv`, `events_v2.csv`,
`report.txt`, `ts_cv/`, `pgrasp_v2_W0.4/model.pt` + `model_meta.json`), review videos in `~/Desktop/grasp_*`.
