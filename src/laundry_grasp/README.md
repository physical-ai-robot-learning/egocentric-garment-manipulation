# laundry_grasp

Grasp held / empty labels for the HandUMI jaw close events, and the causal grasp probability `p_grasp(t)` that the
FLAT-PGRASP run feeds into the masked jaw slots of the state (`laundry_training.env_options`, `PGRASP=<version>`).

**Status (2026-11-05):** current. The generator `grasp-v2-W0.4` is **frozen**: its `model.pt` is never retrained by a
re-run, and its per-demo series and side files are the ones FLAT-PGRASP trains on. Converted from the 11-05 research scripts
(`scripts/laundry/grasp/`) on 11-05 with behaviour preserved; every output below was re-made with the package and compared
with the original scripts' output (see Provenance). The exploratory scripts are in
[`archive/laundry/grasp_exploration/`](../../archive/laundry/grasp_exploration/README.md).

## Why

The jaw servo is read torque-off, position only: no load or current, and the closed position does not separate cloth from
nothing (the right jaw closes to 5263–5279 ticks every time). So "jaw closed" does not say whether the cloth is held. The cue
has to come from the wrist camera.

## Inputs → outputs

| input | where (default) |
|---|---|
| takes: `meta.json`, `<wrist cam>/t_ns.npy`, `wrist_NNNNN.jpg`, `umi_jaw.npz` (`t_ns`, `g_cont`, `state`, `ev_t`, `ev_kind`) | `EGO_DATA_ROOT/datasets/tag_rig/<take>` |
| idle trim (demo windows) | `EGO_TRAIN_ROOT/reports/idle_trim.csv` |
| per-take split docs `<set>_split.json` (demo keys of the flatten sets) | `EGO_GRASP_SPLITS` (default `EGO_GRASP_WORK/split_docs`; on 11-05 they sat in the build session's scratchpad, pass `--split-dir`) |
| labeling rounds `grasp_label_v0_30`, `grasp_gold_r2`, `grasp_gold_r3` (`events.csv`, `gold_selection.json`, `gold_events.csv`) | `EGO_GRASP_ROUNDS` (default `~/Desktop`) |
| round-1 clips that showed two closes | [`configs/laundry/grasp/r1_twoclose_ids.txt`](../../configs/laundry/grasp/r1_twoclose_ids.txt) (59 ids) |
| v1 event table | `EGO_GRASP_WORK/grasp_labels_v1/grasp_events_v1.csv` |

| output | written by |
|---|---|
| `grasp_labels_v2/events_v2.csv` (every v2 close event of the 343 demos), `gold_v2.csv` (gold remapped to v2 events), `report.txt`, `cnn_cv_v2.npz` | `events` |
| `grasp_labels_v2/gold_overrides_v2.json` | the user (review of the confident CV errors, `render --wrong`; round 4 `grasp_gold_r4_loose`) |
| `grasp_labels_v2/ts_cv/ts_cv_raw.npy`, `ts_cv_report.txt` | `cv` |
| `grasp_labels_v2/pgrasp_v2_W0.4/model.pt`, `model_meta.json`, `demos/<take>_demoNN.npz` (`version`, `t_ns`, `L_p`, `L_closed`, `R_p`, `R_closed`; 15 Hz, demo window ± 1 s) | `generate` |
| `<LeRobot set>/meta/pgrasp_<version>.npy` (float32, total_frames × 2, [L, R] by global index) + `.json` | `sidefile` |
| labeling round folder: `clips/`, `gold_selection.json`, `gold_events.csv` | `gold_ui` |
| review videos `<demo>_pgrasp.mp4` (+ `.npz`), `<KIND>__<id>.mp4`; v0 videos + `events.csv` + `labels/` | `render`, `rule_v0` |

L = left hand = `wrist_left` camera, R = right hand = `wrist` camera. Event ids are `<take>_dNN_<hand><n>` (demo 1-based,
n = close event of that hand in the demo).

## Entry points

Run from the repository root with `PYTHONPATH=src` (or the package installed), in the Mac environment with torch,
torchvision and OpenCV on MPS. The 11-05 generator output was made with opencv 5.0 + torch 2.10
(`~/bh_indy7_LeRobot/.venv`); opencv 4.13 gives per-tick p differences up to ~0.1 (JPEG decode / resize), so regenerate a
version only in the environment that made it.

| step | command |
|---|---|
| v0 rule (round-1 sample) | `python -m laundry_grasp.rule_v0 OUT take/demoNN …` |
| gold labeling | `python -m laundry_grasp.gold_ui DIR [PORT]` (127.0.0.1:8098; keys 1 held, 2 empty, 3 uncertain, 0 skip; `GOLD_ALL=1` takes every event of `events.csv`; `GOLD_L3` / `GOLD_T3` relabel key 3) |
| v2 events + gold remap + event CV | `python -m laundry_grasp.events OUT [--split-dir D] [--no-cv]` |
| frame-level CV (window sweep) | `python -m laundry_grasp.cv OUT [report]` |
| review videos | `python -m laundry_grasp.render OUT take_dNN … [--model model.pt]`, `… OUT --wrong` (`PG_HZ`, `PG_WIN_S`, default 15 Hz / 0.2 s) |
| **frozen generator** | `python -m laundry_grasp.generate EGO_TRAIN_ROOT/data/grasp_labels_v2/pgrasp_v2_W0.4 --split-dir D [--demo take/demoNN …]` |
| side file | `python -m laundry_grasp.sidefile SET_DIR PGRASP_DIR [--check-only]` |

## Pipeline (in the order it was built)

| step | module | what |
|---|---|---|
| v0 rule | `rule_v0` | per close event: jaw-tip disk dark share + tip flow vs ego flow → held / empty / unknown; too weak as a label (the user's gold check), kept as the sampler of round 1 |
| gold | `gold_ui`, `gold` | blind labeling page → `gold_events.csv`; rounds 1–3 merged by `gold.roundGold` (later round wins; round-1 clips with two closes dropped) |
| v1 | (archived) | CLIP + logistic regression, temporal CLIP + motor features, jaw-ROI ResNet-18 (chosen); take-grouped CV |
| v2 events | `events` | tighter jaw hysteresis for this pipeline only (open > 0.30, closed < 0.15, dwell 0.1 s; v1 hid 148 partial re-opens); gold remapped by close time (\|dt\| ≤ 0.3 s); `umi_jaw.npz` and the training jaw labels are **not** changed |
| classifier | `model`, `crops` | ImageNet ResNet-18 → 1 logit = P(not held) on 256 px jaw-ROI crops (tip from the fixed jaw pixels, top 153 px above the tip) at 224; 4 frames per close at +0.3…0.9 s + 4 open-approach frames as not-held negatives; 8 epochs, AdamW 1e-4, seed 0 |
| p(t) | `pgrasp`, `cv`, `render` | causal frame-level p_grasp(t): every 15 Hz tick scored (open frames too), mean over `[max(last v2 jaw change, t − W), t]`; window sweep CV |
| frozen | `generate` | **`grasp-v2-W0.4`**: classifier on all gold (+ `gold_overrides_v2.json`: slip excluded, loose = not held) + open-approach negatives → per-demo `t_ns, L_p, R_p` at 15 Hz, W = 0.4 s |
| side file | `sidefile` | `<set>/meta/pgrasp_<version>.npy` for an aggregated LeRobot set without rebuilding it; checks per episode that frame 0 sits at the demo window start (≤ 0.25 s); zero-order hold on the wrist source time |

## Numbers (11-05)

- Gold: 585 events over four rounds, 543 remapped to v2 events (held 372, empty 171).
- Event CNN, take-grouped 5-fold, v2 crops: held P/R/F1 0.90/0.90/0.90, empty 0.79/0.77/0.78.
- Causal p_grasp(t), W = 0.4 s: held median 0.99, 12 % of held events dip below 0.2 at least once; empty median 0.03, 17 %
  touch 0.9 at least once; open approach median 0.00.
- Caveat: the frozen model is trained on all gold, so p on gold events is in-sample.

## Layout

| module | contents (was) |
|---|---|
| `demos` | `demoWindow`, demo keys / video names / event ids, `HAND_CAMS`, `splitDocKeys` (helpers of `grasp_label_v0`, key readers of `grasp_v2_events` / `grasp_pgrasp_gen`) |
| `crops` | `jawMaskAndTip`, `roiOrigin`, `cropRoi`, `JawRoiCropper` (`grasp_label_v0.jaw_mask_and_tip`, `grasp_render_prob.crop_fn`, the crop code repeated in four scripts) |
| `events` | `segmentJaw`, `jawV2`, `closeIntervals`, `eventsV2`, `remapGold`, CLI (`grasp_v2_events`) |
| `gold` | `roundGold`, `loadOverrides`, `eventClass` (`grasp_label_v1_temporal.gold`, override handling) |
| `model` | `trainFold`, `predict`, `toTensor`, `eventFrameTimes`, `approachFrameTimes`, `trainingSet`, `trainAll`, `loadModel` (`grasp_cnn_v1`, `grasp_render_prob.train_all`) |
| `pgrasp` | `causalWindowMean`, `lastChangeTimes`, `framePGrasp`, `closedAt`, `VERSION` / `HZ` / `WINDOW_S` (the rule repeated in `grasp_render_prob`, `grasp_ts_cv`, `grasp_pgrasp_gen`) |
| `generate` | frozen generator CLI (`grasp_pgrasp_gen`) |
| `sidefile` | `buildSideFile`, `holdValues`, `checkFrameZero`, CLI (`grasp_pgrasp_sidefile`) |
| `cv` | `takeFolds`, `bandF1`, frame-level CV CLI + `report`, `eventLevelCv` (`grasp_ts_cv`, CV part of `grasp_v2_events`, `grasp_label_v1_temporal.f1s`) |
| `render` | `pSeries`, `renderDemo`, `renderOverrideClips`, CLI (`grasp_render_prob`) |
| `gold_ui` + `gold_ui_page.html` | labeling server (`grasp_gold_ui`) |
| `rule_v0` | v0 features, `labelHand`, `renderDemo`, CLI (`grasp_label_v0`) |
| `video` | mp4v writer, ffmpeg H.264 (`EGO_FFMPEG`), single-close clip span |
| `tests/` | synthetic take (jaw series + wrist jpgs): hysteresis, close events, demo windows, causal window incl. restart at a jaw change (bit-equal to the 11-05 expression), side-file hold / frame-0 check, crop geometry, gold merge, labeling routes, CLIs |

## Provenance

- Code: the 11-05 scripts in `~/laundry_dit_training/datasets/grasp_*.py` (identical to the former `scripts/laundry/grasp/`).
  Old → new map: `docs/laundry/structure_refactor/2026-11-04_oa_layout.md`, section 11-05.
- Data: `~/laundry_dit_training/data/grasp_labels_v2/` (`gold_v2.csv`, `events_v2.csv`, `gold_overrides_v2.json`,
  `report.txt`, `cnn_cv_v2.npz`, `ts_cv/`, `pgrasp_v2_W0.4/`), review videos in `~/Desktop/grasp_*`.
- Conversion check (11-05, real data, opencv 5.0 env): v2 event and gold tables, the event-level CV (`cnn_cv_v2.npz`, 5 folds of
  MPS training), the training sets of the frozen model and of CV fold 0, p_grasp of three demos, the val-set side file,
  the CV report, v0 labels + video, labeling page + clips, override clips — all byte-identical to the original scripts' output.
- Known quirks kept as they were (not fixed): see the 11-05 section of the refactor log ("의심 버그").
