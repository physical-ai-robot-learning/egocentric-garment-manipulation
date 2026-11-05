# grasp_exploration — superseded grasp-label experiments (11-05)

Research scripts of the 11-05 grasp held / empty work that are **not** on the path of the frozen p_grasp generator
`grasp-v2-W0.4`. Copied verbatim from `scripts/laundry/grasp/` when that folder became the package
[`src/laundry_grasp`](../../../src/laundry_grasp/README.md) (11-05). They are kept as the record of what was tried; they are
not maintained and not run. They still edit `sys.path`, rewrite `sys.argv`, hard-code `~/Desktop/grasp_*` folders and import
the old module names (`grasp_label_v0`, `grasp_cnn_v1`, `grasp_label_v1_temporal`, `grasp_v2_events`, `grasp_render_prob`),
which no longer exist as files; the code they used now lives in `laundry_grasp` (map in
`docs/laundry/structure_refactor/2026-11-04_oa_layout.md`, section 11-05).

| script | what it was | superseded by |
|---|---|---|
| `grasp_clf_v1.py` | frozen CLIP ViT-B/16 features of 3 wrist frames + closed duration → logistic regression, leave-one-take-out CV on the 188 round-1 gold events | jaw-ROI ResNet-18 (`laundry_grasp.model`) |
| `grasp_label_v1_temporal.py` | temporal CLIP features of the jaw ROI (8 frames) + motor / v0 flow features → logistic regression. Its `gold()` (round merge) and `f1s()` are now `laundry_grasp.gold.roundGold` and `laundry_grasp.cv.bandF1` | ResNet-18 |
| `grasp_cnn_v1.py` | the ResNet-18 recipe on v1 crops (v1 jaw segmentation), take-grouped 5-fold CV, average with the motor model. `train_fold` / `predict` / `DTS` are now `laundry_grasp.model.trainFold` / `predict` / `EVENT_FRAME_OFFSETS_S` | v2 crops (`laundry_grasp.cv.eventLevelCv`) |
| `grasp_mine_r3.py` | round-3 labeling set: CNN on all gold, score every unqueued event, likely empties + uncertain band → `events.csv` | done (round 3 labelled 11-05) |
| `grasp_apply_v1.py` | v1 CNN on every close event with the 0.9 / 0.2 rule (gold overrides) → `grasp_events_v1.csv` | event-level labels dropped for the causal p_grasp(t) |
| `grasp_render_v1.py` | whole-demo review videos with the v1 CNN labels | `laundry_grasp.render` |
| `grasp_ts_cv2.py` | frame-level CV with cached crops for several jaw-ROI sizes (more context for hidden bites) | ROI kept at 256 px (frozen) |
| `grasp_wrong_clips.py` | clips of the confidently wrong v2 CV events (gold empty with P > 0.9, gold held with P < 0.2) for the user review that produced `gold_overrides_v2.json` | `laundry_grasp.render --wrong` (same events, with the causal strip) |
