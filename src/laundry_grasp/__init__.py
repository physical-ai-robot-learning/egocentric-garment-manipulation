"""Grasp labels of the laundry pipeline: held / empty per HandUMI jaw close event, and the causal grasp probability p_grasp(t) that
the FLAT-PGRASP run feeds into the masked jaw slots of the state (laundry_training.env_options PGRASP).

The jaw servo is read torque-off, position only, so a closed jaw does not say whether cloth is held; the cue comes from the wrist
camera (a jaw-ROI ResNet-18 trained on the user's gold labels).

Modules (each CLI runs as python -m laundry_grasp.<module>):
    demos      demo windows, demo keys / video names / event ids, wrist cameras
    crops      jaw mask + tip finder, jaw-ROI crop (JawRoiCropper)
    events     v2 jaw segmentation (jawV2), close events, gold remap; CLI writes events_v2.csv / gold_v2.csv
    gold       gold labels of the labeling rounds, overrides
    model      the ResNet-18 recipe (trainFold, predict, trainAll, loadModel)
    pgrasp     the causal p_grasp(t) rule (causalWindowMean, lastChangeTimes)
    generate   FROZEN generator grasp-v2-W0.4 (per-demo p_grasp series)
    sidefile   <set>/meta/pgrasp_<version>.npy for an aggregated LeRobot set
    cv         take-grouped CV (frame-level CLI, event-level for events)
    render     p_grasp review videos
    gold_ui    blind labeling page
    rule_v0    v0 rule labeler (round-1 sample)
    video      ffmpeg / clip helpers
"""
from __future__ import annotations
