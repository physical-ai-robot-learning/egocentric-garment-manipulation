#!/bin/bash
# [2026-10-20] UMI take -> fused TCP trajectory (+ jaw events), the label pipeline of the replay-parity work:
#   umi_jaw infer (wrist jaw events) -> umi_tcp_traj --imu-fill (tag reference) -> wrist_m3_prep -> rsync to the 5090 NFS
#   -> MASt3R-SLAM via Ray (5090, node:100.64.0.5) -> fetch CSV -> umi_fuse_slam --validate --save
# Usage: scripts/umi_pipeline.sh datasets/tag_rig/<take> [hand=오른손|왼손]   (run once per hand; 10-20 bimanual)
set -euo pipefail
cd "$(dirname "$0")/.."
TAKE=${1%/}; HAND=${2:-오른손}; NAME=$(basename "$TAKE"); PY=.venv/bin/python
if [ "$HAND" = "왼손" ]; then SIDE=left; WD=wrist_left; TAG=umi_${NAME}_left; else SIDE=right; WD=wrist; TAG=umi_$NAME; fi
M3=/srv/data/johann/trackA_mast3r_pose_v1; export RAY_ADDRESS=http://100.64.0.1:8265
echo "== 1/6 집게 판정"; $PY -m visual_teleop.tools.umi_jaw infer "$TAKE/$WD" 2>&1 | grep "\[집게\]" || true
echo "== 2/6 태그 궤적"; $PY scripts/umi_tcp_traj.py "$TAKE" --hand "$HAND" --imu-fill 2>&1 | grep -E "IMU|$HAND|구간" || true
echo "== 3/6 MASt3R 입력"; $PY scripts/wrist_m3_prep.py "$TAKE" --hand $SIDE | tail -1
echo "== 4/6 업로드"; /opt/homebrew/bin/rsync -a "datasets/m3_frames/$TAG" bh-ai-5090@gpu-5090:$M3/frames/
echo "== 5/6 MASt3R-SLAM (5090)"; ray job submit --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.001}' -- bash $M3/run_ray.sh 1 0 m3_ss1 "$TAG" 2>&1 | grep -E "done|ALL_DONE|Error|failed" || true
mkdir -p "datasets/m3_runs/$TAG"; scp -q "bh-ai-5090@gpu-5090:$M3/runs/$TAG/m3_ss1.csv" "bh-ai-5090@gpu-5090:$M3/runs/$TAG/m3_ss1.json" "datasets/m3_runs/$TAG/"
echo "== 6/6 융합"; $PY scripts/umi_fuse_slam.py "$TAKE" "datasets/m3_runs/$TAG/m3_ss1.csv" "datasets/m3_frames/$TAG/frames.json" --hand $SIDE --validate --save 2>&1 | grep -vE "^$|Warn"
echo "PIPELINE_DONE $TAKE"
