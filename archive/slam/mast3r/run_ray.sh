#!/bin/bash
# [2026-09-23] Ray entrypoint on the 5090 (node:100.64.0.5). Usage: run_ray.sh <subsample> <seed> <variant_name> <tag> [<tag> ...]
set -u
export M3_REPO=/srv/data/johann/mast3r_slam_official M3_BASE=/srv/data/johann/trackA_mast3r_pose_v1
PY=/home/bh-ai-5090/micromamba/envs/m3slam/bin/python
SS=$1; SEED=$2; NAME=$3; shift 3
cd "$M3_BASE"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
for T in "$@"; do
  echo "=== $T ss=$SS seed=$SEED -> runs/$T/$NAME.csv"; date
  "$PY" run_perframe.py --frames "frames/$T/rgb" --calib "frames/$T/calib.yaml" --subsample "$SS" --seed "$SEED" --out "runs/$T/$NAME.csv" 2>&1 | grep -v "^FPS" | tail -4
done
echo ALL_DONE
