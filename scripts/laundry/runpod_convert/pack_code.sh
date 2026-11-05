#!/bin/bash
# Build the code+config+weights tarball for the RunPod tag_rig conversion (see DEPS.md).  Read-only on every source file.
# Layout inside the tarball = paths relative to $HOME ("home/..."), so on the pod it is extracted to <ROOT>/code/home and used as
# a HOME template (pod_convert_take.sh copies it per job and runs with HOME=<job>/home -> every Path.home() in the chain resolves).
#   usage: pack_code.sh [OUT_TAR]      (default: ~/laundry_dit_training/remote/runpod_convert/dist/rp_convert_code_<YYYYmmdd_HHMM>.tgz)
#   DRY_RUN=1 pack_code.sh  -> only list the files + total size
#   NO_WEIGHTS=1 pack_code.sh   -> leave out the CoTracker checkpoint (100 MB); the pod then downloads it from HF on first use.
set -euo pipefail
H=$HOME; HERE=$(cd "$(dirname "$0")" && pwd)
# 11-04 refactor: the repository (this script lives in <repo>/scripts/laundry/runpod_convert) ships its src/ packages too; the pod chain
# runs laundry_tracking with PYTHONPATH=<job HOME>/<repo>/src.  The repository must sit under $HOME (tar paths are relative to it).
REPO=$(cd "$HERE/../../.." && pwd); case "$REPO" in "$H"/*) REPO_REL=${REPO#$H/} ;; *) echo "abort: repository $REPO is not under $H" >&2; exit 1 ;; esac
OUT=${1:-$HERE/dist/rp_convert_code_$(date +%Y%m%d_%H%M).tgz}; mkdir -p "$(dirname "$OUT")"
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
E=ego_collector; IC=$E/datasets/m3_runs/_imu_consistency; B=$IC/orbbec/v2b; F=$E/datasets/m3_runs/_fgo_1023/tools; L=laundry_dit_training
cd "$H"
pyfiles() { find "$@" -type f -name '*.py' -not -path '*/__pycache__/*' -not -name '._*'; }
{
  # ---- code (python only, whole small packages so lazy imports cannot be missed)
  pyfiles $E/scripts $E/visual_teleop $E/ego_teleop $E/handumi_collector $E/ego_collector
  find $IC -maxdepth 1 -type f -name '*.py'
  find $IC/stitch -maxdepth 1 -type f -name '*.py'
  find $B -maxdepth 1 -type f -name '*.py'
  pyfiles $B/cotrack $B/fusion
  pyfiles $F
  pyfiles $L/datasets
  pyfiles laundry_system/contracts
  pyfiles "$REPO_REL/src"                                          # laundry_tracking / laundry_common / laundry_shared (repository)
  find "$REPO_REL/src" -type f \( -name '*.json' -o -name '*.yaml' \) -not -path '*/tests/*' -not -name '._*'   # package resources
  find "$REPO_REL/configs/laundry" -type f \( -name '*.json' -o -name '*.yaml' \) -size -2M -not -name '._*'
  # ---- configs / static inputs
  find $E/configs \( -name '*.json' -o -name '*.yaml' \) -size -2M -not -name '._*'
  find $E/visual_teleop/config -maxdepth 1 -name '*.yaml'
  find $E/ego_teleop $E/handumi_collector \( -name '*.yaml' -o -name '*.json' \) -size -1M -not -path '*/__pycache__/*' 2>/dev/null || true
  find $B -maxdepth 1 -type f -name '*.json'                       # body_offset_by_rig_v1, start_seed_v1 + per-take seeds, FROZEN_*
  echo $B/out/rig_model_left.npz; echo $B/out/rig_model_right.npz  # solve2 rigid body model
  ls $IC/out/RX_pooled_*.npy                                       # core.py / wdo.calibrate (not on the Cg path, tiny, kept for safety)
  echo $L/reports/idle_trim.csv                                    # trims(); new_take_prep appends to the per-job copy
  echo $L/reports/orbbec_v31speed_episode_stats.csv                # episode_quality.py "in130" table (alt path of the Desktop csv)
  echo handumi-sw/assets/rebot_b601/rebot_b601_bimanual.urdf        # pink IK (visual_teleop/robot/ik_pink.py URDF)
  echo handumi-sw/assets/rebot_b601/README.md
  # ---- CoTracker3 (torch.hub cache: repo code + online checkpoint)
  CT=.cache/torch/hub/facebookresearch_co-tracker_main
  echo $CT/hubconf.py; echo $CT/setup.py; echo $CT/LICENSE.md
  pyfiles $CT/cotracker
  echo .cache/torch/hub/trusted_list
  [ "${NO_WEIGHTS:-0}" = 1 ] || echo .cache/torch/hub/checkpoints/scaled_online.pth
  # ---- these scripts
  find $L/remote/runpod_convert -maxdepth 1 -type f \( -name '*.sh' -o -name '*.txt' -o -name '*.md' -o -name '*.py' \)
} | grep -v -E '\.bak|/\._|\.DS_Store' | sort -u > "$LIST"
missing=0; while read -r f; do [ -e "$f" ] || { echo "MISSING $f" >&2; missing=1; }; done < "$LIST"
[ $missing = 0 ] || { echo "abort: missing files" >&2; exit 1; }
if [ "${DRY_RUN:-0}" = 1 ]; then cat "$LIST"; echo "DRY_RUN: $(wc -l < "$LIST") files, $(tr '\n' '\0' < "$LIST" | xargs -0 du -ck | tail -1 | cut -f1) KB"; exit 0; fi
# prefix every path with home/ so the tarball extracts to <dest>/home/...
COPYFILE_DISABLE=1 tar -czf "$OUT" -s ',^,home/,' -T "$LIST"
echo "files: $(wc -l < "$LIST")  size: $(du -h "$OUT" | cut -f1)  -> $OUT"
tar -tzf "$OUT" | grep -c '^home/' >/dev/null && echo "ok"
