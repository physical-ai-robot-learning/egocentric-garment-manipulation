#!/bin/bash
# One-time (idempotent) setup on a RunPod pod for the tag_rig conversion.  Everything lives on the network volume so CPU pods and the
# GPU (CoTracker) pod that mount the same volume share code, envs and data:
#   $ROOT/code/home        extracted code tarball = HOME template (ego_collector/..., laundry_dit_training/..., laundry_system/...,
#                          handumi-sw/assets/..., .cache/torch/hub/...)
#   $ROOT/envs/{ego,ik,ct} uv venvs (ego: py3.11 CPU stages, ik: py3.12 prep + robotize IK, ct: py3.11 CUDA torch for CoTracker)
#   $ROOT/uv-python        uv-managed interpreters (on the volume, so the venvs work on every pod mounting it at the same path)
#   $ROOT/raw/tag_rig/<take>   uploaded raw takes (upload_take.sh)
#   $ROOT/jobs/<PACK>/...      per-job HOME copy, CoTracker pack, results (pod_convert_take.sh)
# usage (on the pod):  bash pod_setup.sh /workspace/rp_convert/rp_convert_code_<stamp>.tgz [ego,ik,ct]
#   first pod: with the tarball.  Every further pod on the same volume: `bash $ROOT/code/home/laundry_dit_training/remote/runpod_convert/pod_setup.sh -`
#   ("-" = keep the code template; still installs rsync/uv in that container and builds/validates the envs it needs).
#   ENVS default: "ego,ik" on a pod without a GPU, "ego,ik,ct" when nvidia-smi works.  Re-run with a new tarball to update the code
#   (the template is replaced; existing job dirs keep their own copy).
#   TORCH_INDEX (default https://download.pytorch.org/whl/cu124) -- CUDA wheel index for the ct env; cu124 wheels need driver >= 550.
set -euo pipefail
ROOT=${ROOT:-/workspace/rp_convert}
TAR=${1:?usage: pod_setup.sh <code.tgz>|- [envs]}
if [ -n "${2:-}" ]; then ENVS=$2; elif nvidia-smi -L >/dev/null 2>&1; then ENVS=ego,ik,ct; else ENVS=ego,ik; fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
export UV_PYTHON_INSTALL_DIR=$ROOT/uv-python UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv} UV_LINK_MODE=copy
mkdir -p "$ROOT"/{code,envs,raw/tag_rig,jobs}
log() { echo "[$(date '+%F %T')] $*"; }

# ---- system bits (rsync for the Mac <-> volume transfers; libgl not needed: opencv *headless*)
if ! command -v rsync >/dev/null || ! command -v curl >/dev/null; then
  log "apt: rsync curl"; (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync curl ca-certificates >/dev/null) || log "WARN apt failed"
fi
command -v uv >/dev/null || { log "install uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

# ---- code template (do not replace it while jobs are being CREATED on other pods; running jobs use their own copy)
if [ "$TAR" != - ]; then
  log "extract $TAR -> $ROOT/code"
  rm -rf "$ROOT/code/home.new"; mkdir -p "$ROOT/code/home.new"
  tar -xzf "$TAR" -C "$ROOT/code/home.new" --no-same-owner --no-same-permissions --strip-components=1     # tarball paths are home/<...>
  [ -d "$ROOT/code/home" ] && mv "$ROOT/code/home" "$ROOT/code/home.old.$(date +%s)"
  mv "$ROOT/code/home.new" "$ROOT/code/home"
fi
[ -d "$ROOT/code/home/ego_collector" ] || { echo "no code template at $ROOT/code/home (pass the tarball)"; exit 1; }
HT=$ROOT/code/home; RC=$HT/laundry_dit_training/remote/runpod_convert
# env symlinks the chain resolves through $HOME:  tag_rig_to_lerobot.IK_PY = ~/xvla-mac/bin/python, EGO_PY = ~/ego_collector/.venv/bin/python
ln -sfn "$ROOT/envs/ik" "$HT/xvla-mac"; ln -sfn "$ROOT/envs/ego" "$HT/ego_collector/.venv"
# Safety net for literal Mac paths (none in the chain code as of 11-02; json files may store them): /Users/jeonghwanlee -> template (read-only use)
if [ ! -e /Users/jeonghwanlee ]; then mkdir -p /Users && ln -s "$HT" /Users/jeonghwanlee || true; fi

mkvenv() {   # name pyver reqfile [extra uv pip args...]
  local n=$1 v=$2 r=$3; shift 3
  if [ -x "$ROOT/envs/$n/bin/python" ] && [ "$(cat "$ROOT/envs/$n/.req_sha" 2>/dev/null)" = "$(sha1sum "$r" | cut -c1-40)" ]; then log "env $n up to date"; return; fi
  log "env $n (python $v) <- $(basename "$r")"
  uv python install "$v"
  uv venv --python "$v" "$ROOT/envs/$n" --clear
  uv pip install --python "$ROOT/envs/$n/bin/python" -r "$r" "$@"
  sha1sum "$r" | cut -c1-40 > "$ROOT/envs/$n/.req_sha"
}
IFS=, read -ra WANT <<< "$ENVS"
for e in "${WANT[@]}"; do case $e in
  ego) mkvenv ego 3.11 "$RC/req_ego.txt" ;;
  ik)  mkvenv ik 3.12 "$RC/req_ik.txt" ;;
  ct)  mkvenv ct 3.11 "$RC/req_ct.txt" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match ;;
  *) echo "unknown env $e"; exit 1 ;;
esac; done

# ---- smoke imports (cheap; catches a missing wheel before a job starts)
export HOME=$HT LANG=C.UTF-8 PYTHONIOENCODING=utf-8
E=$HT/ego_collector; IC=$E/datasets/m3_runs/_imu_consistency
if [ -x "$ROOT/envs/ego/bin/python" ]; then
  log "smoke ego"
  "$ROOT/envs/ego/bin/python" -c "
import sys; sys.path[:0]=['$E/scripts','$E','$IC','$IC/orbbec/v2b','$IC/stitch','$E/datasets/m3_runs/_fgo_1023/tools']
import cv2, numpy, scipy, pandas
import tag_rig_fuse, tag_rig_pose, core, camcal, fgo_data, v32_common, imu_motion, build_candidate_v0, wdo
assert hasattr(cv2, 'aruco'); print('ego ok cv2', cv2.__version__, 'numpy', numpy.__version__)"
fi
if [ -x "$ROOT/envs/ik/bin/python" ]; then
  log "smoke ik"
  "$ROOT/envs/ik/bin/python" -c "
import sys; sys.path[:0]=['$E','$HT/laundry_dit_training/datasets','$HT/laundry_system']
from visual_teleop import config as vcfg; from visual_teleop.robot.ik_pink import PinkIkSolver; from visual_teleop.robot.ik import ArmIk
from ego_teleop.robot.rebot_client import SIDE_SLICE; import xvla_core, umi_to_lerobot, tag_rig_to_lerobot
vcfg.load(); PinkIkSolver('left'); print('ik ok')"
fi
if [ -x "$ROOT/envs/ct/bin/python" ]; then
  log "smoke ct (loads CoTracker3 online from the packed hub cache)"
  "$ROOT/envs/ct/bin/python" -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
m = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_online'); print('cotracker ok step', m.step)"
fi
log "SETUP_DONE root=$ROOT envs=$ENVS"
