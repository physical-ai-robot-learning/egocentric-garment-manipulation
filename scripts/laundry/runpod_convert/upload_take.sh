#!/bin/bash
# Mac -> RunPod network volume: raw tag_rig takes (and optionally the code tarball).  rsync -rlt only (the volume forbids chown).
#   RP_HOST=root@<ip> RP_PORT=<port> ./upload_take.sh 20261102_133624 20261102_134434
#   RP_HOST=... RP_PORT=... ./upload_take.sh --code dist/rp_convert_code_<stamp>.tgz
# env: RP_KEY (default ~/.runpod/ssh/runpodctl-ssh-key), ROOT (default /workspace/rp_convert),
#      TAKE_SRC (default ~/ego_collector/datasets/tag_rig), JOBS (parallel takes, default 2)
# The whole take is sent (color/depth jpg+png for steps 1-5, wrist*/wrist_*.jpg for the step-2 optical-flow cam<->rig pairing,
# npz/json).  --partial makes an interrupted upload resumable; re-running is a no-op for files already there (size+mtime).
set -euo pipefail
: "${RP_HOST:?set RP_HOST=root@<ip>}"; : "${RP_PORT:?set RP_PORT}"
RP_KEY=${RP_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}; ROOT=${ROOT:-/workspace/rp_convert}
TAKE_SRC=${TAKE_SRC:-$HOME/ego_collector/datasets/tag_rig}; JOBS=${JOBS:-2}
RSYNC=/opt/homebrew/bin/rsync; [ -x $RSYNC ] || RSYNC=rsync          # brew rsync 3.x (system openrsync lacks --info=progress2)
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -i $RP_KEY -p $RP_PORT"
$SSH "$RP_HOST" "mkdir -p $ROOT/raw/tag_rig"
if [ "${1:-}" = --code ]; then
  TAR=${2:?--code <tarball>}; $RSYNC -rlt --partial --info=progress2 -e "$SSH" "$TAR" "$RP_HOST:$ROOT/"; echo "uploaded $ROOT/$(basename "$TAR")"; exit 0
fi
[ $# -gt 0 ] || { echo "usage: upload_take.sh <take> [...] | --code <tgz>"; exit 1; }
up() {
  local t=$1 src=$TAKE_SRC/$1
  [ -f "$src/meta.json" ] || { echo "SKIP $t: no $src/meta.json"; return 1; }
  for r in 1 2 3 4 5; do
    $RSYNC -rlt --partial --exclude='._*' --exclude='.DS_Store' --info=progress2 -e "$SSH" "$src" "$RP_HOST:$ROOT/raw/tag_rig/" && { echo "uploaded $t ($(du -sh "$src" | cut -f1))"; return 0; }
    echo "$t rsync retry $r"; sleep 30
  done; echo "FAILED $t"; return 1
}
fail=0; pids=()
for t in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  up "$t" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
