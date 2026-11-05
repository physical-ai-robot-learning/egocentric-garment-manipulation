#!/bin/bash
# 11-04 BH cluster rules: gateway 8266, GPU by label (no CUDA_VISIBLE_DEVICES), data under /mnt/shared/users/johann, env = pyproject (uv run).
# usage: submit.sh <local pack dir>   -> uploads to mine/data/cotrack/<pack>, runs CoTracker3 online, pulls tracks.npz back
set -e
# the gateway allows ONE rrsync per user dir at a time ("Another instance of rrsync") -> wait and retry the transfer (not the job)
retry_rsync() { for i in $(seq 1 60); do /opt/homebrew/bin/rsync "$@" && return 0; echo "rsync busy/failed, retry $i"; sleep 20; done; return 1; }
 P=$(cd "$1" && pwd); PK=$(basename $P); H=$(cd $(dirname $0) && pwd); GPU=${GPU:-rtx4090b}
RS=retry_rsync; export RAY_ADDRESS=http://100.64.0.1:8266; RAY=~/ray-env/bin/ray
$RS -aL --mkpath --exclude tracks.npz $P/ johann@100.64.0.1:mine/data/cotrack/$PK/
# --no-wait + status poll: a dropped gateway connection while waiting no longer loses a job that is still running (11-04 135359)
SID=$(cd $H && $RAY job submit --no-wait --entrypoint-num-gpus 1 --entrypoint-resources "{\"$GPU\": 1}" --working-dir . -- \
  bash -c "export TORCH_HOME=/mnt/shared/users/johann/scratch/torch_hub; uv run python run_cotracker.py /mnt/shared/users/johann/data/cotrack/$PK" 2>&1 | grep -oE "johann-[0-9]+-[0-9]+" | head -1)
[ -n "$SID" ] || { echo "submit failed"; exit 1; }; echo "submitted $SID"
while :; do st=$($RAY job status $SID 2>&1 | grep -oE "succeeded|failed|stopped" | head -1); [ -n "$st" ] && break; sleep 30; done
echo "$SID $st"; [ "$st" = succeeded ] || exit 1
$RS -aK --include='*/' --include='tracks.npz' --exclude='*' johann@100.64.0.1:mine/data/cotrack/$PK/ $P/
echo "COTRACK_DONE $PK $(ls $P/*/tracks.npz | wc -l)"
