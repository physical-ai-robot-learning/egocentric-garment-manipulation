#!/bin/bash
# 11-04 BH cluster rules: gateway 8266, GPU by label (no CUDA_VISIBLE_DEVICES), data under /mnt/shared/users/johann, env = pyproject (uv run).
# usage: submit.sh <local pack dir>   -> uploads to mine/data/cotrack/<pack>, runs CoTracker3 online, pulls tracks.npz back
set -e
# the gateway allows ONE rrsync per user dir at a time ("Another instance of rrsync") -> wait and retry the transfer (not the job)
retry_rsync() {   # 11-04 18:0x: one gateway rsync at a time from this Mac (local mkdir lock), so parallel takes queue here instead of burning retries on the gateway lock
  until mkdir /tmp/johann_gw_rsync.lock 2>/dev/null; do sleep 5; done
  local rc=1; for i in $(seq 1 10); do /opt/homebrew/bin/rsync "$@" && { rc=0; break; }; echo "rsync failed, retry $i"; sleep 30; done
  rmdir /tmp/johann_gw_rsync.lock; return $rc; }
 P=$(cd "$1" && pwd); PK=$(basename $P); H=$(cd $(dirname $0) && pwd); GPU=${GPU:-any}
GW=${RAY_GATEWAY_HOST:-100.64.0.1}   # BH Ray gateway (tailnet); override with RAY_GATEWAY_HOST
RS=retry_rsync; export RAY_ADDRESS=http://$GW:8266; RAY=~/ray-env/bin/ray
$RS -aL --mkpath --exclude tracks.npz --delete-excluded $P/ johann@$GW:mine/data/cotrack/$PK/   # 11-11: --delete-excluded drops a stale remote tracks.npz (run_cotracker skips demos that have one -> a re-packed take got the OLD tracks back)
# --no-wait + status poll: a dropped gateway connection while waiting no longer loses a job that is still running (11-04 135359)
# 11-04 user "병렬로": up to 3 jobs at once (cluster rule: <= 3 GPUs per person incl. pending), any free GPU (cu128 works on Ada + Blackwell)
SLOT=""; while [ -z "$SLOT" ]; do for i in 1 2 3; do mkdir /tmp/johann_ct_slot$i 2>/dev/null && SLOT=/tmp/johann_ct_slot$i && break; done; [ -z "$SLOT" ] && sleep 15; done
trap 'rmdir $SLOT 2>/dev/null' EXIT
RES=(); [ "$GPU" != any ] && RES=(--entrypoint-resources "{\"$GPU\": 1}")
SID=$(cd $H && $RAY job submit --no-wait --entrypoint-num-gpus 1 "${RES[@]}" --working-dir . -- \
  bash -c "export TORCH_HOME=/mnt/shared/users/johann/scratch/torch_hub; uv run python run_cotracker.py /mnt/shared/users/johann/data/cotrack/$PK" 2>&1 | grep -oE "johann-[0-9]+-[0-9]+" | head -1)
[ -n "$SID" ] || { echo "submit failed"; exit 1; }; echo "submitted $SID"
while :; do st=$($RAY job status $SID 2>&1 | grep -oE "succeeded|failed|stopped" | head -1); [ -n "$st" ] && break; sleep 30; done
echo "$SID $st"; [ "$st" = succeeded ] || exit 1
$RS -aK --include='*/' --include='tracks.npz' --exclude='*' johann@$GW:mine/data/cotrack/$PK/ $P/
echo "COTRACK_DONE $PK $(ls $P/*/tracks.npz | wc -l)"
