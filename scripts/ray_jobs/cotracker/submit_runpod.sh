#!/bin/bash
# 11-04 16:5x: the BH gateway mine/ went read-only (Permission denied) -> user "그냥 runpod 쓰자". Same contract as submit.sh:
# usage: submit_runpod.sh <local pack dir>  -> rsync pack to the RunPod CoTracker pod, run CoTracker3 online, pull tracks.npz back
set -e; P=$(cd "$1" && pwd); PK=$(basename $P)
HOST=${CT_HOST:?set CT_HOST=root@<pod ip>}; PORT=${CT_PORT:?set CT_PORT=<pod ssh port>}; K=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH="ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -i $K -p $PORT"; RS=/opt/homebrew/bin/rsync
retry() { for i in $(seq 1 10); do "$@" && return 0; echo "retry $i: $*" | cut -c1-120; sleep 20; done; return 1; }
# one ssh stream to a far pod is latency-bound (~0.1 MB/s); office uplink is ~2-3 MB/s -> one rsync per demo dir, NPAR at a time
$SSH $HOST "mkdir -p /workspace/ct/packs/$PK"
export RS SSH HOST PK
ls -d $P/*/ | xargs -P ${NPAR:-12} -n 1 $(dirname $0)/up_one.sh
retry $SSH $HOST "cd /workspace/ct && python run_cotracker.py /workspace/ct/packs/$PK > /workspace/ct/$PK.log 2>&1; tail -2 /workspace/ct/$PK.log"
retry $RS -aK --no-o --no-g --include='*/' --include='tracks.npz' --exclude='*' -e "$SSH" $HOST:/workspace/ct/packs/$PK/ $P/
echo "COTRACK_DONE $PK $(ls $P/*/tracks.npz | wc -l)"
