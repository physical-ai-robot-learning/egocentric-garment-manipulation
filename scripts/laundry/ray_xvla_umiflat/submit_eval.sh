#!/bin/bash
# 11-12 user "매 ckpt는 val이랑 test 돌려주고 저장해줘": offline eval (src/eval_offline.py via run.sh STAGE=ceval) of one cluster checkpoint
# on the merged val + test sets, 1 GPU on the BH cluster (gateway 8266, any free GPU).  Results are written NEXT TO the checkpoint
# (mine/ckpt/xvla/<run>/checkpoints/<step>/eval_<split>.json) and copied to the Mac (~/runpod_stage/evals/<run>/<step>_<split>.json).
# usage: submit_eval.sh <run> <step>      (or CKDIR=<cluster-relative ckpt dir under /mnt/shared/users/johann> submit_eval.sh <tag> <step>)
set -e
RUN=$1; S=$2; H=$(cd $(dirname $0) && pwd)
CKDIR=${CKDIR:-ckpt/xvla/$RUN/checkpoints/$S}
ROOTS=${ROOTS:-flatcum1111_S03G_xvla_val,flatcum1111_S03G_xvla_test}
export RAY_ADDRESS=http://100.64.0.1:8266; RAY=~/ray-env/bin/ray
SID=$(cd $H && $RAY job submit --no-wait --entrypoint-num-gpus 1 --working-dir . -- \
  bash -c "CK=/mnt/shared/users/johann/$CKDIR EVAL_ROOTS=$ROOTS STRIDE=${STRIDE:-15} STAGE=ceval NGPU=1 bash run.sh" 2>&1 | grep -oE "johann-[0-9]+-[0-9]+" | head -1)
[ -n "$SID" ] || { echo "submit failed"; exit 1; }; echo "eval $RUN $S submitted $SID"
while :; do st=$($RAY job status $SID 2>&1 | grep -oE "succeeded|failed|stopped" | head -1); [ -n "$st" ] && break; sleep 60; done
$RAY job logs $SID 2>&1 | grep -E "^\[eval\] (left|right|score)" || true
echo "eval $RUN $S $SID $st"; [ "$st" = succeeded ] || exit 1
D=$HOME/runpod_stage/evals/$RUN; mkdir -p $D
until mkdir /tmp/johann_gw_rsync.lock 2>/dev/null; do sleep 5; done
for r in ${ROOTS//,/ }; do /opt/homebrew/bin/rsync -a johann@100.64.0.1:mine/$CKDIR/eval_$r.json $D/${S}_${r#flatcum1111_S03G_xvla_}.json || true; done
rmdir /tmp/johann_gw_rsync.lock
ls $D
