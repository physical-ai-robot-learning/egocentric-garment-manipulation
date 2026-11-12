#!/bin/zsh
# 11-12 user: pull every finished checkpoint of the RunPod X-VLA run (pod 8gj4zc7snqoaqe, H100x4) to the Mac and on to the cluster.
#   cluster mine/ckpt/xvla/<run>/checkpoints/<step>/pretrained_model   every checkpoint
#   cluster mine/ckpt/xvla/<run>/latest_training_state/                newest checkpoint only (+ STEP file): resume if the pod dies
# The Mac keeps only the newest pulled checkpoint (staging).  A checkpoint is finished when training_state/training_step.json
# exists (lerobot writes it last).
RUN=${RUN:-XVLA-UMIFLAT-WSD-CUM1111-TF32-GB64}
SSH=(ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p 18852 -o ConnectTimeout=20 root@157.66.254.11)
E="ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p 18852 -o ConnectTimeout=20"
RS=/opt/homebrew/bin/rsync
POD=/workspace/umiflat/ckpt/$RUN/checkpoints
CL=johann@100.64.0.1:mine/ckpt/xvla/$RUN
ST=$HOME/runpod_stage/ckpt_pull/$RUN; mkdir -p $ST
DONE=$ST/pulled.txt; touch $DONE
LOCK=/tmp/johann_gw_rsync.lock
while true; do
  for s in $($SSH "cd $POD 2>/dev/null && for d in [0-9]*; do [ -f \$d/training_state/training_step.json ] && echo \$d; done" 2>/dev/null); do
    grep -qx $s $DONE && continue
    echo "$(date '+%m-%d %T') pull $s"
    $RS -a --partial -e "$E" root@157.66.254.11:$POD/$s $ST/ || { echo "  pod rsync failed"; continue; }
    echo $s > $ST/$s/training_state/STEP
    until mkdir $LOCK 2>/dev/null; do sleep 30; done
    $RS -a --partial --mkpath $ST/$s/pretrained_model $CL/checkpoints/$s/ \
      && $RS -a --partial --delete --mkpath $ST/$s/training_state/ $CL/latest_training_state/; rc=$?
    rmdir $LOCK
    [ $rc = 0 ] || { echo "  cluster rsync failed rc $rc"; continue; }
    echo $s >> $DONE; echo "$(date '+%m-%d %T') $s on cluster"
    # 11-12 user "매 ckpt는 val이랑 test 돌려주고 저장해줘": offline eval on the cluster (1 GPU), json next to the checkpoint + Mac copy
    ( bash $HOME/laundry_dit_training/ray_jobs/xvla_umiflat/submit_eval.sh $RUN $s > $ST/eval_$s.log 2>&1; echo "$(date '+%m-%d %T') eval $s: $(tail -1 $ST/eval_$s.log)" ) &
    for o in $(ls $ST | grep -E '^[0-9]+$' | sort | sed '$d'); do rm -rf $ST/$o; done   # older Mac staging copies (on the cluster)
  done
  $SSH "test -f /workspace/umiflat/ckpt/$RUN/cstable.done" 2>/dev/null && { echo "run done"; break; }
  sleep 600
done
