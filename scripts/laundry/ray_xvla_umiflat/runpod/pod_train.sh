#!/bin/bash
# 11-12 user: X-VLA UMI-flatten WSD stable run on 300-demo S03G + 11-11 tag-free 29 (flatcum1111, 330 demos), from the 5-epoch checkpoint (E5_055694) to epoch 100,
# H100 SXM x4 TF32, global batch 16 (4 x 4, one micro-batch).  The stable run is continued later on the cumulative set (run.sh cstable).
# 1 epoch = 195,479 samples (1/3 of the 586,436 rows) -> 95 more epochs = 1,160,657 optimizer steps; save every SAVE_EP epochs.
set -e -o pipefail
U=/workspace/umiflat; export UV_CACHE_DIR=$U/uv_cache UV_PYTHON_INSTALL_DIR=$U/uv_python PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd $U/job && uv sync --frozen 2>&1 | tail -2
export U LR=$U/data/lerobot OUT=$U/ckpt BASE=$U/ckpt/E5_055694/pretrained_model NGPU=${NGPU:-4} PER=${PER:-4} MICRO=${MICRO:-${PER:-4}} NW=${NW:-10} LOGF=${LOGF:-100} XVLA_TF32=1
EP=195479; GB=16; ADD=$(( (95 * EP + GB - 1) / GB )); SAVE=$(( (${SAVE_EP:-10} * EP + GB - 1) / GB ))
mkdir -p $U/logs
# checkpoint backup to the US-CA-2 volume (S3 API) every 20 min when S3 keys are present (pods outside US-CA-2 have no volume)
if [ -f $HOME/.runpod_s3/secret ]; then ( while sleep 1200; do
    AWS_ACCESS_KEY_ID=$(cat $HOME/.runpod_s3/access_key_id) AWS_SECRET_ACCESS_KEY=$(cat $HOME/.runpod_s3/secret) AWS_DEFAULT_REGION=us-ca-2 \
      aws --endpoint-url https://s3api-us-ca-2.runpod.io s3 sync $OUT/${CRUN:-XVLA-UMIFLAT-WSD-CUM1111-TF32-GB16} \
      s3://wlbldpcve2/umiflat/ckpt/${CRUN:-XVLA-UMIFLAT-WSD-CUM1111-TF32-GB16} --only-show-errors --delete && echo "$(date +%T) s3 backup ok"
  done ) > $U/logs/s3_backup.log 2>&1 & fi
( while sleep 60; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; done ) > $U/logs/gpu.log 2>&1 &
CUM=flatcum1111_S03G_xvla_train CRUN=${CRUN:-XVLA-UMIFLAT-WSD-CUM1111-TF32-GB16} INIT=$BASE ADD=$ADD SAVE_STEPS=$SAVE STAGE=cstable bash run.sh
