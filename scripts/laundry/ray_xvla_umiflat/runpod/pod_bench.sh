#!/bin/bash
# 11-11 RunPod H100x4 smoke + throughput for the continual X-VLA schedule (ray_jobs/xvla_umiflat run.sh stages cstable/cdecay/ceval)
# on the existing 300-demo S03G X-VLA set.  Volume layout (/workspace = network volume wlbldpcve2):
#   /workspace/umiflat/{job (run.sh + src + pyproject/uv.lock + .venv), data/lerobot/<sets>, ckpt/E5_055694/pretrained_model, uv_cache}
set -e -o pipefail
U=/workspace/umiflat; export UV_CACHE_DIR=$U/uv_cache UV_PYTHON_INSTALL_DIR=$U/uv_python PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd $U/job && uv sync --frozen 2>&1 | tail -2
export U LR=$U/data/lerobot OUT=$U/ckpt BASE=$U/ckpt/E5_055694/pretrained_model NGPU=${NGPU:-4} PER=${PER:-4} MICRO=${MICRO:-4} NW=${NW:-8} LOGF=${LOGF:-20}
C="CUM=flat1103_1104_1105_S03G_xvla_train"
L=$U/logs; mkdir -p $L
rm -rf $OUT/BENCH_TF32 $OUT/BENCH_TF32-D* $OUT/BENCH_FP32
echo "== 1 TF32 new stable run from E5: ${BENCH_STEPS:-300} steps"; env $C CRUN=BENCH_TF32 INIT=$BASE ADD=${BENCH_STEPS:-300} XVLA_TF32=1 STAGE=cstable bash run.sh 2>&1 | tee $L/1_tf32.log | grep -vE "^\s*$" | tail -5
echo "== 2 TF32 resume +60 (next-day path)"; env $C CRUN=BENCH_TF32 ADD=60 XVLA_TF32=1 STAGE=cstable bash run.sh 2>&1 | tee $L/2_resume.log | tail -3
echo "== 3 TF32 decay branch 40"; env $C CRUN=BENCH_TF32 DECAY=40 XVLA_TF32=1 STAGE=cdecay bash run.sh 2>&1 | tee $L/3_decay.log | tail -3
echo "== 4 FP32 strict 120 steps (speed reference)"; env $C CRUN=BENCH_FP32 INIT=$BASE ADD=120 XVLA_TF32=0 STAGE=cstable bash run.sh 2>&1 | tee $L/4_fp32.log | tail -3
D=$(ls -d $OUT/BENCH_TF32-D*/checkpoints/[0-9]* | tail -1)
echo "== 5 offline eval $D on val (stride ${EVAL_STRIDE:-300})"; CK=$D EVAL_ROOTS=flat1103_1104_1105_S03G_xvla_val STRIDE=${EVAL_STRIDE:-300} STAGE=ceval NGPU=1 bash run.sh 2>&1 | tee $L/5_eval.log | grep "\[eval\]" | tail -5
CK=$BASE EVAL_ROOTS=flat1103_1104_1105_S03G_xvla_val STRIDE=${EVAL_STRIDE:-300} STAGE=ceval NGPU=1 bash run.sh 2>&1 | tee $L/5_eval_E5.log | grep "\[eval\]" | tail -5
echo "BENCH_DONE"
