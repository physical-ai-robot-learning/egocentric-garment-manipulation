#!/bin/bash
# 11-10 user: X-VLA SoftFold 650k (SOFTFOLD-REL32BIN-RELONLY-D21-B8-1000K/0650000) -> UMI flatten S03G full SFT, new domain 22,
#   FP32 (no AMP, no TF32), continuous gripper (MSE, no BCE), jaw state 0 (GSM), NO grad clip (user rule), AdamW lr 1e-4 (VLM x0.1 via
#   xvla-adamw), wd 1e-4, betas 0.9/0.95, cosine warmup 1000 -> 2.5e-6 over the whole run, image aug off (= the SoftFold run).
#   Budget: max 100 epochs, 1 epoch = 178,219 samples (1/3 of the 534,657 0.3x frames, user 11-10); save every 5 epochs (5, 10, 20 ...
#   are evaluated; early stop on robot success).  Global batch = NGPU x PER per OPTIMIZER step; MICRO = samples per
#   forward/backward (gradient accumulation PER/MICRO in src/train_umiflat.py, 11-10 user): RunPod 4 x 4 (MICRO 4) == cluster 2 x 8 (MICRO 4) = 16.
# usage (ray job / pod, --working-dir .):  STAGE=smoke|train [RESUME=1] [NGPU=2] [PER=8] [MICRO=4] bash run.sh
set -e -o pipefail
U=${U:-/mnt/shared/users/johann}; LR=${LR:-$U/data/lerobot}; OUT=${OUT:-$U/ckpt/xvla}; BASE=${BASE:-$U/ckpt/xvla/base_sf650k_d22}
PY="uv run --frozen python"
NGPU=${NGPU:-2}; PER=${PER:-8}; export XVLA_MICRO=${MICRO:-2}; GB=$((NGPU * PER)); DSP=flat1103_1104_1105_S03G_xvla; DS=${DSP}_train
EPOCH_SAMPLES=178219; EPOCHS=${EPOCHS:-100}
STEPS=$(( (EPOCHS * EPOCH_SAMPLES + GB - 1) / GB )); SAVE=$(( (${SAVE_EP:-5} * EPOCH_SAMPLES + GB - 1) / GB ))
RUN=${RUN:-XVLA-SF650K-D22-UMIFLAT-S03G-REL32-GSM-FP32-GB${GB}-E${EPOCHS}}
export PYTHONPATH=$PWD/src:$PYTHONPATH HF_HOME=$U/scratch/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# videos = the DiT S03G sets' videos (same rows, same files): hard links, never copies
for sp in train val test; do
  case $STAGE in cstable|cdecay|ceval|cum) break;; esac   # 11-12: continual stages read $CUM / EVAL_ROOTS, not the 300-set
  src=$LR/flat1103_1104_1105_S03G_$sp
  [ -f $LR/${DSP}_$sp/meta/info.json ] || { [ $sp = train ] && { echo "missing $LR/${DSP}_$sp"; exit 1; }; echo "[warn] no $LR/${DSP}_$sp"; continue; }   # 11-11: val/test optional (RunPod smoke)
  [ -d $LR/${DSP}_$sp/videos ] || { [ -d $src/videos ] && cp -al $src/videos $LR/${DSP}_$sp/videos && echo "[videos] linked $src -> ${DSP}_$sp"; } || true
done
case $STAGE in cstable|cdecay|ceval|cum) ;; *) [ -d $LR/$DS/videos ] || { echo "no videos for $DS"; exit 1; };; esac
grep -q '"domain_id": 22' $BASE/policy_preprocessor.json || { echo "base $BASE is not domain 22"; exit 1; }
ARGS=(--policy.path=$BASE --policy.device=cuda --policy.push_to_hub=false --policy.dtype=float32
  --policy.optimizer_lr=1e-4 --policy.optimizer_weight_decay=1e-4 --policy.optimizer_grad_clip_norm=0
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=$STEPS --policy.scheduler_decay_lr=2.5e-6
  --dataset.repo_id=local/$DS --dataset.root=$LR/$DS --dataset.video_backend=pyav --dataset.image_transforms.enable=false
  --rename_map='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
  --tolerance_s=0.001 --batch_size=$PER --num_workers=${NW:-8} --log_freq=${LOGF:-100} --eval_freq=0 --wandb.enable=false --seed=1000)
LAUNCH=(--num_processes $NGPU --mixed_precision no); [ $NGPU -gt 1 ] && LAUNCH=(--multi_gpu "${LAUNCH[@]}")
ACC="$PY -m accelerate.commands.launch"
echo "[$STAGE] $DS: NGPU $NGPU x PER $PER (micro $XVLA_MICRO) = global $GB | epoch $EPOCH_SAMPLES samples = $(( EPOCH_SAMPLES / GB )) steps | $EPOCHS epochs = $STEPS steps | save every $SAVE (${SAVE_EP:-5} ep)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if [ "$STAGE" = smoke ]; then
  rm -rf $OUT/SMOKE_umiflat
  # 1) 60 updates, save at 50 (memory, speed, plugin preflight, save)  2) resume from 50 to 70 (the cross-hardware resume path)
  ( RELONLY_PREFLIGHT=3 $ACC "${LAUNCH[@]}" src/train_umiflat.py "${ARGS[@]}" --steps=${SMOKE_STEPS:-60} --save_freq=${SMOKE_SAVE:-50} \
      --output_dir=$OUT/SMOKE_umiflat --job_name=SMOKE_umiflat \
    && $ACC "${LAUNCH[@]}" src/train_umiflat.py --config_path=$OUT/SMOKE_umiflat/checkpoints/$(printf %06d ${SMOKE_SAVE:-50})/pretrained_model/train_config.json \
      --resume=true --steps=$(( ${SMOKE_SAVE:-50} + 20 )) --batch_size=$PER ) &
  pid=$!; while kill -0 $pid 2>/dev/null; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; sleep 30; done
  wait $pid; ls -la $OUT/SMOKE_umiflat/checkpoints/*/ 2>/dev/null; du -sh $OUT/SMOKE_umiflat/checkpoints/* 2>/dev/null; exit 0
fi
# ---- [11-11 user] daily continual schedule (src/continual.py): WSD stable run that is resumed every day on the cumulative set
#   (datasets/xvla_cumulative.py, frozen stats, meta/episode_day.json) + a short decay branch per day for eval / deploy.
#   CUM = cumulative root name under $LR (train split); CRUN = the stable run name (one per curriculum, never decayed)
#   cstable: first call INIT=<ckpt dir with pretrained_model> starts the run (fresh AdamW, warmup WARM); later calls resume the
#            newest stable checkpoint and add ADD optimizer steps on today's CUM (recent days RECENT_DAYS get RECENT_FRAC)
#   cdecay : DECAY steps of linear decay from the newest stable checkpoint into $OUT/${CRUN}-D<step> (never resumed by cstable)
#   ceval  : offline eval (src/eval_offline.py) of CK on EVAL_ROOTS (comma list of roots under $LR) -> $CK/eval_<root>.json
if [ "$STAGE" = cstable ] || [ "$STAGE" = cdecay ]; then
  : ${CUM:?CUM=<cumulative root>} ${CRUN:?CRUN=<stable run name>}
  [ -f $LR/$CUM/meta/episode_day.json ] || { echo "$LR/$CUM: no meta/episode_day.json (datasets/xvla_cumulative.py)"; exit 1; }
  export XVLA_SCHED=wsd XVLA_WSD_WARMUP=${WARM:-1000} XVLA_RECENT_DAYS=${RECENT_DAYS:-2} XVLA_RECENT_FRAC=${RECENT_FRAC:-0.5}
  DSA=(--dataset.repo_id=local/$CUM --dataset.root=$LR/$CUM)
  newest() { for c in $(ls -d $1/checkpoints/[0-9]* 2>/dev/null | sort -r); do [ -f $c/training_state/optimizer_param_groups.json ] && [ -f $c/pretrained_model/model.safetensors ] && { echo $c; break; }; done; }
  CK=$(newest $OUT/$CRUN)
fi
if [ "$STAGE" = cstable ]; then
  : ${ADD:?ADD=<optimizer steps to add today>}
  ( while sleep 300; do
      for c in $(ls -d $OUT/$CRUN/checkpoints/[0-9]* 2>/dev/null | sort | head -n -2); do [ -d $c/training_state ] && rm -rf $c/training_state; done
      [ -f $OUT/$CRUN/cstable.done ] && break
    done ) &
  rm -f $OUT/$CRUN/cstable.done
  if [ -z "$CK" ]; then
    : ${INIT:?no stable checkpoint in $OUT/$CRUN: INIT=<ckpt dir> for the first day}
    P=$INIT; [ -d $INIT/pretrained_model ] && P=$INIT/pretrained_model
    [ -e $OUT/$CRUN ] && { echo "$OUT/$CRUN exists without a resumable checkpoint"; exit 1; }
    echo "[cstable] NEW $CRUN from $P on $CUM: $ADD steps (warmup $XVLA_WSD_WARMUP)"
    ARGS0=(); for x in "${ARGS[@]}"; do case $x in --policy.path=*) ARGS0+=(--policy.path=$P);; --dataset.repo_id=*|--dataset.root=*) ;; *) ARGS0+=("$x");; esac; done
    $ACC "${LAUNCH[@]}" src/train_umiflat.py "${ARGS0[@]}" "${DSA[@]}" --steps=$ADD --save_freq=${SAVE_STEPS:-$ADD} --output_dir=$OUT/$CRUN --job_name=$CRUN
  else
    S0=$(cat $CK/training_state/training_step.json | grep -oE '[0-9]+' | head -1)
    echo "[cstable] RESUME $CRUN from $CK (step $S0) on $CUM: +$ADD -> $((S0 + ADD))"
    $ACC "${LAUNCH[@]}" src/train_umiflat.py --config_path=$CK/pretrained_model/train_config.json --resume=true --batch_size=$PER \
      "${DSA[@]}" --steps=$((S0 + ADD)) --save_freq=${SAVE_STEPS:-$ADD}
  fi
  touch $OUT/$CRUN/cstable.done; echo "[cstable] done $(date): $(newest $OUT/$CRUN)"; exit 0
fi
if [ "$STAGE" = cdecay ]; then
  : ${DECAY:?DECAY=<decay steps>}
  [ -n "$CK" ] || { echo "no stable checkpoint in $OUT/$CRUN"; exit 1; }
  S0=$(cat $CK/training_state/training_step.json | grep -oE '[0-9]+' | head -1); DR=$OUT/${CRUN}-D$S0
  [ -e $DR ] && { echo "$DR exists"; exit 1; }
  mkdir -p $DR/checkpoints && cp -al $CK $DR/checkpoints/$(basename $CK)
  export XVLA_WSD_DECAY_FROM=$S0 XVLA_WSD_DECAY_STEPS=$DECAY
  echo "[cdecay] $CK (step $S0) -> $DR: linear decay over $DECAY steps"
  $ACC "${LAUNCH[@]}" src/train_umiflat.py --config_path=$DR/checkpoints/$(basename $CK)/pretrained_model/train_config.json --resume=true \
    --batch_size=$PER "${DSA[@]}" --steps=$((S0 + DECAY)) --save_freq=$DECAY --output_dir=$DR --job_name=${CRUN}-D$S0
  rm -rf $DR/checkpoints/$(basename $CK)          # the hard-linked stable copy (the stable run keeps the original)
  echo "[cdecay] done $(date): $DR/checkpoints/$(printf %06d $((S0 + DECAY)))"; exit 0
fi
if [ "$STAGE" = cum ]; then   # CUM_OUT=<name> PARTS="root:YYYYMMDD root:YYYYMMDD ..." (roots under $LR, in order); stats frozen to FROZEN
  : ${CUM_OUT:?} ${PARTS:?}
  PA=(); for p in $PARTS; do r=${p%%:*}; [ -d $LR/$r/videos ] || { s=${r/_xvla_/_}; cp -al $LR/$s/videos $LR/$r/videos && echo "[videos] linked $s -> $r"; }; PA+=(--part $LR/$p); done
  $PY src/xvla_cumulative.py --out $LR/$CUM_OUT --stats-from $LR/${FROZEN:-flat1103_1104_1105_S03G_xvla_train} "${PA[@]}"; exit 0
fi
if [ "$STAGE" = ceval ]; then
  : ${CK:?CK=<checkpoint dir>} ${EVAL_ROOTS:?EVAL_ROOTS=<comma list of roots under $LR>}
  for r in ${EVAL_ROOTS//,/ }; do
    [ -d $LR/$r/videos ] || { s=${r/_xvla_/_}; [ -d $LR/$s/videos ] && cp -al $LR/$s/videos $LR/$r/videos; }
    $PY src/eval_offline.py --ckpt $CK --root $LR/$r --stride ${STRIDE:-15} --out $CK/eval_$r.json
  done; exit 0
fi
if [ "$STAGE" = train ]; then
  # training_state (FP32 AdamW ~7 GB) kept only on the newest 2 checkpoints (a stop mid-save leaves the newest incomplete)
  ( while sleep 300; do
      for c in $(ls -d $OUT/$RUN/checkpoints/[0-9]* 2>/dev/null | sort | head -n -2); do [ -d $c/training_state ] && rm -rf $c/training_state; done
      [ -f $OUT/$RUN/train.done ] && break
    done ) &
  if [ "${RESUME:-0}" = 1 ]; then
    CK=$(for c in $(ls -d $OUT/$RUN/checkpoints/[0-9]* | sort -r); do [ -f $c/training_state/optimizer_param_groups.json ] && [ -f $c/pretrained_model/model.safetensors ] && { echo $c; break; }; done)
    echo "[resume] from $CK (step $(cat $CK/training_state/training_step.json))"
    $ACC "${LAUNCH[@]}" src/train_umiflat.py --config_path=$CK/pretrained_model/train_config.json --resume=true --batch_size=$PER
  else
    [ -d $OUT/$RUN ] && [ -z "$(ls $OUT/$RUN/checkpoints 2>/dev/null)" ] && { echo "[train] $OUT/$RUN has no checkpoint (failed start) -> removed"; rm -rf $OUT/$RUN; }
    [ -e $OUT/$RUN/checkpoints ] && { echo "$OUT/$RUN exists (RESUME=1?)"; exit 1; }
    $ACC "${LAUNCH[@]}" src/train_umiflat.py "${ARGS[@]}" --steps=$STEPS --save_freq=$SAVE --output_dir=$OUT/$RUN --job_name=$RUN
  fi
  touch $OUT/$RUN/train.done; echo "[train] done $(date)"
fi
