#!/bin/bash
# 11-04 OA-refactor code (company main 7852b9c, src/laundry_training). laundry FLATTEN DiT on the BH cluster (gateway job, 2x RTX 4090 DDP on gpu-4090). Same UMI recipe as remote/rtx6000_flat1103_run.sh / FLAT149:
#   init = H100 663012, continuous jaw (GRIP_BCE=0), LR 3e-4 cosine over the run, DDIM 50/16, AUG, GRAD_CLIP=0, global batch 1024 (2 x 512 = 8 x 64 accum).
#   11-04: MASK_CROSS_GAP=1 (xyz masked from the first masked step of each chunk on); user: 100 epochs on 2x 4090.
# usage (ray job, --working-dir .):  STAGE=smoke|train DS=<dataset name under /mnt/shared/users/johann/data/lerobot> bash run.sh
set -e -o pipefail
U=/mnt/shared/users/johann; LR=${LR:-$U/data/lerobot}; OUT=${OUT:-$U/ckpt/ldt}; INIT=${INIT:-$U/legacy/ldt6000/checkpoints/init663012/pretrained_model}
PY="uv run --frozen python"; ACC="$PY -m accelerate.commands.launch"
NGPU=${NGPU:-2}; PER=${PER:-$((1024 / NGPU))}
export GRIP_STATE_MASK=1 GRIP_BCE=0 GRAD_ACCUM=${GRAD_ACCUM:-$((PER / ${MICRO:-32}))} GRAD_CLIP=0 METRICS_EVERY=10 ACCELERATE_MIXED_PRECISION=bf16 MASK_CROSS_GAP=1
export AUG=1 AUG_G_ROT=5 AUG_W_ROT=5 AUG_G_ZOOM=0.05 AUG_W_ZOOM=0.05 AUG_G_SHIFT=0.025 AUG_W_SHIFT=0.025 AUG_B=0.3 AUG_C=0.4 AUG_SAT=0.5 AUG_HUE=0.08 AUG_CH=0.0
export PYTHONPATH=$PWD/src:$PYTHONPATH HF_HOME=$U/scratch/hf
mkdir -p $OUT
export LR
if [ "$STAGE" = agg ]; then   # 11-04 night: per-take S03G sets of 11-03 + 11-04 -> one train / val / test (lerobot aggregate_datasets)
  AGG=${AGG:-flat1103_1104_S03G} SUF=${SUF:-S03G} $PY - <<'PY'
import json, os, pathlib, shutil
from lerobot.datasets.aggregate import aggregate_datasets
D = pathlib.Path(os.environ["LR"]); A = os.environ["AGG"]
for sp in ("train", "val", "test"):
    SUF = os.environ.get("SUF", "S03G")   # 11-05: SUF=S10G -> the speed-1.0 twins of the per-take sets
    roots = sorted(r for r in list(D.glob(f"flat1103n_*_{SUF}_{sp}")) + list(D.glob(f"flat1104_*_{SUF}_{sp}")) + list(D.glob(f"flat1105_*_{SUF}_{sp}")) if (r / "meta/info.json").exists())
    dst = D / f"{A}_{sp}"
    if dst.exists(): shutil.rmtree(dst)
    aggregate_datasets([f"local/{r.name}" for r in roots], f"local/{dst.name}", roots=roots, aggr_root=dst)
    eps = []
    for r in roots:
        s = r / "meta/split.json"
        if s.exists(): eps += json.loads(s.read_text())["episodes"].get(sp, [])
    (dst / "meta/split.json").write_text(json.dumps({"method": f"11-03 + 11-04 flatten {sp} (+ 11-05 when present): per-take {SUF} sets (Cg, {'1.0x' if SUF == 'S10G' else '0.3x'}, continuous jaw, gated mask 0.3 s, trimmed)",
                                                     "episodes": {s2: (eps if s2 == sp else []) for s2 in ("train", "val", "test")}}, indent=1))
    info = json.loads((dst / "meta/info.json").read_text()); print(dst.name, len(roots), "sets", info["total_episodes"], "episodes", info["total_frames"], "frames", len(eps), "split keys", flush=True)
PY
  exit 0
fi
DS=${DS:?DS}
# 11-05 FLAT-PGRASP: PGRASP=<version> -> jaw slots of the state carry the causal p_grasp side file (laundry_training.env_options.PGRASP); export it so a
# RESUME (config only) gets it too, and refuse to start without the side file
export PGRASP=${PGRASP:-}; [ -n "$PGRASP" ] && { [ -f $LR/$DS/meta/pgrasp_$PGRASP.npy ] || { echo "PGRASP=$PGRASP but $LR/$DS/meta/pgrasp_$PGRASP.npy missing"; exit 1; }; echo "[pgrasp] $PGRASP"; }
F=$($PY -c "import json;print(json.load(open('$LR/$DS/meta/info.json'))['total_frames'])")
EPOCH=$($PY -c "import math;print(math.ceil($F/1024))"); STEPS=$((EPOCH * ${EPOCHS:-100})); export SCHED_HORIZON=$STEPS
RUN=${RUN:-DITFT-H663012-${DS}-UMIRECIPE-B1024-E${EPOCHS:-100}-4090x2}
COMMON=(--policy.path=$INIT --policy.device=cuda --policy.push_to_hub=false --policy.optimizer_lr=3e-4
  --policy.noise_scheduler_type=DDIM --policy.num_train_timesteps=50 --policy.num_inference_steps=16
  --dataset.repo_id=local/$DS --dataset.root=$LR/$DS --batch_size=$PER --num_workers=${NW:-10} --log_freq=${LOGF:-10} --wandb.enable=false --seed=1000)
LAUNCH=(--num_processes $NGPU --mixed_precision bf16); [ $NGPU -gt 1 ] && LAUNCH=(--multi_gpu "${LAUNCH[@]}")
echo "[$STAGE] $DS frames $F -> epoch = $EPOCH updates, total $STEPS ($NGPU x $PER, accum $GRAD_ACCUM)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if [ "$STAGE" = smoke ]; then
  rm -rf $OUT/SMOKE_ddp; t0=$(date +%s)
  $ACC "${LAUNCH[@]}" -m laundry_training.train_dit_tagrig "${COMMON[@]}" --steps=${SMOKE_STEPS:-20} --save_freq=100000 --output_dir=$OUT/SMOKE_ddp --job_name=SMOKE_ddp > $U/scratch/smoke_ddp.log 2>&1 || { grep -vE "Warning|warn" $U/scratch/smoke_ddp.log | grep -B5 -A25 "Traceback" | head -80; exit 1; }; grep -E "step|loss|updt" $U/scratch/smoke_ddp.log | tail -25
  echo "SMOKE wall $(( $(date +%s) - t0 )) s for ${SMOKE_STEPS:-20} updates (incl. startup)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 0
fi
if [ "$STAGE" = train ]; then
  # 11-04 user: save EVERY epoch (--save_freq=$EPOCH). pretrained_model (~1.1 GB) kept for every epoch; training_state (optimizer) kept only on the
  # newest checkpoint (needed for RESUME=1 after a 48 h stop) -> ~110 GB for 100 epochs on /mnt/shared
  ( while sleep 120; do
      cs=$(ls -d $OUT/$RUN/checkpoints/[0-9]* 2>/dev/null | sort | head -n -2) || true   # keep training_state on the newest 2 (a stop mid-save leaves the newest incomplete)
      for c in $cs; do [ -d $c/training_state ] && rm -rf $c/training_state; done
      # 11-06 /mnt/shared full (ENOSPC at epoch 44) -> user: keep every KEEP_EVERY-th epoch + best val + newest 2 only. An epoch goes only once the
      # Mac eval watcher pushed it into evaluated.txt (its done_steps.txt) -- BEST_VAL_STEP comes from the same push
      if [ -n "$KEEP_EVERY" ] && [ -f $OUT/$RUN/evaluated.txt ]; then
        best=$(cut -c1-6 $OUT/$RUN/BEST_VAL_STEP 2>/dev/null)
        for c in $cs; do st=$(basename $c)
          grep -qx $st $OUT/$RUN/evaluated.txt || continue; [ "$st" = "$best" ] && continue; [ $(( 10#$st / EPOCH % KEEP_EVERY )) -eq 0 ] && continue
          rm -rf $c
        done
      fi
      [ -f $OUT/$RUN/train.done ] && break
    done ) &
  if [ "${RESUME:-0}" = 1 ]; then
    CK=$(for c in $(ls -d $OUT/$RUN/checkpoints/[0-9]* | sort -r); do [ -f $c/training_state/optimizer_param_groups.json ] && [ -f $c/pretrained_model/model.safetensors ] && { echo $c; break; }; done)   # newest COMPLETE checkpoint
    echo "[resume] from $CK (step $(cat $CK/training_state/training_step.json))"
    $ACC "${LAUNCH[@]}" -m laundry_training.train_dit_tagrig --config_path=$CK/pretrained_model/train_config.json --resume=true
  else
    [ -e $OUT/$RUN/checkpoints ] && [ "${FRESH:-0}" != 1 ] && { echo "$OUT/$RUN exists (RESUME=1?)"; exit 1; }; [ "${FRESH:-0}" = 1 ] && rm -rf $OUT/$RUN
    $ACC "${LAUNCH[@]}" -m laundry_training.train_dit_tagrig "${COMMON[@]}" --steps=$STEPS --save_freq=$EPOCH --output_dir=$OUT/$RUN --job_name=$RUN
  fi
  echo "$INIT" > $OUT/$RUN/finetuned_from.txt; touch $OUT/$RUN/train.done; echo "[train] done $(date)"
fi
