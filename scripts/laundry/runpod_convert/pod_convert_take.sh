#!/bin/bash
# RunPod version of scratchpad/fl1102c_chain.sh (11-02 tag_rig -> Cg derived + robotize + episode quality), no Ray, nproc-aware.
# 11-04 refactor: the fixed-camera pose steps (1, 3, 4, 5, 6, 7) run the repository package laundry_tracking (python -m ..., packed by
# pack_code.sh under the job HOME, PYTHONPATH = its src/); jaw_from_motor / tag_rig_to_lerobot / robotize_ik and the CPU CoTracker
# fallback (run_cotracker_mps.py) still run from the packed ego_collector / laundry_dit_training trees.
# Runs on a pod that mounts the network volume prepared by pod_setup.sh.  Every job gets its own HOME copy of the code template, so
# the files the chain writes INSIDE the code tree (idle_trim.csv append, start_seed_<take>.json, v2b/out, _fgo_1023/tagrot,
# stitch/qa, data/tag_rig_derived*) never collide between jobs/pods.  Raw takes are shared read-mostly from $ROOT/raw/tag_rig;
# steps 0/1 write umi_jaw.npz / umi_traj_*.npz / segments.json into the raw take dir (same as on the Mac) -> never put one take in
# two concurrent jobs.
#
# usage (on the pod):
#   PACK=ctpackFL1102X TAKES="20261102_133624 20261102_134434" bash pod_convert_take.sh
# env:
#   PACK      unique job / CoTracker pack name (required).  keys file name = keys_<lower(PACK without 'ctpack')>_gs.txt
#   TAKES     space separated take ids (required); each must exist under $ROOT/raw/tag_rig/<take> (upload_take.sh)
#   DEMOS     per-take 0-based demo lists, "take=0,1,2 take2=all" (default all = every idle-trim row of the take after step 3)
#   DC        Cg derived root name under laundry_dit_training/data (default tag_rig_derived_cgg_1102)
#   STAGES    all (default) | comma list of pre,ct,post   pre = steps 0-4a (CPU), ct = 4b CoTracker (GPU), post = 5-7 + collect (CPU)
#             -> split mode: run STAGES=pre on a CPU pod, STAGES=ct on the GPU pod, STAGES=post on a CPU pod (same volume, same PACK).
#   WORKERS   parallel workers for the per-take / per-shard stages (default: cgroup CPU quota or nproc)
#   IK_WORKERS  robotize_ik.py --workers per take in step 6 (default 6 like the Mac chain); prep (step 2) has 6 hard-coded inside
#   ROBOT_CAL   step 1 --robot-cal (default configs/calibration/robot_cam_v005.json, as fl1102c)
#   CT_ALLOW_CPU=1  run CoTracker on CPU (run_cotracker_mps.py picks cuda/mps/cpu) when no GPU is visible; default: fail without GPU
#   CUDA_VISIBLE_DEVICES  passed through to the CoTracker step (default unset = all GPUs, CoTracker uses device 0)
#   SKIP_DONE=1 solve2 resume (skips demos that already have solve2_{left,right}.npz)
#   ROOT      volume root (default /workspace/rp_convert)
set -uo pipefail
ROOT=${ROOT:-/workspace/rp_convert}
: "${PACK:?set PACK}"; : "${TAKES:?set TAKES}"
DC=${DC:-tag_rig_derived_cgg_1102}; STAGES=${STAGES:-all}; ROBOT_CAL=${ROBOT_CAL:-configs/calibration/robot_cam_v005.json}
ncpu() { local q p; if read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then echo $(( (q + p - 1) / p ));
         elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" -gt 0 ]; then
           echo $(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) )); else nproc; fi; }
WORKERS=${WORKERS:-$(ncpu)}; [ "$WORKERS" -ge 1 ] || WORKERS=1; IK_WORKERS=${IK_WORKERS:-6}
JOB=$ROOT/jobs/$PACK; JH=$JOB/home; S=$JOB; PKD=$JOB/$PACK; LOGD=$JOB/logs; mkdir -p "$LOGD"
TAG=$(echo "${PACK#ctpack}" | tr '[:upper:]' '[:lower:]'); KEYS=$JOB/keys_${TAG}_gs.txt; QCSV=$JOB/q_$PACK.csv
PY=$ROOT/envs/ego/bin/python; IKPY=$ROOT/envs/ik/bin/python; CTPY=$ROOT/envs/ct/bin/python
export HOME=$JH LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
E=$JH/ego_collector; IC=$E/datasets/m3_runs/_imu_consistency; B=$IC/orbbec/v2b; L=$JH/laundry_dit_training; TR=$E/datasets/tag_rig
read -ra NEWT <<< "$TAKES"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGD/chain.log"; }
want() { [ "$STAGES" = all ] || [[ ",$STAGES," == *",$1,"* ]]; }
par() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do wait -n 2>/dev/null || true; done; }   # par N: block until < N bg jobs
declare -A DM
for kv in ${DEMOS:-}; do DM[${kv%%=*}]=${kv#*=}; done

# ------------------------------------------------------------------ job HOME (once per job; later stages / other pods reuse it)
if [ ! -d "$JH/ego_collector" ]; then
  log "create job home $JH from $ROOT/code/home"
  mkdir -p "$JH"
  rsync -rlt --exclude=/.cache --exclude=/handumi-sw "$ROOT/code/home/" "$JH/"
  ln -sfn "$ROOT/code/home/.cache" "$JH/.cache"; ln -sfn "$ROOT/code/home/handumi-sw" "$JH/handumi-sw"
  ln -sfn "$ROOT/envs/ik" "$JH/xvla-mac"; ln -sfn "$ROOT/envs/ego" "$E/.venv"
  mkdir -p "$TR" "$E/datasets/m3_runs/_fgo_1023/tagrot" "$IC/stitch/qa" "$B/out" "$L/data"
  printf 'PACK=%s\nTAKES=%s\nDEMOS=%s\nDC=%s\nROBOT_CAL=%s\ncreated=%s\ncode=%s\n' "$PACK" "$TAKES" "${DEMOS:-}" "$DC" "$ROBOT_CAL" "$(date '+%F %T')" \
    "$(readlink -f "$ROOT/code/home")" > "$JOB/job.env"
fi
for t in "${NEWT[@]}"; do
  [ -d "$ROOT/raw/tag_rig/$t" ] || { log "FAILED raw take missing: $ROOT/raw/tag_rig/$t"; exit 1; }
  ln -sfn "$ROOT/raw/tag_rig/$t" "$TR/$t"
done
# repository src/ shipped by pack_code.sh (laundry_tracking, laundry_common, laundry_shared): every laundry_tracking step imports from it
SRCP=$(dirname "$(find "$JH" -maxdepth 5 -type d -path '*/src/laundry_tracking' -not -path '*/.cache/*' 2>/dev/null | head -1)")
[ -d "$SRCP/laundry_tracking" ] || { log "FAILED repository src/ (laundry_tracking) not found under $JH (re-run pack_code.sh)"; exit 1; }
export PYTHONPATH=$SRCP${PYTHONPATH:+:$PYTHONPATH}
log "PACK=$PACK takes=${#NEWT[@]} stages=$STAGES workers=$WORKERS ik_workers=$IK_WORKERS src=$SRCP"

demos_of() {   # take -> space separated 0-based demo list (DEMOS entry, or every idle-trim row of the take)
  local t=$1 d=${DM[$1]:-all}
  if [ "$d" = all ]; then "$PY" -c "
import csv,sys; print(' '.join(sorted({r['demo'] for r in csv.DictReader(open('$L/reports/idle_trim.csv')) if r['take']=='$t'}, key=int)))"
  else echo "${d//,/ }"; fi
}

# ================================================================== pre: steps 0-4a (CPU)
if want pre; then
  cd "$E" || exit 1
  log "0 servo jaw"
  "$PY" scripts/jaw_from_motor.py "${NEWT[@]/#/$TR/}" 2>&1 | grep -v -i warn | tee "$LOGD/0_jaw.log"
  log "1 tag traj (robot cal $ROBOT_CAL)"
  for t in "${NEWT[@]}"; do for h in 오른손 왼손; do
    par "$WORKERS"
    ( "$PY" -m laundry_tracking.calib.umi_tcp_traj "datasets/tag_rig/$t" --hand "$h" --rot imu --imu-fill --robot-cal "$ROBOT_CAL" \
        --rig configs/calibration/umi_tag_rig_v002.json > "$LOGD/1_traj_${t}_$h.log" 2>&1
      grep -E "^(오른손|왼손):|rror" "$LOGD/1_traj_${t}_$h.log" | cut -c1-140 | sed "s/^/$t /" ) &
  done; done; wait
  log "2 prep (assignment + tag-only robotize_ik, 6 IK workers per take inside)"
  cd "$L" || exit 1
  for t in "${NEWT[@]}"; do
    par $(( WORKERS / 6 > 0 ? WORKERS / 6 : 1 ))
    ( "$IKPY" datasets/tag_rig_to_lerobot.py --out "$S/prep_dummy_$t" --takes "$TR/$t" --prep-only > "$LOGD/2_prep_$t.log" 2>&1
      grep -E "^\[prep\]|rror|CONFLICT" "$LOGD/2_prep_$t.log" | cut -c1-160 ) &
  done; wait
  log "3 trims, seeds, tagrot, tracker"
  cd "$B" || exit 1
  for t in "${NEWT[@]}"; do "$PY" -m laundry_tracking.new_take_prep "$t" 2>&1 | tee "$LOGD/3_prep_$t.log" | grep -E "seeds|rror" | cut -c1-160; done   # sequential: appends idle_trim.csv
  cd "$IC" || exit 1
  for t in "${NEWT[@]}"; do par "$WORKERS"; ( "$PY" -m laundry_tracking.tag_rotations "$TR/$t" > "$LOGD/3_tagrot_$t.log" 2>&1; tail -1 "$LOGD/3_tagrot_$t.log" | cut -c1-120 ) & done; wait
  cd "$B" || exit 1
  for t in "${NEWT[@]}"; do par "$WORKERS"
    ( V31_P='{"JUMP":0.10,"V_GATE":1.0}' V31_OUT=trackv31speed "$PY" -m laundry_tracking.orbbec_tracker "$t" > "$LOGD/3_track_$t.log" 2>&1; grep -v -i warn "$LOGD/3_track_$t.log" | tail -1 | cut -c1-160 ) &
  done; wait
  DEMO_SPECS=(); OKT=()
  for t in "${NEWT[@]}"; do
    if [ -f "$L/data/tag_rig_derived/$t/assignment.json" ] && [ -f "$B/start_seed_$t.json" ] && [ -f "$B/out/trackv31speed_$t.npz" ]; then
      OKT+=("$t"); for i in $(demos_of "$t"); do DEMO_SPECS+=("$t:$i"); done
    else log "SKIP $t (prep, seed or track missing)"; fi
  done
  [ ${#DEMO_SPECS[@]} -gt 0 ] || { log "${PACK}_FAILED no demos after prep"; exit 1; }
  printf '%s\n' "${OKT[@]}" > "$JOB/ok_takes.txt"; printf '%s\n' "${DEMO_SPECS[@]}" > "$JOB/demo_specs.txt"
  log "4a CoTracker pack ${#DEMO_SPECS[@]} demos (${#OKT[@]} takes) -> $PKD"
  cd "$B/cotrack" || exit 1
  for t in "${OKT[@]}"; do par "$WORKERS"
    ( "$PY" -m laundry_tracking.cotrack.make_pack "$PKD" $(grep "^$t:" "$JOB/demo_specs.txt") > "$LOGD/4a_pack_$t.log" 2>&1; grep -v -i warn "$LOGD/4a_pack_$t.log" | tail -2 ) &
  done; wait
  log "PRE_DONE $(ls -d "$PKD"/*/ 2>/dev/null | wc -l) pack dirs"
fi

# ================================================================== ct: step 4b CoTracker3 online (GPU)
if want ct; then
  [ -s "$JOB/demo_specs.txt" ] || { log "${PACK}_FAILED ct: run STAGES=pre first"; exit 1; }
  cd "$B/cotrack" || exit 1
  if nvidia-smi -L >/dev/null 2>&1; then CT="-m laundry_tracking.cotrack.run_cotracker"
  elif [ "${CT_ALLOW_CPU:-0}" = 1 ]; then CT=run_cotracker_mps.py; log "WARN no GPU -> CoTracker on CPU (slow)"
  else log "${PACK}_FAILED ct: no GPU visible (set CT_ALLOW_CPU=1 to run on CPU)"; exit 1; fi
  log "4b CoTracker ($CT, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
  # $CT unquoted on purpose: "-m <module>" is two words; run_cotracker_mps.py is a file in $B/cotrack (cwd)
  "$CTPY" $CT "$PKD" 2>&1 | grep -v -i warn | tee "$LOGD/4b_cotracker.log"     # skips dirs that already have tracks.npz (resume-safe)
  log "tracks $(ls "$PKD"/*/tracks.npz 2>/dev/null | wc -l)/$(wc -l < "$JOB/demo_specs.txt")"
fi

# ================================================================== post: steps 5-7 + collect (CPU)
if want post; then
  [ -s "$JOB/demo_specs.txt" ] || { log "${PACK}_FAILED post: run STAGES=pre first"; exit 1; }
  mapfile -t OKT < "$JOB/ok_takes.txt"; ND=$(ls -d "$PKD"/*/ | wc -l); NT=$(ls "$PKD"/*/tracks.npz 2>/dev/null | wc -l)
  [ "$NT" -gt 0 ] || { log "${PACK}_FAILED post: no tracks.npz (run STAGES=ct)"; exit 1; }
  NS=$(( WORKERS < ND ? WORKERS : ND ))
  log "5 solve2 ($NS shards, $NT/$ND demos tracked), gate, fusion R,Cg"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one process per core
  rm -rf "$JOB/${PACK}"_s*
  i=0; for d in "$PKD"/*/; do k=$((i % NS)); mkdir -p "$JOB/${PACK}_s$k"; ln -sfn "${d%/}" "$JOB/${PACK}_s$k/$(basename "$d")"; i=$((i+1)); done
  cd "$B/cotrack" || exit 1
  for ((k=0; k<NS; k++)); do
    ( "$PY" -m laundry_tracking.cotrack.point_solver "$JOB/${PACK}_s$k" > "$LOGD/5_solve2_$k.log" 2>&1
      "$PY" -m laundry_tracking.cotrack.gate "$JOB/${PACK}_s$k" 2>&1 | grep -v -i warn | grep -v " 0.0% tag  0.0% lr  0.0%" ) &
  done; wait
  cd "$B/fusion" || exit 1
  for ((k=0; k<NS; k++)); do "$PY" -m laundry_tracking.fusion.trajectory_fusion "$JOB/${PACK}_s$k" R,Cg --ct solve3 --suffix _g > "$LOGD/5_fus_$k.log" 2>&1 & done; wait
  unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
  log "solve3 $(ls "$PKD"/*/solve3_left.npz 2>/dev/null | wc -l)/$ND"
  log "6 Cg derived ($DC) + robotize"
  CGG_SRC=tag_rig_derived CGG_DST="$DC" "$PY" -m laundry_tracking.fusion.make_derived "$PKD" 2>&1 | grep -v -i warn | tee "$LOGD/6_cgg.log" | tail -3
  cd "$L" || exit 1
  for t in "${OKT[@]}"; do
    [ -d "data/$DC/$t" ] || { log "$t no derived dir"; continue; }
    par $(( WORKERS / IK_WORKERS > 0 ? WORKERS / IK_WORKERS : 1 ))
    dl=$(grep "^$t:" "$JOB/demo_specs.txt" | cut -d: -f2 | paste -sd, -)
    ( "$IKPY" datasets/robotize_ik.py "$TR/$t" "data/$DC/$t" --demos "$dl" --workers "$IK_WORKERS" > "data/$DC/$t/robotize_ik.log" 2>&1
      echo "$t ik $(ls "data/$DC/$t/ik_v1/"*.npz 2>/dev/null | wc -l)" ) &
  done; wait
  log "7 quality"
  cd "$B/cotrack" || exit 1
  "$PY" -m laundry_tracking.cotrack.episode_quality "$QCSV" "$PKD" 2>&1 | tee "$LOGD/7_quality.log" | tail -3
  "$PY" -c "
import pandas as pd; q = pd.read_csv('$QCSV', dtype={'take': str}); print(q['class'].value_counts().to_dict()); print(q[q['class'] != 'GOLD'][['take','demo','class','why']].to_string(index=False))
k = [f'{r.take}/demo{int(r.demo):02d}' for r in q.itertuples() if getattr(r, 'class') in ('GOLD', 'SILVER')]
open('$KEYS', 'w').write(' '.join(k)); print('GOLD+SILVER', len(k))" 2>&1 | tee -a "$LOGD/7_quality.log"

  log "collect -> $JOB/results"
  R=$JOB/results; rm -rf "$R"; mkdir -p "$R/home" "$R/pack/$PACK"
  rel() { echo "${1#$JH/}"; }
  cp_rel() { local f=$1; [ -e "$f" ] || return 0; mkdir -p "$R/home/$(dirname "$(rel "$f")")"; cp -L --preserve=timestamps "$f" "$R/home/$(rel "$f")"; }
  for t in "${NEWT[@]}"; do
    for f in umi_traj_left.npz umi_traj_right.npz segments.json wrist/umi_jaw.npz wrist_left/umi_jaw.npz; do cp_rel "$TR/$t/$f"; done
    for d in "$L/data/tag_rig_derived/$t" "$L/data/$DC/$t"; do [ -d "$d" ] && { mkdir -p "$R/home/$(rel "$d")"; rsync -rlt "$d/" "$R/home/$(rel "$d")/"; }; done
    cp_rel "$B/start_seed_$t.json"; cp_rel "$B/out/trackv31speed_$t.npz"; cp_rel "$E/datasets/m3_runs/_fgo_1023/tagrot/$t.npz"
    for v in fus_R_g fus_Cg_g; do for f in "$IC/stitch/qa/$v/${t}_"*.npz; do cp_rel "$f"; done; done
  done
  "$PY" -c "
import csv; T = set('$TAKES'.split()); rows = [r for r in csv.DictReader(open('$L/reports/idle_trim.csv')) if r['take'] in T]
w = csv.DictWriter(open('$R/idle_trim_rows.csv', 'w', newline=''), fieldnames=['take','demo','demo_len_s','keep_start_s','keep_end_s','cut_head_s','cut_tail_s','kept_s','pose_src'])
w.writeheader(); [w.writerow({k: r.get(k, '') for k in w.fieldnames}) for r in rows]; print('idle_trim rows', len(rows))"
  rsync -rlt --exclude=frames/ "$PKD/" "$R/pack/$PACK/"
  cp --preserve=timestamps "$QCSV" "$KEYS" "$JOB/job.env" "$JOB/demo_specs.txt" "$R/" 2>/dev/null; rsync -rlt "$LOGD/" "$R/logs/"
  log "${PACK}_CG_DONE results $(du -sh "$R" | cut -f1)"
fi
