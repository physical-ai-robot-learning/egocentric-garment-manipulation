#!/bin/zsh
# 11-11 tag-free take -> S03G per-take LeRobot sets (same chain as the pilot 20261111_121020).
# usage: tagfree_take_chain.sh TAKE PACKNAME   (e.g. 20261111_124743 ctpackTF124743)
set -e
T=$1; PK=$2; S=/Users/jeonghwanlee/tagfree_tools/work
PY=~/ego_collector/.venv/bin/python; PYL=~/xvla-mac/bin/python; B=~/ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b; L=~/laundry_dit_training
TR=~/ego_collector/datasets/tag_rig; DC=data/tag_rig_derived_tagfree; F="grep -v -i -E warn|hand-eye|Ds = np"
log() { echo "[$(date '+%F %T')] $*"; }
log "1 prep (tagfree.json, assignment, IMU idle trim, L seeds) + jaw"
cd $B && $PY tagfree_prep.py $T 2>&1 | grep -v -i -E "warn|hand-eye|Ds = np" | tail -3
cd ~/ego_collector && $PY scripts/jaw_from_motor.py $TR/$T 2>&1 | grep -v -i warn
DEMOS=($($PYL -c "
import csv; print(' '.join(r['demo'] for r in csv.DictReader(open('$L/reports/idle_trim.csv')) if r['take']=='$T' and r['demo'] not in '${TF_EXCLUDE:-}'.split(',')))"))   # TF_EXCLUDE=0,3: demos that did not start at the L
log "demos ${DEMOS[*]}"
[ ${#DEMOS} -gt 0 ] && [ -f $B/start_seed_$T.json ] || { log "TF_TAKE_FAILED prep (no idle-trim rows or no seeds)"; exit 1; }
log "2 v31 tracker + yaw pass 1"
cd $B && V31_P='{"JUMP":0.10,"V_GATE":1.0}' V31_OUT=trackv31speed $PY track_v31.py $T 2>&1 | grep -v -i warn | tail -1
$PY tagfree_yaw.py $T 2>&1 | grep -E "^\[L\]" | tail -40
log "3 CoTracker pack + cluster"
rm -rf $S/$PK; cd $B/cotrack && $PY make_pack2.py $S/$PK ${DEMOS/#/$T:} 2>&1 | grep -v -i warn | tail -2
bash $L/ray_jobs/cotracker/submit.sh $S/$PK
NT=$(ls $S/$PK/*/tracks.npz 2>/dev/null | wc -l | tr -d " "); [ "$NT" = "${#DEMOS}" ] || { log "TF_TAKE_FAILED tracks $NT/${#DEMOS}"; exit 1; }
log "4 solve2 + gate, yaw pass 2 (solve3), solve2 + gate again, fusion"
cd $B/cotrack && $PY solve2.py $S/$PK > $S/solve2_$PK.log 2>&1 && $PY gate.py $S/$PK > /dev/null 2>&1
cd $B && $PY tagfree_yaw.py $T --pack $S/$PK 2>&1 | grep -E "^\[L\]"
cd $B/cotrack && $PY solve2.py $S/$PK > $S/solve2_$PK.log 2>&1 && $PY gate.py $S/$PK 2>&1 | grep -E "h-1|veto"
cd $B/fusion && $PY trajectory_fusion_v1.py $S/$PK R,A,Bg --ct solve3 --suffix _tfL 2>&1 | grep -c lag
log "5 derive + IK + quality"
cd $B && $PY tagfree_derive.py $T $S/$PK 2>&1 | grep held
cd $L && $PYL datasets/robotize_ik.py $TR/$T $DC/$T --demos ${(j:,:)DEMOS} --workers 6 --force > $DC/$T/robotize_ik.log 2>&1; grep -E "^demo" $DC/$T/robotize_ik.log | cut -c1-200
cd $B/cotrack && $PY episode_quality.py $S/q_$PK.csv $S/$PK 2>&1 | tail -1
log "6 build S03G (GOLD gap<0.25 train, GOLD gap>=0.25 val/test alternating, SILVER silver, REJECT reject)"
KEYS=($($PYL $S/tagfree_split.py $S/q_$PK.csv $S/flat${T:4:4}tf_${T:9}_S03G_split.json))
cd $L && FILL_MASK_S=0.3 $PYL datasets/tag_rig_unsplit.py --label fused --speed 0.3 --mask-mode gated --grasp-fill-drop off --ik-fail-mm 30 --render 0 --overwrite --grip-label cont \
  --out data/lerobot/flat${T:4:4}tf_${T:9}_S03G --takes $TR/$T --derived $DC --demos ${KEYS[@]} --old-split $S/flat${T:4:4}tf_${T:9}_S03G_split.json > $S/build_flat${T:4:4}tf_${T:9}_S03G.log 2>&1
grep -E "^done|assert" $S/build_flat${T:4:4}tf_${T:9}_S03G.log
log "7 QA video -> Desktop"
mkdir -p ~/Desktop/tagfree_${T:4:4} && $PY $S/render_tagfree.py $T fus_Bg_tfL ~/Desktop/tagfree_${T:4:4}/${T}_tagfree_Bg.mp4 2>&1 | grep video
log "TF_TAKE_DONE $T"
