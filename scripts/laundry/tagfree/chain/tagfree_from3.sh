#!/bin/zsh
# 11-11 resume from step 3 (re-pack + CoTracker) without re-running prep (keeps a hand-edited tagfree_L json). usage: TAKE PACK
set -e
T=$1; PK=$2; S=/Users/jeonghwanlee/tagfree_tools/work
PY=~/ego_collector/.venv/bin/python; PYL=~/xvla-mac/bin/python; B=~/ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b; L=~/laundry_dit_training
DEMOS=($($PYL -c "
import csv; print(' '.join(r['demo'] for r in csv.DictReader(open('$L/reports/idle_trim.csv')) if r['take']=='$T' and r['demo'] not in '${TF_EXCLUDE:-}'.split(',')))"))   # TF_EXCLUDE=0,3: demos that did not start at the L
echo "[$(date '+%F %T')] 3 CoTracker pack + cluster (from3)"
rm -rf $S/$PK; cd $B/cotrack && $PY make_pack2.py $S/$PK ${DEMOS/#/$T:} 2>&1 | grep -v -i warn | tail -2
bash $L/ray_jobs/cotracker/submit.sh $S/$PK
NT=$(ls $S/$PK/*/tracks.npz 2>/dev/null | wc -l | tr -d " "); [ "$NT" = "${#DEMOS}" ] || { echo "TF_TAKE_FAILED tracks $NT/${#DEMOS}"; exit 1; }
zsh $S/tagfree_take_chain_from4.sh $T $PK
