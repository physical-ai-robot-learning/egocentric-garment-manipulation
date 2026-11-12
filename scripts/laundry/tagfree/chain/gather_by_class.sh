#!/bin/zsh
# 11-12 user "영상을 golden silver reject을 따로 모아줘": per-demo QA clips of a day's tag-free takes, sorted by episode_quality class
# -> ~/Desktop/tagfree_<MMDD>_by_class/{GOLD,SILVER,REJECT}/<take>_dNN.mp4 + one ALL.mp4 per class.  GOLD clips carry their split.
# usage: gather_by_class.sh TAKE [TAKE ...]      (q_ctpackTF<HHMMSS>.csv + flat<MMDD>tf_<HHMMSS>_S03G_split.json in this folder)
S=$HOME/tagfree_tools/work; PY=~/ego_collector/.venv/bin/python; PYL=~/xvla-mac/bin/python
O=~/Desktop/tagfree_${1:4:4}_by_class; mkdir -p $O/{GOLD,SILVER,REJECT}
for T in "$@"; do
  H=${T:9}; Q=$S/q_ctpackTF$H.csv; J=$S/flat${T:4:4}tf_${H}_S03G_split.json
  [ -f $Q ] || { echo "no quality csv for $T"; continue; }
  rm -f $O/*/${T}_d*.mp4
  $PYL -c "
import json, pandas as pd
q = pd.read_csv('$Q', dtype={'take': str}); sp = {}
try:
    for s, ks in json.load(open('$J'))['episodes'].items():
        for k in ks: sp[int(k.split('demo')[1])] = s
except FileNotFoundError: pass
for _, r in q.iterrows(): print(r['class'], int(r['demo']), sp.get(int(r['demo']), '-'))" | while read CL D SP; do
    $PY $S/render_tagfree.py $T fus_Bg_tfL $O/$CL/${T}_d$(printf %02d $D).mp4 $D "{\"$D\": \"[$CL $SP]\"}" 2>&1 | grep -c video > /dev/null
  done
done
for CL in GOLD SILVER REJECT; do
  ls $O/$CL/*_d*.mp4 2>/dev/null | sort > $O/$CL/list.txt
  if [ -s $O/$CL/list.txt ]; then
    ( for f in $(cat $O/$CL/list.txt); do echo "file '$f'"; done ) > $O/$CL/concat.txt
    ffmpeg -v error -y -f concat -safe 0 -i $O/$CL/concat.txt -c copy $O/$CL/ALL_$CL.mp4
  fi
  rm -f $O/$CL/list.txt $O/$CL/concat.txt
  echo "$CL $(ls $O/$CL/*_d*.mp4 2>/dev/null | wc -l | tr -d ' ') clips"
done
