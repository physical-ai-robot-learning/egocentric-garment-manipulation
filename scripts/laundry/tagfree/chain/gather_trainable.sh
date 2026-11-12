#!/bin/zsh
# 11-11: per-demo QA clips of the usable (GOLD: train / val / test) tag-free demos -> ~/Desktop/tagfree_trainable_1111 + one combined ALL.mp4
S=/Users/jeonghwanlee/tagfree_tools/work; PY=~/ego_collector/.venv/bin/python; O=~/Desktop/tagfree_trainable_1111; mkdir -p $O
for H in "$@"; do
  J=$S/flat1111tf_${H}_S03G_split.json; [ -f $J ] || { echo "no split $H"; continue; }
  $PY -c "
import json; e=json.load(open('$J'))['episodes']
for s in ('train','val','test'):
    for k in e[s]: print(s, int(k.split('demo')[1]))" | while read SP D; do
    F=$O/${SP}_20261111_${H}_d$(printf %02d $D).mp4
    $PY $S/render_tagfree.py 20261111_$H fus_Bg_tfL $F $D "{\"$D\": \"[$SP]\"}" 2>&1 | grep video
  done
done
ls $O/*_d*.mp4 | sort -t_ -k1,1 > $O/list.txt
( for f in $(cat $O/list.txt); do echo "file '$f'"; done ) > $O/concat.txt
ffmpeg -v error -y -f concat -safe 0 -i $O/concat.txt -c copy $O/ALL.mp4 && rm $O/concat.txt $O/list.txt && echo "ALL $(ls $O/*_d*.mp4 | wc -l) clips"
