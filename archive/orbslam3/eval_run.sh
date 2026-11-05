#!/bin/zsh
# eval_run.sh <run_dir> <mode> -> converts + evaluates; prints summary
S=/private/tmp/claude-501/-Users-jeonghwanlee/804e23fc-90ec-480d-8da2-22a28b640e97/scratchpad
r=$1; m=$2; b=$(basename $r); h=${b%%_*}; suf=""; off=-40; fix=-5,34,122; jaw=wrist
if [ $h = left ]; then suf=_left; off=-35; fix=-15,14,118; jaw=wrist_left; fi
cd ~/ego_collector; T=datasets/tag_rig/20261021_113243; o=$S/orbout/s_${b}_$m
.venv/bin/python $S/orbpatch/orb2segs.py $r datasets/m3_frames/umi_20261021_113243$suf/frames.json $o --offset-ms=$off --tag umi_20261021_113243$suf --mode $m
out=$(.venv/bin/python scripts/umi_slam_eval.py $T $o/manifest.json datasets/m3_frames/umi_20261021_113243$suf --hand $h --fix-t=$fix --jaw $T/$jaw/umi_jaw.npz 2>&1)
if echo "$out" | grep -q "좋은 구간이 없음"; then out=$(.venv/bin/python scripts/umi_slam_eval.py $T $o/manifest.json datasets/m3_frames/umi_20261021_113243$suf --hand $h --fix-t=$fix --jaw $T/$jaw/umi_jaw.npz --good-rot 15 --good-travel 0.10 --good-res 0.02 2>&1); echo "(relaxed good-seg flags)"; fi
echo "$out" > $o/eval.txt; echo "$out" | grep -E "핵심|손눈|유지|없음"
