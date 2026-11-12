#!/bin/zsh
# 11-12: re-run steps 4-7 of today's takes with the tag-free per-arm collision blame (gate.py), then the per-class QA clips
cd $HOME/tagfree_tools/work
for T in 131124 132300; do until grep -qE "TF_TAKE_DONE|TF_TAKE_FAILED|^exit" tagfree_$T.log; do sleep 30; done; done
for T in 124926 125954 131124 132300; do cp q_ctpackTF$T.csv q_ctpackTF$T.oldgate.csv 2>/dev/null; done
( TF_EXCLUDE=0 nice -n 10 zsh tagfree_take_chain_from4.sh 20261112_124926 ctpackTF124926 > rg_124926.log 2>&1; echo "exit $?" >> rg_124926.log ) &
( nice -n 10 zsh tagfree_take_chain_from4.sh 20261112_125954 ctpackTF125954 > rg_125954.log 2>&1; echo "exit $?" >> rg_125954.log ) &
wait
( nice -n 10 zsh tagfree_take_chain_from4.sh 20261112_131124 ctpackTF131124 > rg_131124.log 2>&1; echo "exit $?" >> rg_131124.log ) &
( nice -n 10 zsh tagfree_take_chain_from4.sh 20261112_132300 ctpackTF132300 > rg_132300.log 2>&1; echo "exit $?" >> rg_132300.log ) &
wait
for T in 124926 125954 131124 132300; do echo "== $T old: $(~/xvla-mac/bin/python -c "import pandas as pd;print(pd.read_csv('q_ctpackTF$T.oldgate.csv')['class'].value_counts().to_dict())" 2>/dev/null) new: $(grep -E "^\{'train'" rg_$T.log | tail -1) $(grep -E '^exit' rg_$T.log)"; done
nice -n 10 zsh gather_by_class.sh 20261112_124926 20261112_125954 20261112_131124 20261112_132300
echo RERUN_GATHER_DONE
