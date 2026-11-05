#!/bin/zsh
# 11-04 user: "아니 에폭마다 찍어줘. 우리가 발하고 테스트를 어떤 손해를 보고 만들었는데" -> every epoch checkpoint of the 2x4090 FLAT250 run is
# pulled from mine/ckpt (gateway rsync, local lock) and evaluated offline on the Mac (MPS) on val (39) AND test (38):
# evaluation/tagrig_offline_eval.py --stride 16 (Mac speed ~0.18 s / anchor; stride 4 would take ~2 h per epoch > 69 min epoch), val loss 20 batches.
# 11-05: generic copy for FLAT-PGRASP (env RUN, TAG, DSP = local val/test prefix, EPOCH_UPD, PGRASP) -> reports/$TAG, checkpoints/$TAG.
# Results: reports/FLAT250/epoch_evals.jsonl (+ per-epoch offline_eval.json). Local ckpts: keep best-val + newest only (all epochs stay on mine/).
cd ~/laundry_dit_training
RUN=${RUN:?RUN}; TAG=${TAG:?TAG}; DSP=${DSP:?DSP}; EPOCH_UPD=${EPOCH_UPD:?EPOCH_UPD}; R=mine/ckpt/ldt/$RUN/checkpoints; C=checkpoints/$TAG; O=reports/$TAG
PY=~/bh_indy7_LeRobot/.venv/bin/python; mkdir -p $C $O; touch $O/done_steps.txt
echo "trained with GRIP_STATE_MASK=1 + PGRASP=$PGRASP (state 18,19 = p_grasp)" > $C/GRIP_STATE_MASK
export GRIP_STATE_MASK=1 AUG=0 PGRASP
lock() { until mkdir /tmp/johann_gw_rsync.lock 2>/dev/null; do sleep 10; done; }; unlock() { rmdir /tmp/johann_gw_rsync.lock; }
typeset -A seen
while true; do
  lock; L=$(/opt/homebrew/bin/rsync -r --list-only --include '*/' --include 'model.safetensors' --exclude '*' johann@100.64.0.1:$R/ 2>/dev/null); unlock
  for st in $(echo "$L" | grep -oE "[0-9]{6}/pretrained_model/model.safetensors" | cut -c1-6 | sort -u); do
    grep -qx $st $O/done_steps.txt && continue
    sz=$(echo "$L" | grep "$st/pretrained_model/model.safetensors" | awk '{print $2}')
    [ "${seen[$st]}" != "$sz" ] && { seen[$st]=$sz; continue; }          # size stable over two polls -> fully written
    ep=$(( 10#$st / EPOCH_UPD ))
    lock; /opt/homebrew/bin/rsync -a --mkpath johann@100.64.0.1:$R/$st/pretrained_model/ $C/$st/pretrained_model/; rc=$?; unlock
    [ $rc -ne 0 ] && continue
    for sp in val test; do
      nice -n 10 $PY evaluation/tagrig_offline_eval.py --ckpt $C/$st --data data/lerobot/${DSP}_$sp --stride 16 --video-backend pyav --out $O/$st/$sp > $O/$st.$sp.log 2>&1
    done
    TAG=$TAG $PY - $st $ep <<'PY'
import json, pathlib, sys, shutil
import os; st, ep = sys.argv[1], int(sys.argv[2]); O = pathlib.Path("reports") / os.environ["TAG"]; C = pathlib.Path("checkpoints") / os.environ["TAG"]
rec = dict(epoch=ep, step=int(st))
for sp in ("val", "test"):
    f = O / st / sp / "offline_eval.json"
    if not f.exists(): rec[f"{sp}_error"] = "eval failed (see log)"; continue
    r = json.loads(f.read_text()); rec[f"{sp}_score_mm"] = r.get("val_score_mm"); rec[f"{sp}_loss"] = r.get("val_loss")
    rec[f"{sp}_k32_mm"] = {k: round(v["mean"], 1) if isinstance(v, dict) else v for k, v in r.get("deviation", {}).items() if k.endswith("mm_k32") and "zero" not in k}
    rec[f"{sp}_grip"] = r.get("gripper")
with open(O / "epoch_evals.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
rows = [json.loads(l) for l in open(O / "epoch_evals.jsonl")]
best = min((x for x in rows if x.get("val_score_mm") is not None), key=lambda x: x["val_score_mm"], default=None)
if best: (O / "BEST_VAL_STEP").write_text(f"{best['step']:06d} epoch {best['epoch']} val {best['val_score_mm']}\n")
for d in C.glob("[0-9]*"):                                    # keep best-val + newest locally
    if best and d.name not in (f"{best['step']:06d}", st): shutil.rmtree(d, ignore_errors=True)
print(json.dumps({k: rec.get(k) for k in ("epoch", "step", "val_score_mm", "val_loss", "test_score_mm", "test_loss")}), "best", best and best["epoch"], flush=True)
PY
    echo $st >> $O/done_steps.txt
  done
  sleep 300
done
