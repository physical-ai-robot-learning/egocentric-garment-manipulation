#!/bin/zsh
# 11-05 FLAT-PGRASP 0.3x on RunPod 4x RTX 6000 Ada (pod cytf0h9vgsmze6, deleted by the $170 budget killer ~80 epochs): every epoch checkpoint
# (pretrained_model) is pulled to the PortableSSD (the pod's disk goes with the pod) and the NEWEST pulled one is evaluated on the Mac on
# val + test (tagrig_offline_eval --stride 16, PGRASP side file) -> reports/FLATPG_ADA/epoch_evals.jsonl. An Ada epoch (~30 min) is shorter
# than a Mac val+test eval, so the eval skips to the newest checkpoint instead of queueing (every 1-2 epochs).
cd ~/laundry_dit_training
POD=root@195.26.233.74; PORT=24135; K=~/.runpod/ssh/runpodctl-ssh-key; SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $K -p $PORT"
RUN=DITFT-H663012-flat1103_1104_1105_S03G_train-UMIRECIPE-PGRASPv2W04-GSM-B1024-E100-ADAx4; R=/workspace/johann/ckpt/ldt/$RUN/checkpoints
C=/Volumes/PortableSSD/ldt_ckpt/FLATPG_ADA; O=reports/FLATPG_ADA; DSP=flat1103_1104_1105_S03G; EPOCH_UPD=523
PY=~/bh_indy7_LeRobot/.venv/bin/python; mkdir -p $O; touch $O/done_steps.txt
until mount | grep -q " on /Volumes/PortableSSD "; do echo "[$(date '+%T')] SSD 없음 -> 대기"; sleep 120; done; mkdir -p $C
echo "trained with GRIP_STATE_MASK=1 + PGRASP=grasp-v2-W0.4 (state 18,19 = p_grasp)" > $C/GRIP_STATE_MASK
export GRIP_STATE_MASK=1 AUG=0 PGRASP=grasp-v2-W0.4
typeset -A seen
pull() {   # pull every complete checkpoint not on the SSD yet
  local L st sz
  L=$(${=SSH} $POD "cd $R 2>/dev/null && for d in [0-9]*; do f=\$d/pretrained_model/model.safetensors; [ -f \$f ] && echo \$d \$(stat -c %s \$f); done" </dev/null 2>/dev/null) || return
  echo "$L" | while read st sz; do
    [ -z "$st" ] && continue; [ -f $C/$st/pretrained_model/.pulled ] && continue
    [ "${seen[$st]}" != "$sz" ] && { seen[$st]=$sz; continue; }          # size stable over two polls -> fully written
    /opt/homebrew/bin/rsync -a --partial --mkpath --no-o --no-g -e "${SSH}" $POD:$R/$st/pretrained_model/ $C/$st/pretrained_model/ </dev/null && touch $C/$st/pretrained_model/.pulled && echo "[$(date '+%T')] pulled $st"
  done
}
while true; do
  # 11-06: the SSD can drop off when the USB hub is moved -> never write the checkpoints to an empty mount point on the internal disk
  if ! mount | grep -q " on /Volumes/PortableSSD "; then echo "[$(date '+%T')] SSD 없음 -> 대기"; sleep 120; continue; fi
  pull
  st=$(ls $C | grep -E '^[0-9]{6}$' | sort | tail -1)
  if [ -n "$st" ] && [ -f $C/$st/pretrained_model/.pulled ] && ! grep -qx $st $O/done_steps.txt; then
    ep=$(( 10#$st / EPOCH_UPD )); echo "[$(date '+%T')] eval $st (epoch $ep)"
    for sp in val test; do
      nice -n 10 $PY evaluation/tagrig_offline_eval.py --ckpt $C/$st --data data/lerobot/${DSP}_$sp --stride 16 --video-backend pyav --out $O/$st/$sp > $O/$st.$sp.log 2>&1
    done
    $PY - $st $ep <<'PY'
import json, pathlib, sys
st, ep = sys.argv[1], int(sys.argv[2]); O = pathlib.Path("reports/FLATPG_ADA")
rec = dict(epoch=ep, step=int(st))
for sp in ("val", "test"):
    f = O / st / sp / "offline_eval.json"
    if not f.exists(): rec[f"{sp}_error"] = "eval failed (see log)"; continue
    r = json.loads(f.read_text()); rec[f"{sp}_score_mm"] = r.get("val_score_mm"); rec[f"{sp}_loss"] = r.get("val_loss")
    rec[f"{sp}_k32_mm"] = {k: round(v["mean"], 1) if isinstance(v, dict) else v for k, v in r.get("deviation", {}).items() if k.endswith("mm_k32") and "zero" not in k}
    rec[f"{sp}_grip"] = r.get("gripper")
with open(O / "epoch_evals.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
print(json.dumps({k: rec.get(k) for k in ("epoch", "step", "val_score_mm", "val_loss", "test_score_mm", "test_loss")}), flush=True)
PY
    echo $st >> $O/done_steps.txt
  fi
  sleep 120
done
