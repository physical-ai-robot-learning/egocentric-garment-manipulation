#!/bin/zsh
# 11-11 user: wait for H100 SXM x4 (else H200 SXM x4) in US-CA-2 (volume wlbldpcve2), then create the pod and start
# /workspace/umiflat/job/pod_train.sh in the background on it.  Create is only attempted when the stock table shows the GPU in
# US-CA-2; at most MAX_TRY create attempts, at most MAX_H hours of waiting.  Prints POD_READY <id> / GAVE_UP.
MAX_TRY=${MAX_TRY:-20}; MAX_H=${MAX_H:-12}; T_END=$(( $(date +%s) + MAX_H * 3600 )); n=0
PY=~/xvla-mac/bin/python
stock() { runpodctl gpu list 2>/dev/null | $PY -c "
import json,sys
want=sys.argv[1]
for g in json.load(sys.stdin):
    if g['gpuId']==want:
        print(next((x['stockStatus'] for x in g.get('dataCenterAvailability',[]) if x['dataCenterId']=='US-CA-2'),'none'))" "$1"; }
while [ $(date +%s) -lt $T_END ] && [ $n -lt $MAX_TRY ]; do
  for G in "NVIDIA H100 80GB HBM3" "NVIDIA H200"; do
    s=$(stock "$G"); [ -z "$s" ] || [ "$s" = none ] && continue
    n=$((n + 1)); echo "$(date +%T) $G stock $s in US-CA-2 -> create (try $n)"
    out=$(runpodctl pod create --name umiflat-wsd-x4 --gpu-id "$G" --gpu-count 4 --cloud-type COMMUNITY --data-center-ids US-CA-2 \
      --network-volume-id wlbldpcve2 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --min-cuda-version 12.8 \
      --container-disk-in-gb 60 --public-ip --ports "22/tcp" --wait --wait-timeout 15m 2>&1)
    id=$(print -r -- "$out" | $PY -c "import json,sys
try: print(json.loads(sys.stdin.read()).get('id') or '')
except Exception: print('')")
    if [ -z "$id" ]; then
      echo "  create failed: $(print -r -- "$out" | tr '\n' ' ' | cut -c1-300)"
      id=$(print -r -- "$out" | grep -oE '"id": *"[a-z0-9]+"' | head -1 | grep -oE '[a-z0-9]{8,}')   # kept on wait timeout
      [ -n "$id" ] && { echo "  pod $id kept after a wait timeout -> removed"; runpodctl pod delete $id >/dev/null 2>&1; }
      continue
    fi
    echo "POD_READY $id ($G)"
    SSH=$(runpodctl ssh info $id 2>/dev/null); echo "$SSH" > ~/runpod_stage/pod_ssh_$id.json
    exit 0
  done
  sleep 120
done
echo "GAVE_UP after $n create attempts"
