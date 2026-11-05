#!/bin/bash
# upload one demo dir to the RunPod pod (used by submit_runpod.sh via xargs -P)
d=${1%/}; for i in 1 2 3 4 5; do $RS -aL --no-o --no-g --exclude tracks.npz -e "$SSH" "$d" "$HOST:/workspace/ct/packs/$PK/" && exit 0; sleep 10; done; exit 1
