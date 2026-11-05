#!/bin/bash
# RunPod volume -> Mac: the collected outputs of one job (pod_convert_take.sh writes them to $ROOT/jobs/<PACK>/results, no frames).
#   RP_HOST=root@<ip> RP_PORT=<port> ./fetch_results.sh ctpackFL1102X            # download only -> ./results/<PACK>/
#   RP_HOST=... RP_PORT=... INSTALL=1 ./fetch_results.sh ctpackFL1102X             # + install into the Mac tree (see below)
# results/<PACK>/ layout:
#   home/<path relative to ~>   umi_traj_*.npz + segments.json + wrist*/umi_jaw.npz of each take, data/tag_rig_derived/<take>,
#                               data/<DC>/<take> (Cg derived + ik_v1), v2b/start_seed_<take>.json, v2b/out/trackv31speed_<take>.npz,
#                               _fgo_1023/tagrot/<take>.npz, stitch/qa/fus_{R,Cg}_g/<take>_*.npz
#   idle_trim_rows.csv          the take's idle-trim rows (tag_rig_unsplit.py reads reports/idle_trim.csv -> must be merged)
#   pack/<PACK>/<take>_dNN/     meta.json, tracks.npz, solve2/solve3 npz (CoTracker frames are not fetched)
#   q_<PACK>.csv, keys_<tag>_gs.txt, logs/, job.env, demo_specs.txt
# INSTALL=1: rsync -rlt results/<PACK>/home/ -> ~/ with --ignore-existing (never overwrites a Mac file; skipped files are listed);
#            OVERWRITE=1 instead overwrites and keeps the replaced Mac files under results/<PACK>/mac_backup_<stamp>/;
#            idle_trim rows are APPENDED to ~/laundry_dit_training/reports/idle_trim.csv only for takes not already in it (backup first).
set -euo pipefail
PACK=${1:?usage: fetch_results.sh <PACK>}
: "${RP_HOST:?set RP_HOST=root@<ip>}"; : "${RP_PORT:?set RP_PORT}"
RP_KEY=${RP_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}; ROOT=${ROOT:-/workspace/rp_convert}
HERE=$(cd "$(dirname "$0")" && pwd); DST=$HERE/results/$PACK; mkdir -p "$DST"
RSYNC=/opt/homebrew/bin/rsync; [ -x $RSYNC ] || RSYNC=rsync
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -i $RP_KEY -p $RP_PORT"
$SSH "$RP_HOST" "grep -q _CG_DONE $ROOT/jobs/$PACK/logs/chain.log" || { echo "job $PACK not finished (no _CG_DONE in chain.log)"; $SSH "$RP_HOST" "tail -5 $ROOT/jobs/$PACK/logs/chain.log" || true; exit 1; }
for r in 1 2 3; do $RSYNC -rlt --info=progress2 -e "$SSH" "$RP_HOST:$ROOT/jobs/$PACK/results/" "$DST/" && break; echo "retry $r"; sleep 20; done
echo "fetched -> $DST ($(du -sh "$DST" | cut -f1))"; ls "$DST"
[ "${INSTALL:-0}" = 1 ] || { echo "(download only; INSTALL=1 to install into ~)"; exit 0; }

if [ "${OVERWRITE:-0}" = 1 ]; then
  BK=$DST/mac_backup_$(date +%Y%m%d_%H%M%S); $RSYNC -rlt --itemize-changes --backup --backup-dir="$BK" "$DST/home/" "$HOME/" | grep -v '^\.' || true
  echo "installed (overwrite); replaced files backed up under $BK"
else
  $RSYNC -rlt --ignore-existing --itemize-changes "$DST/home/" "$HOME/" | grep -v '^\.' || true
  echo "installed (ignore-existing). Files that already existed on the Mac and were NOT replaced:"
  (cd "$DST/home" && find . -type f | while read -r f; do [ -e "$HOME/$f" ] && ! cmp -s "$f" "$HOME/$f" && echo "  differs: ~/${f#./}"; done) || true
fi
CSV=$HOME/laundry_dit_training/reports/idle_trim.csv
python3 - "$DST/idle_trim_rows.csv" "$CSV" <<'EOF'
import csv, shutil, sys, time
new, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(new))); have = {r["take"] for r in csv.DictReader(open(dst))}
add = [r for r in rows if r["take"] not in have]
if not add: print("idle_trim.csv: nothing to add (takes already present:", sorted({r['take'] for r in rows}), ")"); sys.exit()
shutil.copy2(dst, dst + time.strftime(".bak_fetch_%Y%m%d_%H%M%S"))
fields = next(csv.reader(open(dst)))
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); [w.writerow({k: r.get(k, "") for k in fields}) for r in add]
print(f"idle_trim.csv: appended {len(add)} rows for {sorted({r['take'] for r in add})}")
EOF
K=$(ls "$DST"/keys_*_gs.txt 2>/dev/null | head -1); DC=$(grep -o '^DC=.*' "$DST/job.env" | cut -d= -f2)
echo "keys: $K ($(wc -w < "$K") GOLD+SILVER)"
cat <<EOF
Mac-side LeRobot build (same flags as scratchpad/flat_build_upload.py), e.g. per take:
  cd ~/laundry_dit_training && ~/xvla-mac/bin/python datasets/tag_rig_unsplit.py --label fused --speed 0.5 --mask-mode stitched \\
    --ik-fail-mm 30 --render 0 --overwrite --grip-label cont --out data/lerobot/<name> --takes ~/ego_collector/datasets/tag_rig/<take> \\
    --derived data/$DC --demos \$(cat $K) --old-split <split.json>
EOF
