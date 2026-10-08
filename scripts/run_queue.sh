#!/bin/bash
# Sequential experiment queue (1GB box: never run two trainings concurrently).
# Usage: scripts/run_queue.sh experiments/configs/A.json experiments/configs/B.json ...
export MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0
ROOT=$(realpath "$(dirname "$0")/..")
RES="$ROOT/experiments/results"
mkdir -p "$RES"
for cfg in "$@"; do
  cfg=$(realpath "$cfg")
  n=$(basename "$cfg" .json)
  if [ -f "$RES/$n/final.json" ]; then echo "skip $n"; continue; fi
  echo "=== $n $(date)"
  (cd "$ROOT/src" && python3 -m natsu.train --config "$cfg" > "$RES/$n.stdout" 2>&1) || echo "FAILED $n"
  tail -n 1 "$RES/$n.stdout" | cut -c1-400
done
echo QUEUE_DONE
