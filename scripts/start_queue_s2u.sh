#!/bin/bash
# L10e (10M, Engram lr x5 control vs auto-picked x20). After s2t.
cd "$(dirname "$0")/.."
sleep 60
while pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2t[.]sh" > /dev/null; do sleep 30; done
scripts/run_queue.sh experiments/configs/L10e_C4_10M_engramlr5.json > experiments/queue_s2u.log 2>&1 < /dev/null
