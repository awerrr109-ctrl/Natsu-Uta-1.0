#!/bin/bash
# 10M ladder stage. Waits for s2h, sizes L10b iso-param to L10a inside the queue, then runs both.
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2h.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defgh][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
MALLOC_ARENA_MAX=1 python3 scripts/iso_param_match.py $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json > experiments/sizing_s2i.log 2>&1 || exit 1
setsid nohup scripts/run_queue.sh $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json > experiments/queue_s2i.log 2>&1 < /dev/null &
