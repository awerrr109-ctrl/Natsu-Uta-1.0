#!/bin/bash
# waits for s2e to start and finish, then runs refutation controls
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2e.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[de][.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E5c_bornagain_E4c_to_moe.json > experiments/queue_s2f.log 2>&1 < /dev/null &
