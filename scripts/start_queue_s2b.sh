#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_queue[.]sh" > /dev/null; do sleep 20; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E8a_add_moe_noloop.json $C/E8b_add_loop3fixed_moe.json > experiments/queue_s2b.log 2>&1 < /dev/null &
