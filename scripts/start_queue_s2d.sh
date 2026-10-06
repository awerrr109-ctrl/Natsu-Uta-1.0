#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_queue[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4q_loop3_moe_engram_lookahead_memgate.json $C/E4r_loop3_moe_engram_lookahead_nomemgate.json > experiments/queue_s2d.log 2>&1 < /dev/null &
