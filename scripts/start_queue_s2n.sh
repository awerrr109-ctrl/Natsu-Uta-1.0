#!/bin/bash
# F013 ablation on LM (path vs lr x5) after E4pp regressed (+0.0074). Runs after s2l, before s2i (s2i waits on this script).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2l.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghlm][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4pq_moe_engrampaper_lr1.json $C/E4pr_moe_engramv1_lr5.json > experiments/queue_s2n.log 2>&1 < /dev/null &
sleep 5
