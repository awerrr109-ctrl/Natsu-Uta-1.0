#!/bin/bash
# table-lr: replicate x20 (2 more seeds), extend to x100; after s2q, before s2i (s2i waits on this).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2q.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghlmnopq][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4pu_engramv1_lr20_s1.json $C/E4pu_engramv1_lr20_s2.json $C/E4pu_engramv1_lr100.json > experiments/queue_s2r.log 2>&1 < /dev/null &
sleep 5
