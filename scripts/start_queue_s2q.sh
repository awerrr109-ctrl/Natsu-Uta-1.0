#!/bin/bash
# table-lr sweep x10/x20/x50 after s2p, before s2i (s2i waits on this).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2p.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghlmnop][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4pu_engramv1_lr10.json $C/E4pu_engramv1_lr20.json $C/E4pu_engramv1_lr50.json > experiments/queue_s2q.log 2>&1 < /dev/null &
sleep 5
