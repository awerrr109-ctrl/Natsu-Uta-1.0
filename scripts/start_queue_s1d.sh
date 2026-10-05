#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f run_queue.sh > /dev/null; do sleep 15; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4e_ts_hyb_loop3_moe_sharedkv.json $C/E4f_ts_loop3_moe_engram.json > experiments/queue_s1d.log 2>&1 < /dev/null &
