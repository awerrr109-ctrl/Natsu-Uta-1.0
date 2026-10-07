#!/bin/bash
# Engram table-lr follow-up (E4pr replication, paper path lr x2.5, v1+lr5 on varchain). After s2n, before 10M (s2i waits on this script).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2n.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghlmn][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4pr_moe_engramv1_lr5_s1.json $C/E4pr_moe_engramv1_lr5_s2.json $C/E4ps_moe_engrampaper_lr2p5.json $C/E6Lh_v1_lr5_s0.json $C/E6Lh_v1_lr5_s1.json $C/E4pt_moe_engramv1_lr5_fal05.json $C/E6Li_norm_dil_only_s0.json $C/E6Li_norm_dil_only_s1.json > experiments/queue_s2o.log 2>&1 < /dev/null &
sleep 5
