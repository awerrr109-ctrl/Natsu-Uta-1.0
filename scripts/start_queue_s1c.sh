#!/bin/bash
# waits for any running queue to finish, then runs LM comparisons first, then remaining synthetic probes
cd "$(dirname "$0")/.."
while pgrep -f run_queue.sh > /dev/null; do sleep 10; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E3b_ts_hyb4.json $C/E4a_ts_hyb_loop3.json $C/E4b_ts_hyb_loop3_moe.json \
  $C/E4c_ts_hyb_moe_noloop.json $C/E4d_ts_natsu_toy.json $C/E3a_ts_attn4.json $C/E4z_ts_hyb8_isoflop.json \
  $C/E1v2c_loop2x3.json $C/E2a_mqar_attn.json $C/E2b_mqar_gdn.json $C/E2c_mqar_hybrid_gga.json \
  > experiments/queue_s1c.log 2>&1 < /dev/null &
