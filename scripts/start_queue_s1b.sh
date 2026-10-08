#!/bin/bash
cd "$(dirname "$0")/.."
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E1v2a_dense4.json $C/E1v2c_loop2x3.json $C/E2a_mqar_attn.json $C/E2b_mqar_gdn.json \
  $C/E2c_mqar_hybrid_gga.json $C/E1v2b_dense8.json $C/E1v2d_loop2x4_rand.json $C/E3a_ts_attn4.json $C/E3b_ts_hyb4.json \
  $C/E3c_ts_hyb_loop3_rand_sd.json > experiments/queue_s1b.log 2>&1 < /dev/null &
