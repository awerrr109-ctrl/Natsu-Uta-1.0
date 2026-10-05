#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "natsu.train" > /dev/null; do sleep 15; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E4j_moe_engram_noloop.json $C/E4g_loop3_moe_fixedR_noinj.json $C/E4h_loop3_moe_unif_idinj.json $C/E4i_loop3_moe_fixedR_idinj.json \
  $C/E4f_ts_loop3_moe_engram.json $C/E4e_ts_hyb_loop3_moe_sharedkv.json $C/E2a_mqar_attn.json $C/E2b_mqar_gdn.json $C/E2c_mqar_hybrid_gga.json \
  > experiments/queue_s1e.log 2>&1 < /dev/null &
