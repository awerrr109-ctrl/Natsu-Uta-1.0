#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "natsu.train" > /dev/null; do sleep 15; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/B1_bpe_moe_noloop.json $C/B2_bpe_moe_engram.json $C/B3_bpe_loop3fixed_moe.json \
 $C/E4k_loop3fixed_moe_engram.json $C/E4m_loop3_moe_fixed_then_unif.json $C/E4n_loop3_moe_poissonR.json $C/E4o_loop3_moe_fixed_lti.json $C/E4p_loop3_moe_lookahead_gate.json $C/E4c_ts_hyb_moe_noloop_s1.json $C/E4j_moe_engram_noloop_s1.json $C/E4g_loop3_moe_fixedR_noinj_s1.json \
 $C/E4i_loop3_moe_fixedR_idinj.json $C/E4l_loop3fixed_moe_selfdistill.json $C/E4e2_loop3fixed_moe_sharedkv.json \
 $C/E4c_ts_hyb_moe_noloop_s2.json $C/E4j_moe_engram_noloop_s2.json $C/E4g_loop3_moe_fixedR_noinj_s2.json \
 > experiments/queue_s2a.log 2>&1 < /dev/null &
