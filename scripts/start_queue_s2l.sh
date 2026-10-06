#!/bin/bash
# H14.7 factorised Engram table + F011 rerun E4s2. Runs after s2h and BEFORE the long 10M stage (s2i waits for s2l), because its answer may change the 10M design.
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2h.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghm][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_engram_factorised_cache(); print('ok engram factorised'); t.test_vip_key_order_matches_builder(); print('ok vip key order'); t.test_loop_router_resscale_cache(); print('ok loop router')" ) > experiments/test_s2l.log 2>&1
grep -q "ok loop router" experiments/test_s2l.log || exit 1
C=experiments/configs
MALLOC_ARENA_MAX=1 python3 scripts/iso_param_match.py $C/E4j_moe_engram_noloop.json $C/E4x_engram_factor_r8_small_moreexperts.json moe_expert_mult >> experiments/sizing_s2l.log 2>&1 || exit 1
setsid nohup scripts/run_queue.sh $C/E6Ld_varchain_moe_engram_fullexperts_s0.json $C/E4v_engram_factor_r8_4xslots.json $C/E4w_engram_factor_r4_8xslots.json $C/E4x_engram_factor_r8_small_moreexperts.json $C/E4s2_moe_engramv2_vip_fixedkeys.json $C/E8i_add_moe_engram_fullexperts.json $C/E4y_loop3_moe_looprouter.json $C/E4y2_loop3_moe_looprouter_resscale_lti.json > experiments/queue_s2l.log 2>&1 < /dev/null &
