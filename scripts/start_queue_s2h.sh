#!/bin/bash
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2g.log ] && [ ! -f experiments/test_s2g.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defg][.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E5a_distill_E4z_to_moe_s1.json $C/E6La_varchain_moe_noloop_s0.json $C/E6Lb_varchain_moe_engram_s0.json $C/E6La_varchain_moe_noloop_s1.json $C/E6Lb_varchain_moe_engram_s1.json $C/E6Lc_varchain_moe_engram_delayed_s0.json $C/E6Lc_varchain_moe_engram_delayed_s1.json $C/E4u_moe_engram_delayed.json > experiments/queue_s2h.log 2>&1 < /dev/null &
