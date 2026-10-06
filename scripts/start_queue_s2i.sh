#!/bin/bash
# 10M ladder stage. Waits for s2h, sizes L10b iso-param to L10a inside the queue, then runs both.
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2l.log ] && [ ! -f experiments/test_s2l.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghl][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
{ MALLOC_ARENA_MAX=1 python3 scripts/iso_param_match.py $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json moe_expert_mult &&
  MALLOC_ARENA_MAX=1 python3 scripts/iso_param_match.py $C/L10a_C4_10M.json $C/L10c_dense_10M_noEngram_noMoE.json ffn_mult &&
  MALLOC_ARENA_MAX=1 python3 scripts/iso_param_match.py $C/L10a_C4_10M.json $C/L10d_dense_10M_engram.json ffn_mult; } > experiments/sizing_s2i.log 2>&1 || exit 1
setsid nohup scripts/run_queue.sh $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json $C/L10c_dense_10M_noEngram_noMoE.json $C/L10d_dense_10M_engram.json > experiments/queue_s2i.log 2>&1 < /dev/null &
