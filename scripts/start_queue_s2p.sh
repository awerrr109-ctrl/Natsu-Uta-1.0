#!/bin/bash
# Runs after s2o, before s2i (s2i waits on this). Gate: digit-mask cache test. E6Li x2 (N7 norm+dilation), E8k/E8l (digit mask).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2o.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghlmno][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_engram_skip_digits_cache(); print('ok skip digits')" ) > experiments/test_s2p.log 2>&1
grep -q "ok skip digits" experiments/test_s2p.log || exit 1
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E8k_add_moe_engram_skipdigits.json $C/E8l_add_moe_engram_lr5_skipdigits.json > experiments/queue_s2p.log 2>&1 < /dev/null &
sleep 5
