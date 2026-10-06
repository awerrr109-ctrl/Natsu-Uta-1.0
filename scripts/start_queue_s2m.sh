#!/bin/bash
# F013 paper-faithful Engram check. Runs right after post_s2 and BEFORE s2l (s2l waits on this script), because a positive
# answer changes the Engram variant used by every later Engram run (E4v-x, E6Ld, 10M).
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2h.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defgh][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_engram_paper_cache(); print('ok engram paper'); t.test_engram_cache(); print('ok engram')" ) > experiments/test_s2m.log 2>&1
grep -q "ok engram$" experiments/test_s2m.log || exit 1
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E6Le_varchain_moe_engrampaper_s0.json $C/E6La_varchain_moe_noloop_s2.json $C/E6Lb_varchain_moe_engram_s2.json $C/E4pp_moe_engrampaper_noloop.json > experiments/queue_s2m.log 2>&1 < /dev/null &
sleep 5
