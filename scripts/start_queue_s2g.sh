#!/bin/bash
# after s2f: run the digit_pos cache test first; only if it passes, run the Abacus 2-cell probe
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2f.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[def][.]sh" > /dev/null; do sleep 30; done
cd tests && MALLOC_ARENA_MAX=1 python3 test_model.py > ../experiments/test_s2g.log 2>&1; ok=$?; cd ..
grep -q "ok digit_pos" experiments/test_s2g.log || { echo "TEST FAILED rc=$ok" >> experiments/test_s2g.log; exit 1; }
C=experiments/configs
setsid nohup scripts/run_queue.sh $C/E8g_add_moe_abacus.json $C/E8h_add_loop3fixed_moe_abacus.json > experiments/queue_s2g.log 2>&1 < /dev/null &
