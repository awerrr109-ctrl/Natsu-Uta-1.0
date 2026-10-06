#!/bin/bash
# F011 rerun: test gate (vip key order) + E4s2
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghij][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_vip_key_order_matches_builder(); print('ok vip key order')" ) > experiments/test_s2k.log 2>&1
grep -q "ok vip key order" experiments/test_s2k.log || exit 1
setsid nohup scripts/run_queue.sh experiments/configs/E4s2_moe_engramv2_vip_fixedkeys.json > experiments/queue_s2k.log 2>&1 < /dev/null &
