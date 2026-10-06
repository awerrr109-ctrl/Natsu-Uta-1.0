#!/bin/bash
# E9 memory-editing probe: edit runs, then collateral bpb of each edited checkpoint vs the base E4j
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2i.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghi][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
# gate: unit tests touched this session (sharded sampler, facts/init_from via a 2-step smoke train)
( cd tests && MALLOC_ARENA_MAX=1 timeout 900 python3 -c "import test_harness as t; t.test_sharded_tokens_epoch_exact(); print('ok sharded')" ) > experiments/test_s2j.log 2>&1
grep -q "ok sharded" experiments/test_s2j.log || echo "SHARDED TEST FAILED" >> experiments/test_s2j.log
timeout 1200 tests/test_dist_cpu.sh >> experiments/test_s2j.log 2>&1 || echo "DIST TEST FAILED" >> experiments/test_s2j.log
( cd src && MALLOC_ARENA_MAX=1 timeout 900 python3 -m natsu.train --config ../experiments/configs/E9a_facts_edit_engram_only.json --set train.steps=2 train.save=false name=smoke_e9 ) >> experiments/test_s2j.log 2>&1 || { echo "E9 SMOKE FAILED" >> experiments/test_s2j.log; exit 1; }
rm -rf experiments/results/smoke_e9*
C=experiments/configs
scripts/run_queue.sh $C/E9a_facts_edit_engram_only.json $C/E9b_facts_edit_full.json $C/E9c_facts_edit_ffn_only.json > experiments/queue_s2j.log 2>&1 < /dev/null
cd src
for ck in E4j_moe_engram_noloop E9a_facts_edit_engram_only E9b_facts_edit_full E9c_facts_edit_ffn_only; do
  [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.lm_eval --ckpt checkpoints/$ck.pt >> ../experiments/e9_collateral.jsonl 2>> ../experiments/e9_collateral.err
done
