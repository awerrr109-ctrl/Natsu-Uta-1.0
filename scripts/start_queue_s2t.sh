#!/bin/bash
# F017 recovery: 10M stage with batch 4 x accum 2 after a 3-step memory probe; + ICL arm for v1_lr20. After s2j.
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghijlmnopqrs][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_moe_sparse_equals_dense(); print('ok moe sparse')" ) > experiments/test_s2t.log 2>&1
grep -q "ok moe sparse" experiments/test_s2t.log || exit 1
for o in "moe_sparse=false" "moe_sparse=true" "moe_sparse=true batch=2" "moe_sparse=true chunk=64"; do
  MALLOC_ARENA_MAX=1 timeout 600 python3 scripts/mem_profile.py $C/L10a_C4_10M.json $o >> experiments/mem_profile_s2t.jsonl 2>&1
done
( cd src && MALLOC_ARENA_MAX=1 timeout 900 python3 -m natsu.train --config ../$C/L10a_C4_10M.json --set train.steps=3 train.save=false train.eval_every=1000 name=probe_l10a ) > experiments/probe_s2t.log 2>&1
grep -q FINAL experiments/probe_s2t.log || { echo "PROBE FAILED" >> experiments/probe_s2t.log; exit 1; }
rm -rf experiments/results/probe_l10a*
scripts/run_queue.sh $C/E9d_facts_edit_engram_replay.json $C/E9e_facts_edit_full_replay.json $C/E9f_facts_edit_engram_rowmask.json > experiments/queue_s2t_e9.log 2>&1 < /dev/null
( cd src && for ck in E9d_facts_edit_engram_replay E9e_facts_edit_full_replay E9f_facts_edit_engram_rowmask; do
    [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.lm_eval --ckpt checkpoints/$ck.pt >> ../experiments/e9_collateral.jsonl 2>> ../experiments/e9_collateral.err; done )
scripts/run_queue.sh $C/E6Lj_v1_lr20_s0.json $C/E6Lj_v1_lr20_s1.json $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json $C/L10c_dense_10M_noEngram_noMoE.json $C/L10d_dense_10M_engram.json > experiments/queue_s2t.log 2>&1 < /dev/null
