#!/bin/bash
# F017 recovery: 10M stage with batch 2 x accum 4 after a 3-step memory probe; + ICL arm for v1_lr20. After s2j.
cd "$(dirname "$0")/.."
while [ ! -f experiments/queue_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[defghijlmnopqrs][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
C=experiments/configs
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_model as t; t.test_moe_sparse_equals_dense(); print('ok moe sparse')" ) > experiments/test_s2t.log 2>&1
grep -q "ok moe sparse" experiments/test_s2t.log || exit 1
( cd src && MALLOC_ARENA_MAX=1 timeout 900 python3 -m natsu.train --config ../$C/L10a_C4_10M.json --set train.steps=3 train.save=false train.eval_every=1000 name=probe_l10a ) > experiments/probe_s2t.log 2>&1
grep -q FINAL experiments/probe_s2t.log || { echo "PROBE FAILED" >> experiments/probe_s2t.log; exit 1; }
rm -rf experiments/results/probe_l10a*
scripts/run_queue.sh $C/E9d_facts_edit_engram_replay.json $C/E9e_facts_edit_full_replay.json $C/E9f_facts_edit_engram_rowmask.json > experiments/queue_s2t_e9.log 2>&1 < /dev/null
( cd src && for ck in E9d_facts_edit_engram_replay E9e_facts_edit_full_replay E9f_facts_edit_engram_rowmask; do
    [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.lm_eval --ckpt checkpoints/$ck.pt >> ../experiments/e9_collateral.jsonl 2>> ../experiments/e9_collateral.err; done )
scripts/run_queue.sh $C/E6Lj_v1_lr20_s0.json $C/E6Lj_v1_lr20_s1.json > experiments/queue_s2t_icl.log 2>&1 < /dev/null
python3 scripts/transition_table.py 0.2 E6Lj > experiments/e6lj.txt
if ! grep -q "transitioned 2/2" experiments/e6lj.txt; then   # F017 rule: v1_lr20 needs ICL 2/2, else revert 10M Engram configs to lr x5
  python3 - <<'PY'
import json
for n in ["L10a_C4_10M","L10d_dense_10M_engram"]:
    p=f"experiments/configs/{n}.json"; c=json.load(open(p)); c["train"]["engram_lr_mult"]=5.0; c["note"]+=" | reverted to lr x5 (E6Lj < 2/2)"; json.dump(c,open(p,"w"),indent=1)
PY
fi
scripts/run_queue.sh $C/L10a_C4_10M.json $C/L10b_C0like_10M_noEngram.json $C/L10c_dense_10M_noEngram_noMoE.json $C/L10d_dense_10M_engram.json > experiments/queue_s2t.log 2>&1 < /dev/null
# G1c (killed in post_s2b by OOM at qbatch 4096 x chunk 10000): resume with a memory-safe batch, then ctrl/knn runs
K=data_cache/knn_E4j_moe_engram_noloop_800000
[ -d $K ] || ( cd src && MALLOC_ARENA_MAX=1 python3 -m natsu.knnlm build --ckpt checkpoints/E4j_moe_engram_noloop.pt --data data_cache/ts_train.bin --tokens 800000 --out $K >> ../experiments/knnlm_s2b.jsonl 2>&1 )
rm -rf data_cache/knn_tg_E4j_0.8M
( cd src && MALLOC_ARENA_MAX=1 timeout 21600 python3 -m natsu.knnlm targets --ckpt checkpoints/E4j_moe_engram_noloop.pt --store $K --start 3200000 --tokens 1638400 --seq 128 \
    --qbatch 1024 --chunk 20000 --out data_cache/knn_tg_E4j_0.8M > ../experiments/knn_targets_s2t.log 2>&1 )
[ -f data_cache/knn_tg_E4j_0.8M/meta.json ] && scripts/run_queue.sh $C/G1c_ctrl_E4j_region.json $C/G1c_knn_E4j_region.json > experiments/queue_s2t_g1c.log 2>&1 < /dev/null
