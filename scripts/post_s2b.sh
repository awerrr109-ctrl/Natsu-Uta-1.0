#!/bin/bash
# F015 rerun of TTS curves at 2 and 3 digits (inside the models' working range) + E4j 3.2M kNN row if missing. Runs after s2j.
cd "$(dirname "$0")/.."
while [ ! -f experiments/test_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[a-z][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
cd src; L=../experiments/tts_s2b.log
for d in 2 3; do for ck in E8a_add_moe_noloop E8b_add_loop3fixed_moe E8e_add_moe_core6 E8g_add_moe_abacus E8c_add_moe_engram E8d_add_loop3fixed_moe_engram; do
  echo "=== tts_eval $ck digits=$d $(date)" >> $L
  [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.tts_eval --ckpt ../checkpoints/$ck.pt --n_problems 64 --Ns 1,4,16 --digits $d >> $L 2>&1
done; done
# kNN-LM: extend lambda grid past 0.5 (the curve had not turned at 0.5) on both stores, E4c and E4j
cd ..
for ck in E4c_ts_hyb_moe_noloop E4j_moe_engram_noloop; do for N in 800000 3200000; do
  [ -d data_cache/knn_${ck}_$N ] && (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm eval --ckpt checkpoints/$ck.pt --store data_cache/knn_${ck}_$N --lams 0.5,0.65,0.8 >> ../experiments/knnlm_s2b.jsonl 2>> ../experiments/knnlm_s2b.err)
done; done
# G1b datastore cost: JL projection + int8 keys (gate: unit test). bytes/token 258 (fp16, D=128) -> 68 / 36 / 20
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_harness as t; t.test_knn_compress_preserves_neighbours(); print('ok knn compress')" ) > experiments/test_s2b.log 2>&1
if grep -q "ok knn compress" experiments/test_s2b.log; then
  for ck in E4c_ts_hyb_moe_noloop E4j_moe_engram_noloop; do S=data_cache/knn_${ck}_3200000; [ -d $S ] || continue
    for r in 64 32 16; do
      (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm compress --store $S --out ${S}_jl$r >> ../experiments/knnlm_s2b.jsonl 2>> ../experiments/knnlm_s2b.err &&
       MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm eval --ckpt checkpoints/$ck.pt --store ${S}_jl$r --lams 0.25,0.5,0.65 >> ../experiments/knnlm_s2b.jsonl 2>> ../experiments/knnlm_s2b.err)
    done
  done
fi
