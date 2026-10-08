#!/bin/bash
# F015 rerun of TTS curves at 2 and 3 digits (inside the models' working range) + E4j 3.2M kNN row if missing. Runs after s2j.
cd "$(dirname "$0")/.."
while [ ! -f experiments/test_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[a-z][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
cd src; L=../experiments/tts_s2b.log
for d in 2 3; do for ck in E8a_add_moe_noloop E8b_add_loop3fixed_moe E8e_add_moe_core6 E8g_add_moe_abacus E8c_add_moe_engram E8d_add_loop3fixed_moe_engram; do
  echo "=== tts_eval $ck digits=$d $(date)" >> $L
  [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.tts_eval --ckpt ../checkpoints/$ck.pt --n_problems 64 --Ns 1,4,16 --digits $d >> $L 2>&1
done; done
# kNN stores were removed by post_s2's cleanup -> rebuild what this stage needs (E4c/E4j 3.2M for lambda grid + JL ablation; E4j 0.8M for G1c)
cd ..
K=experiments/knnlm_s2b.jsonl; E=experiments/knnlm_s2b.err
kb() { [ -d data_cache/knn_$1_$2 ] || (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm build --ckpt checkpoints/$1.pt --data data_cache/ts_train.bin --tokens $2 --out data_cache/knn_$1_$2 >> ../$K 2>> ../$E); }
for ck in E4c_ts_hyb_moe_noloop E4j_moe_engram_noloop; do
  kb $ck 3200000 && (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm eval --ckpt checkpoints/$ck.pt --store data_cache/knn_${ck}_3200000 --lams 0.5,0.65,0.8 >> ../$K 2>> ../$E)
done
# G1b datastore cost: JL projection + int8 keys (gate: unit test). bytes/token 258 (fp16, D=128) -> 68 / 36 / 20
( cd tests && MALLOC_ARENA_MAX=1 timeout 600 python3 -c "import test_harness as t; t.test_knn_compress_preserves_neighbours(); print('ok knn compress'); t.test_region_loader_alignment(); print('ok region')" ) > experiments/test_s2b.log 2>&1
if grep -q "ok knn compress" experiments/test_s2b.log; then
  for ck in E4c_ts_hyb_moe_noloop E4j_moe_engram_noloop; do S=data_cache/knn_${ck}_3200000; [ -d $S ] || continue
    for r in 64 32 16; do
      (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm compress --store $S --dim $r --out ${S}_jl$r >> ../$K 2>> ../$E &&
       MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm eval --ckpt checkpoints/$ck.pt --store ${S}_jl$r --lams 0.25,0.5,0.65 >> ../$K 2>> ../$E)
      rm -rf ${S}_jl$r
    done
    rm -rf $S
  done
fi
# G1c retrieval-distilled Engram: cached p_kNN targets for train[3.2M, 3.2M+1.64M) from E4j's 0.8M store (disjoint), seq=128 = training seq
if grep -q "ok region" experiments/test_s2b.log; then
  kb E4j_moe_engram_noloop 800000
  [ -f data_cache/knn_tg_E4j_0.8M/meta.json ] || (cd src && MALLOC_ARENA_MAX=1 timeout 14400 python3 -m natsu.knnlm targets --ckpt checkpoints/E4j_moe_engram_noloop.pt \
      --store data_cache/knn_E4j_moe_engram_noloop_800000 --start 3200000 --tokens 1638400 --seq 128 --out data_cache/knn_tg_E4j_0.8M > ../experiments/knn_targets_s2b.log 2>&1)
  if [ -f data_cache/knn_tg_E4j_0.8M/meta.json ]; then
    C=experiments/configs
    scripts/run_queue.sh $C/G1c_ctrl_E4j_region.json $C/G1c_knn_E4j_region.json > experiments/queue_s2b_g1c.log 2>&1
  fi
fi
# INFERENCE_SPEC v0.3: expert-cache LRU replay, looped vs non-looped MoE (does loop routing collapse cut flash I/O per token?)
for ck in E4g_loop3_moe_fixedR_noinj E4l_loop3fixed_moe_selfdistill E4c_ts_hyb_moe_noloop E4y_loop3_moe_looprouter E4y3_loop3_moe_looplora8; do
  [ -f checkpoints/$ck.pt ] && (cd src && MALLOC_ARENA_MAX=1 timeout 900 python3 -m natsu.route_diag --ckpt checkpoints/$ck.pt --cache 2,4,6,8,12,16 >> ../experiments/route_cache_s2b.jsonl 2>> ../experiments/route_cache_s2b.err)
done
echo "=== post_s2b DONE $(date)" >> experiments/tts_s2b.log
