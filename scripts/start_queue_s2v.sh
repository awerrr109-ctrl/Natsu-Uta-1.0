#!/bin/bash
# s2v (after s2u): (1) grad_ckpt+shared_first test and memory probe; (2) F018 JL-64/16 rerun (E4j only, to save time);
# (3) TTS 2/3 digits for the digit-gated Engram E8k/E8l.
cd "$(dirname "$0")/.."
sleep 60
while pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[tu][.]sh" > /dev/null; do sleep 30; done
( cd tests && MALLOC_ARENA_MAX=1 timeout 900 python3 -c "import test_model as t; t.test_grad_ckpt_matches_shared_first_loops(); print('ok gradckpt shared_first'); t.test_shared_kv_cache(); print('ok shared kv')" ) > experiments/test_s2v.log 2>&1
( MALLOC_ARENA_MAX=1 timeout 900 python3 scripts/mem_profile.py experiments/configs/L10a_C4_10M.json batch=4 grad_ckpt=true; MALLOC_ARENA_MAX=1 timeout 900 python3 scripts/mem_profile.py experiments/configs/L10a_C4_10M.json batch=4 grad_ckpt=false ) > experiments/mem_s2v.log 2>&1
K=experiments/knnlm_s2v.jsonl; E=experiments/knnlm_s2v.err; ck=E4j_moe_engram_noloop; S=data_cache/knn_${ck}_3200000
[ -d $S ] || (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm build --ckpt checkpoints/$ck.pt --data data_cache/ts_train.bin --tokens 3200000 --out data_cache/knn_${ck}_3200000 >> ../$K 2>> ../$E)
for r in 64 16; do
  (cd src && MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm compress --store $S --dim $r --out ${S}_jl$r >> ../$K 2>> ../$E &&
   python3 -c "import json,sys; d=[json.loads(l) for l in open('../$K') if 'compressed' in l][-1]; assert d['dim']==$r, d" 2>> ../$E &&
   MALLOC_ARENA_MAX=1 timeout 3600 python3 -m natsu.knnlm eval --ckpt checkpoints/$ck.pt --store ${S}_jl$r --lams 0.5,0.65 >> ../$K 2>> ../$E)
  rm -rf ${S}_jl$r
done
rm -rf $S
cd src; L=../experiments/tts_s2v.log
for d in 2 3; do for c in E8k_add_moe_engram_skipdigits E8l_add_moe_engram_lr5_skipdigits; do
  echo "=== tts_eval $c digits=$d $(date)" >> $L
  MALLOC_ARENA_MAX=1 python3 -m natsu.tts_eval --ckpt ../checkpoints/$c.pt --n_problems 64 --Ns 1,4,16 --digits $d >> $L 2>&1
done; done
echo "=== s2v DONE $(date)" >> $L
