#!/bin/bash
# F015 rerun of TTS curves at 2 and 3 digits (inside the models' working range) + E4j 3.2M kNN row if missing. Runs after s2j.
cd "$(dirname "$0")/.."
while [ ! -f experiments/test_s2j.log ] || pgrep -f "run_queue[.]sh" > /dev/null || pgrep -f "start_queue_s2[a-z][.]sh" > /dev/null || pgrep -f "post_s2[.]sh" > /dev/null; do sleep 30; done
cd src; L=../experiments/tts_s2b.log
for d in 2 3; do for ck in E8a_add_moe_noloop E8b_add_loop3fixed_moe E8e_add_moe_core6 E8g_add_moe_abacus E8c_add_moe_engram E8d_add_loop3fixed_moe_engram; do
  echo "=== tts_eval $ck digits=$d $(date)" >> $L
  [ -f ../checkpoints/$ck.pt ] && MALLOC_ARENA_MAX=1 python3 -m natsu.tts_eval --ckpt ../checkpoints/$ck.pt --n_problems 64 --Ns 1,4,16 --digits $d >> $L 2>&1
done; done
