#!/bin/bash
# Runs memory-heavy evals only after every queue/launcher has exited (F006 rule).
cd "$(dirname "$0")/.."
idle() { ! pgrep -f "run_queue[.]sh" >/dev/null && ! pgrep -f "start_queue_s2[a-h][.]sh" >/dev/null; }
until idle; do sleep 60; done; sleep 60; until idle; do sleep 60; done
export MALLOC_ARENA_MAX=1
cd src
L=../experiments/post_s2.log
for ck in E4q_loop3_moe_engram_lookahead_memgate E4r_loop3_moe_engram_lookahead_nomemgate; do
  echo "=== gate_eval $ck $(date)" >> $L
  [ -f ../checkpoints/$ck.pt ] && python3 -m natsu.gate_eval --ckpt ../checkpoints/$ck.pt >> $L 2>&1
done
for ck in E8a_add_moe_noloop E8b_add_loop3fixed_moe E8c_add_moe_engram E8d_add_loop3fixed_moe_engram; do
  echo "=== tts_eval $ck $(date)" >> $L
  [ -f ../checkpoints/$ck.pt ] && python3 -m natsu.tts_eval --ckpt ../checkpoints/$ck.pt --n_problems 64 --Ns 1,4,16 >> $L 2>&1
done
echo "=== DONE $(date)" >> $L
