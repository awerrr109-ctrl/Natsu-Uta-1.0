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
for ck in E4c_ts_hyb_moe_noloop E4j_moe_engram_noloop E4z_ts_hyb8_isoflop E4g_loop3_moe_fixedR_noinj E4k_loop3fixed_moe_engram E5c_bornagain_E4c_to_moe E4u_moe_engram_delayed E5a_distill_E4z_to_moe_s1 E4p_loop3_moe_lookahead_gate E4e2_loop3fixed_moe_sharedkv E3a_ts_attn4 E3b_ts_hyb4; do
  [ -f ../checkpoints/$ck.pt ] && python3 -m natsu.ds_eval --ckpt checkpoints/$ck.pt --n 200 >> ../experiments/ds_eval_s2.jsonl 2>> ../experiments/ds_eval_s2.err
done
for ck in E4g_loop3_moe_fixedR_noinj E4p_loop3_moe_lookahead_gate E4k_loop3fixed_moe_engram E8b_add_loop3fixed_moe; do
  [ -f ../checkpoints/$ck.pt ] && python3 -m natsu.route_diag --ckpt checkpoints/$ck.pt >> ../experiments/route_diag_s2.jsonl 2>> ../experiments/route_diag_s2.err
done
cd .. && nice -n 19 python3 research/analyze.py >> experiments/post_s2.log 2>&1; cd src
echo "=== DONE $(date)" >> $L
