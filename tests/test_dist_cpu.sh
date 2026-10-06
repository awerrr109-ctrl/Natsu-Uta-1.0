#!/bin/bash
# CPU gloo x2 smoke: train 4 steps, save, resume to 6 steps, with MTP + engram_delay. Asserts exit 0 and data_state.json written.
set -e
cd "$(dirname "$0")/../src"
T=$(mktemp -d)
python3 - <<'PY' "$T"
import numpy as np, sys; np.random.default_rng(0).integers(0, 250, 200000).astype(np.uint16).tofile(sys.argv[1] + "/tok.bin")
PY
OV='{"vocab_size":260,"d_model":64,"n_heads":2,"n_kv_heads":1,"head_dim":32,"n_prelude":1,"n_core":1,"n_coda":1,"pattern":"ga","chunk":16,"moe_experts":4,"moe_topk":2,"moe_shared":1,"engram_slots":64,"engram_heads":1,"engram_dim":16,"mtp":1}'
CFG=../configs/ladder/C4_toy.json
MALLOC_ARENA_MAX=1 torchrun --nproc_per_node 2 -m natsu.train_dist --config $CFG --override "$OV" --data $T/tok.bin --cpu --steps 4 --seq 64 --batch 2 --save_every 4 --out $T/run --engram_delay 0.2,0.6
test -f $T/run/step_4/data_state.json
MALLOC_ARENA_MAX=1 torchrun --nproc_per_node 2 -m natsu.train_dist --config $CFG --override "$OV" --data $T/tok.bin --cpu --steps 6 --seq 64 --batch 2 --save_every 2 --out $T/run --resume $T/run/step_4 --engram_delay 0.2,0.6
test -f $T/run/step_6/model.pt
echo "ok dist cpu"
