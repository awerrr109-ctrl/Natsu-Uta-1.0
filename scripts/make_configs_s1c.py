"""Session-1 E3/E4: iso-PARAMETER language-modeling comparison on TinyStories (bytes).
All models ~0.8M params except E4z (iso-FLOP dense upper bound, ~1.6M)."""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")
base = dict(d_model=128, n_heads=2, n_kv_heads=1, head_dim=64, chunk=32, n_prelude=1, n_core=2, n_coda=1)
ts = {"kind": "memmap", "train": "data_cache/ts_train.bin", "valid": "data_cache/ts_valid.bin"}
t = dict(seq=128, batch=16, steps=800, lr=3e-3, warmup=60, eval_every=400, log_every=100, threads=2, eval_batches=12, save=True)
moe = dict(moe_experts=8, moe_topk=2, moe_shared=1, moe_expert_mult=0.296)   # FFN params == dense FFN params (iso-param)
C = {
 "E3a_ts_attn4":          (dict(pattern="a"), {}),
 "E3b_ts_hyb4":           (dict(pattern="ga"), {}),
 "E4a_ts_hyb_loop3":      (dict(pattern="ga", n_loops=3, reinject=True), {"loop_sampling": "uniform", "eval_loops": [2, 4, 6]}),
 "E4b_ts_hyb_loop3_moe":  (dict(pattern="ga", n_loops=3, reinject=True, **moe), {"loop_sampling": "uniform", "eval_loops": [2, 4, 6]}),
 "E4c_ts_hyb_moe_noloop": (dict(pattern="ga", **moe), {}),
 "E4d_ts_natsu_toy":      (dict(pattern="ga", n_loops=3, reinject=True, depth_gate=True, mtp=1, **moe),
                           {"loop_sampling": "uniform", "self_distill": 0.5, "gate_penalty": 0.01, "eval_loops": [2, 4]}),
 "E4z_ts_hyb8_isoflop":   (dict(pattern="ga", n_prelude=1, n_core=6, n_coda=1), {}),
}
for n, (m, tr) in C.items():
    json.dump({"name": n, "model": {**base, **m}, "train": {**t, **tr}, "data": ts, "extra_eval": {}}, open(f"{OUT}/{n}.json", "w"), indent=1)
print(list(C))
