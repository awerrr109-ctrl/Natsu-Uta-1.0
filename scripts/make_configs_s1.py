"""Session-1 toy experiments (see docs/EXPERIMENT_PLAN.md). Each config tests one hypothesis."""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")
base_m = dict(d_model=128, n_heads=2, n_kv_heads=1, head_dim=64, chunk=32)
def w(name, model, train, data, extra=None):
    json.dump({"name": name, "model": {**base_m, **model}, "train": train, "data": data,
               "extra_eval": extra or {}}, open(f"{OUT}/{name}.json", "w"), indent=1)

# E1: recurrent depth on k-hop pointer chasing (H1.1, H4.1). Train hops 1..4, eval OOD hops 6 and 8.
chain = {"kind": "synthetic", "task": "chain", "kw": {"hops": 4, "distract": 4}}
ood = {"hop6": {"min_hops": 6, "hops": 6}, "hop8": {"min_hops": 8, "hops": 8}}
t1 = dict(seq=96, batch=32, steps=1500, lr=2e-3, warmup=100, eval_every=500, log_every=100, threads=1, eval_batches=4, eval_loops=[2,3,4,6,8], save=False)
w("E1a_dense4_attn", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1), t1, chain, ood)
w("E1b_dense8_attn", dict(pattern="a", n_prelude=2, n_core=4, n_coda=2), t1, chain, ood)
w("E1c_loop2x3_attn", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1, n_loops=3), {**t1, "loop_sampling":"fixed"}, chain, ood)
w("E1d_loop2x3_attn_rand", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1, n_loops=4, reinject=True), {**t1, "loop_sampling":"uniform"}, chain, ood)
# E2: token mixer recall on MQAR (H3.1/H3.2). Same params roughly.
mq = {"kind": "synthetic", "task": "mqar", "kw": {"n_pairs": 24, "n_q": 8}}
t2 = dict(seq=192, batch=32, steps=1200, lr=2e-3, warmup=100, eval_every=400, log_every=100, threads=1, eval_batches=4, save=False)
w("E2a_mqar_attn", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1), t2, mq)
w("E2b_mqar_gdn", dict(pattern="g", n_prelude=1, n_core=2, n_coda=1), t2, mq)
w("E2c_mqar_hybrid_gga", dict(pattern="gga", n_prelude=1, n_core=1, n_coda=1), t2, mq)
# E3: TinyStories LM (bytes), matched unique params; tests mixer + looping on natural text.
ts = {"kind": "memmap", "train": "data_cache/ts_train.bin", "valid": "data_cache/ts_valid.bin"}
t3 = dict(seq=256, batch=16, steps=1500, lr=2e-3, warmup=100, eval_every=500, log_every=100, threads=1, eval_batches=8, save=True)
w("E3a_ts_attn4", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1), t3, ts)
w("E3b_ts_hyb4", dict(pattern="ga", n_prelude=1, n_core=2, n_coda=1), t3, ts)
w("E3c_ts_hyb_loop3_rand_sd", dict(pattern="ga", n_prelude=1, n_core=2, n_coda=1, n_loops=3, reinject=True, loop_lora_rank=8),
  {**t3, "loop_sampling": "uniform", "self_distill": 0.5, "eval_loops":[2]}, ts)
