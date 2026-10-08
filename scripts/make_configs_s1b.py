"""E1 v2 (after F001): learnability-gated pointer chasing with per-hop eval; addition as second reasoning task."""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")
base_m = dict(d_model=128, n_heads=2, n_kv_heads=1, head_dim=64, chunk=32)
def w(name, model, train, data, extra):
    json.dump({"name": name, "model": {**base_m, **model}, "train": train, "data": data, "extra_eval": extra}, open(f"{OUT}/{name}.json", "w"), indent=1)
chain = {"kind": "synthetic", "task": "chain", "kw": {"min_hops": 1, "hops": 3, "distract": 2}}
ev = {f"hop{h}": {"min_hops": h, "hops": h} for h in (1, 2, 3, 4, 6)}
t = dict(seq=64, batch=32, steps=2500, lr=2e-3, warmup=100, eval_every=500, log_every=250, threads=2, eval_batches=4, eval_loops=[2,3,4,6], save=False)
w("E1v2a_dense4", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1), t, chain, ev)
w("E1v2c_loop2x3", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1, n_loops=3), t, chain, ev)
w("E1v2b_dense8", dict(pattern="a", n_prelude=2, n_core=4, n_coda=2), t, chain, ev)
w("E1v2d_loop2x4_rand", dict(pattern="a", n_prelude=1, n_core=2, n_coda=1, n_loops=4, reinject=True), {**t, "loop_sampling": "uniform"}, chain, ev)
