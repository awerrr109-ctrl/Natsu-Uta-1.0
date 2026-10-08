"""Exact parameter / FLOP accounting of 9B-scale candidates using the real model code on the meta device
(zero RAM). Output: docs/generated/candidates.md + configs/*.json"""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig
ROOT = os.path.join(os.path.dirname(__file__), "..")
V = 131072
common = dict(vocab_size=V, max_seq=262144, tie_embeddings=False, rope_theta=1e6, chunk=64)
C = {
 # C0: reference = Qwen3.5-9B-like dense 3:1 GDN hybrid (our reproduction of the strongest public 9B layout)
 "C0_dense_hybrid_3to1": dict(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn_mult=3.0,
        n_prelude=0, n_core=32, n_coda=0, n_loops=1, pattern="ggga"),
 # C1: dense looped hybrid (Huginn/Ouro style): 4 prelude + 8-layer core x4 + 4 coda
 "C1_dense_looped": dict(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn_mult=3.0,
        n_prelude=4, n_core=24, n_coda=4, n_loops=2, pattern="ggga", reinject=True),
 # C2: looped-MoE hybrid (R4 evidence), no memory
 "C2_looped_moe": dict(d_model=3072, n_heads=24, n_kv_heads=4, head_dim=128, ffn_mult=3.0,
        n_prelude=2, n_core=12, n_coda=2, n_loops=3, pattern="ggga",
        moe_experts=56, moe_topk=6, moe_shared=2, moe_expert_mult=0.375, loop_lora_rank=0),
 # C3 (MAIN candidate "Natsu-9B"): looped-MoE hybrid core + PKM memory coda + MTP + per-loop router bias + depth gate
 "C3_natsu_loopmoe_mem": dict(d_model=3072, n_heads=24, n_kv_heads=4, head_dim=128, ffn_mult=3.0,
        n_prelude=2, n_core=12, n_coda=2, n_loops=3, pattern="ggga",
        moe_experts=48, moe_topk=6, moe_shared=2, moe_expert_mult=0.375,
        pkm_keys=512, pkm_topk=32, depth_gate=True, reinject=True, mtp=1),
 # C4: non-looped MoE hybrid with same stored params (control for "is it the loop or the MoE?")
 "C4_moe_noloop": dict(d_model=3072, n_heads=24, n_kv_heads=4, head_dim=128, ffn_mult=3.0,
        n_prelude=2, n_core=12, n_coda=2, n_loops=1, pattern="ggga",
        moe_experts=48, moe_topk=6, moe_shared=2, moe_expert_mult=0.375, pkm_keys=512, pkm_topk=32, mtp=1),
}
os.makedirs(f"{ROOT}/configs", exist_ok=True); os.makedirs(f"{ROOT}/docs/generated", exist_ok=True)
rows = ["| cand | stored params | non-embed | active params/token (1 pass) | fwd GFLOP/token @4k ctx (all loops) | eff. depth |", "|---|---|---|---|---|---|"]
for n, kw in C.items():
    cfg = NatsuConfig(**{**common, **kw})
    with torch.device("meta"):
        m = Natsu(cfg)
    P = m.param_count(); Pne = P - m.embed.weight.numel() - (0 if m.head is None else m.head.weight.numel())
    # active params: non-MoE params + active experts
    act = 0
    for name, p in m.named_parameters():
        if "w_up" in name or "w_down" in name:
            E = p.shape[0]; act += p.numel() * (cfg.moe_topk + cfg.moe_shared) / E
        elif "values" in name:
            act += cfg.pkm_topk * cfg.d_model
        elif "embed" in name:
            act += cfg.d_model
        else:
            act += p.numel()
    fl = m.flops_per_token(seq=8192)
    depth = cfg.n_prelude + cfg.n_core * cfg.n_loops + cfg.n_coda
    rows.append(f"| {n} | {P/1e9:.2f}B | {Pne/1e9:.2f}B | {act/1e9:.2f}B | {fl/1e9:.1f} | {depth} |")
    json.dump({"name": n, "model": {**common, **kw}}, open(f"{ROOT}/configs/{n}.json", "w"), indent=1)
open(f"{ROOT}/docs/generated/candidates.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
