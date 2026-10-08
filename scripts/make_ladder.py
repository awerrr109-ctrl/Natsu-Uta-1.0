"""Size ladder for the main line C4 (hybrid 3:1 GDN:attn + fine-grained MoE + Engram), named by STORED params.
Engram table ≈ 20% of the sparse (MoE experts + table) budget (R17 U-curve). Writes configs/ladder/*.json + docs/generated/ladder.md"""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig
ROOT = os.path.join(os.path.dirname(__file__), "..")
L = [
 ("toy",  dict(d_model=128, n_heads=2, n_kv_heads=1, head_dim=64, n_prelude=1, n_core=2, n_coda=1, pattern="ga", moe_experts=8, moe_topk=2, moe_expert_mult=0.17, engram_slots=512, engram_heads=2, engram_dim=32, vocab_size=260)),
 ("10M",  dict(d_model=320, n_heads=5, n_kv_heads=1, head_dim=64, n_prelude=1, n_core=4, n_coda=1, pattern="ggga", moe_experts=8, moe_topk=2, moe_expert_mult=0.33, engram_slots=4096, engram_heads=2, engram_dim=32, vocab_size=4096)),
 ("50M",  dict(d_model=512, n_heads=8, n_kv_heads=2, head_dim=64, n_prelude=1, n_core=6, n_coda=1, pattern="ggga", moe_experts=12, moe_topk=2, moe_expert_mult=0.33, engram_slots=12288, engram_heads=4, engram_dim=32, vocab_size=32768)),
 ("100M", dict(d_model=640, n_heads=10, n_kv_heads=2, head_dim=64, n_prelude=2, n_core=8, n_coda=2, pattern="ggga", moe_experts=12, moe_topk=2, moe_expert_mult=0.33, engram_slots=16384, engram_heads=4, engram_dim=64, vocab_size=32768)),
 ("300M", dict(d_model=1024, n_heads=16, n_kv_heads=4, head_dim=64, n_prelude=2, n_core=8, n_coda=2, pattern="ggga", moe_experts=16, moe_topk=2, moe_expert_mult=0.33, engram_slots=49152, engram_heads=4, engram_dim=64, vocab_size=32768)),
 ("1B",   dict(d_model=1536, n_heads=12, n_kv_heads=4, head_dim=128, n_prelude=2, n_core=12, n_coda=2, pattern="ggga", moe_experts=24, moe_topk=3, moe_expert_mult=0.25, engram_slots=98304, engram_heads=8, engram_dim=64, vocab_size=65536, tie_embeddings=False)),
 ("9B",   dict(d_model=3072, n_heads=24, n_kv_heads=4, head_dim=128, n_prelude=2, n_core=20, n_coda=2, pattern="ggga", moe_experts=36, moe_topk=4, moe_shared=2, moe_expert_mult=0.25, engram_slots=786432, engram_heads=8, engram_dim=128, vocab_size=131072, tie_embeddings=False, mtp=1)),
]
rows = ["| stage | stored | non-embed | Engram table | table share of sparse | active/token | GFLOP/token @4k | 20×params tokens |", "|---|---|---|---|---|---|---|---|"]
os.makedirs(f"{ROOT}/configs/ladder", exist_ok=True)
for n, kw in L:
    kw = {"moe_shared": 1, "engram_orders": [2, 3], **kw}
    with torch.device("meta"):
        m = Natsu(NatsuConfig(**kw))
    P = m.param_count(); emb = m.embed.weight.numel() + (0 if m.head is None else m.head.weight.numel())
    tab = m.engram.table.weight.numel()
    exp = sum(p.numel() for nm, p in m.named_parameters() if "w_up" in nm or "w_down" in nm)
    act = 0
    for nm, p in m.named_parameters():
        if "w_up" in nm or "w_down" in nm:
            act += p.numel() * (kw["moe_topk"] + kw["moe_shared"]) / p.shape[0]
        elif "engram.table" in nm:
            act += len(kw["engram_orders"]) * kw["engram_heads"] * kw["engram_dim"]
        elif "embed" in nm:
            act += kw["d_model"]
        else:
            act += p.numel()
    rows.append(f"| {n} | {P/1e6:,.1f}M | {(P-emb)/1e6:,.1f}M | {tab/1e6:,.1f}M | {tab/(tab+exp)*100:.0f}% | {act/1e6:,.1f}M | {m.flops_per_token(seq=4096)/1e9:.3f} | {20*P/1e9:.2f}B |")
    json.dump({"name": f"C4_{n}", "model": kw}, open(f"{ROOT}/configs/ladder/C4_{n}.json", "w"), indent=1)
open(f"{ROOT}/docs/generated/ladder.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
