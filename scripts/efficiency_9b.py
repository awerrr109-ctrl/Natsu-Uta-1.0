"""Analytic efficiency comparison at 9B scale (deliverable G). All numbers are derived from configs via the real model code
(meta device) + explicit formulas; no measured GPU numbers. Anchors: Qwen3.5-9B layout (R8) reproduced as C0.
Axes: stored params, active params, FLOPs/token (prefill@4k, decode), weight memory (bf16 / 4-bit), resident vs offloadable,
KV+state memory at 32k and 256k context, decode memory traffic per token (bandwidth-bound latency proxy)."""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig, AttnMixer, GDNMixer
ROOT = os.path.join(os.path.dirname(__file__), "..")

C0 = dict(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn_mult=3.0, n_prelude=0, n_core=32, n_coda=0, pattern="ggga",
          vocab_size=248320, tie_embeddings=False)    # Qwen3.5-9B-like (R8: 32 layers, 3:1, FFN 12288, vocab 248k)
C4 = json.load(open(f"{ROOT}/configs/ladder/C4_9B.json"))["model"]
C3 = {**C4, "n_core": 12, "n_loops": 2, "loop_kv": "shared_first", "moe_experts": 56}   # looped variant at ~iso stored params


def analyse(name, kw):
    with torch.device("meta"):
        m = Natsu(NatsuConfig(**kw))
    c = m.c
    P = m.param_count()
    table = m.engram.table.weight.numel() if m.engram is not None else 0
    experts = sum(p.numel() for n, p in m.named_parameters() if "w_up" in n or "w_down" in n)
    emb_in = m.embed.weight.numel()
    act_frac = (c.moe_topk + c.moe_shared) / (c.moe_experts + c.moe_shared) if c.moe_experts else 1
    # decode bytes read per token (4-bit weights): dense part + active experts + touched table rows + output head
    dense = P - table - experts - emb_in
    rows_touched = (len(c.engram_orders) * c.engram_heads * c.engram_dim) if table else 0
    bytes_tok = (dense + experts * act_frac * min(1, 1)) * 0.5 + rows_touched * 0.5 + c.d_model * 2
    # attention layers with KV: count distinct KV caches (shared_first -> core attention cached once)
    kinds = [c.pattern[i % len(c.pattern)] for i in range(c.n_prelude + c.n_core + c.n_coda)]
    n_attn_once = sum(1 for i, k in enumerate(kinds) if k != "g" and not (c.n_prelude <= i < c.n_prelude + c.n_core))
    n_attn_core = sum(1 for i, k in enumerate(kinds) if k != "g" and c.n_prelude <= i < c.n_prelude + c.n_core)
    kv_layers = n_attn_once + n_attn_core * (1 if c.loop_kv == "shared_first" or c.n_loops == 1 else c.n_loops)
    kv_per_tok = kv_layers * 2 * c.n_kv_heads * c.head_dim * 2           # bf16
    n_gdn = sum(1 for k in kinds if k == "g") + sum(1 for i, k in enumerate(kinds) if k == "g" and c.n_prelude <= i < c.n_prelude + c.n_core) * (c.n_loops - 1)
    state = n_gdn * c.n_heads * c.head_dim * c.head_dim * c.gdn_expand_v * 4  # fp32 state per sequence
    return dict(name=name, stored_B=P / 1e9, engram_table_B=table / 1e9, experts_B=experts / 1e9,
                resident_B_if_table_offloaded=(P - table) / 1e9,
                gflop_tok_prefill4k=m.flops_per_token(seq=8192) / 1e9, gflop_tok_decode32k=m.flops_per_token(seq=65536) / 1e9,
                weights_bf16_GB=P * 2 / 1e9, weights_4bit_GB=P * 0.5 / 1e9, resident_4bit_GB_table_offloaded=(P - table) * 0.5 / 1e9,
                kv_MB_at_32k=kv_per_tok * 32768 / 1e6, kv_GB_at_256k=kv_per_tok * 262144 / 1e9, recurrent_state_MB=state / 1e6,
                decode_bytes_per_token_GB=bytes_tok / 1e9)


rows = [analyse("C0 Qwen3.5-9B-like dense 3:1", C0), analyse("C4 Natsu-9B (MoE+Engram)", C4), analyse("C3' looped variant (shared KV)", C3)]
keys = list(rows[0])[1:]
L = ["| metric | " + " | ".join(r["name"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
for k in keys:
    L.append(f"| {k} | " + " | ".join(f"{r[k]:.3f}" for r in rows) + " |")
os.makedirs(f"{ROOT}/docs/generated", exist_ok=True)
open(f"{ROOT}/docs/generated/efficiency_9b.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
