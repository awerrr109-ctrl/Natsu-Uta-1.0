"""
Structured knowledge map (principle #7). Source of truth = this file (curated entries) + reads table (provenance).
Every entry references evidence ids (R*/E*), and every cell is tagged E (evidence), I (inference), H (hypothesis), ? (unknown).
Renders docs/KNOWLEDGE_MAP.md.
"""
import os

ARCH = [  # idea, mechanism, param-eff, compute-eff, memory-eff, context, reasoning, stability, weaknesses, refs
 ("Gated DeltaNet (linear attn, delta rule)", "O(1) state, delta-rule overwrite, gating", "≈ (E R8)", "linear in L (E)", "O(1) state (E)",
  "weak exact recall alone (E R9)", "?", "good (E R8)", "no addressable recall", "R8,R9,E3b"),
 ("3:1 GDN:attention hybrid", "few attention layers give recall", "+ (E E3b vs E3a −0.067 bpb toy)", "+ (E)", "KV in 1/4 layers (E)",
  "good (E R8)", "?", "good", "now the baseline (Qwen3.5)", "R8,R10,E3a,E3b"),
 ("Fine-grained MoE (+shared)", "sparse FFN experts", "+ stored-param capacity (E)", "active ≪ total (E)", "all experts resident",
  "—", "+ via loops (E R4)", "aux-free balancing (E DeepSeek)", "CPU/batch-1 bandwidth bound (I)", "R4,E4c"),
 ("Looped core (block loop)", "shared weights iterated R times", "r^0.46 (E R2)", "worse at iso-FLOP (E R4,R21)", "KV ×R unless shared (E R14)",
  "—", "+ manipulation, − memorisation (E R1,R3)", "explosion w/o LTI injection (E R20)", "latency ×R; recipe-sensitive (E F005); substitutes with Engram on bpb (E F007); no gain on addition at iso-param (E E8b) — positional bottleneck? (H R36)", "R1-R5,R20,R21,R33,R36,E4*,E8b"),
 ("LTI stable injection (Parcae)", "h←Āh+B̄e+R(h,e), ρ(Ā)<1", "770M≈1.3B (E R20)", "same", "same", "—", "?", "+ (E R20)", "?", "R20,E4o"),
 ("Token-adaptive depth (lookahead labels)", "decider trained on measured CE gain", "—", "+ slope 2.74 vs 1.79 (E R21)", "—", "—", "+3.4 pts (E R21)",
  "avoids halting collapse (I vs R16)", "gate AUC only 0.66 at toy (E E4p); exit training makes all depths usable (E E4p R1..R6 1.51)", "R16,R21,E4p"),
 ("Engram n-gram memory", "hashed n-gram table, gated fusion", "iso-param win (E R17; E4j −0.059 bpb toy)", "iso-FLOP win (E R17)",
  "offloadable, prefetchable (E R17); editable as removable overlay (E R39)", "+ NIAH 84→97, VT 77→89 (E R17); toy varchain worse (weak E E6)", "+ BBH 5 (E R17)", "lower seed variance 4.4× (E own)", "iso-param win only vs MoE, not dense (E R40, R17 Table 1); gain shrinks byte→BPE→9B (0.053→0.014→~0.004 bpb); optimum ρ≈75–80% (E R17)", "R17,R39,R40,E4j(3 seeds)"),
 ("Product-key memory", "learned key lookup", "+ iso-token (E R7,R15)", "≈ iso-time (E R15)", "SSD decode ok, prefill bad (E R15)", "—", "—",
  "?", "retrofit fails (E R15)", "R7,R15"),
 ("Digit-position embedding (Abacus)", "embed position within digit run", "—", "—", "—", "length gen. 20→100 digits (E R36)", "enables loop gains on arithmetic (E R36)", "?", "arithmetic-specific", "R36,E8g,E8h"),
 ("Shared KV across loops (MELT)", "single KV per layer updated by gate", "—", "—", "3.98× less KV (E R14)", "≈ (E R14)", "≈ (E R14)", "?", "fixed R (E R14)", "R14,E4e2"),
]
TRAIN = [
 ("seed variance", "0.003–0.014 bpb at toy; data order dominates; claims need ≥0.02 or ≥3 seeds", "F008"),
 ("MoE hyperparameters", "iso-total MoE ≥ dense given more tokens; LR ∝ E^-0.25 (E R42)", "TRAINING_SPEC"),
 ("distributed data", "epoch-exact sharded sampler + exact resume (fixed F009, motivated by parcae#10)", "train_dist.py"),
 ("optimizer", "Muon (1 state) + AdamW for 1-D/embeddings; block-wise 8-bit Adam 3.9× less state (E test)", "Muon not yet A/B tested here (?)"),
 ("schedule", "WSD 1−sqrt cooldown", "implemented; used in all runs"),
 ("loop sampling", "3-seed: loop −0.013 bpb (t≈2) at 1.73× FLOPs; recipe differences <0.02 are within seed noise (F008); exit training makes all depths usable (E4p)", "F005 upd.4, F008"),
 ("distillation", "DPT KD: −0.035 (E5a) / −0.022 on top of Engram (E5b 1.406, toy best); gain is under-trained regime, fades at high tok/param (E R35)", "E5a,E5b; E5c control queued"),
 ("synthetic data", "rephrase 1.48×, megadocs 1.80× data efficiency, ~30% synthetic optimum (E arXiv:2603.18534, Kang'25)", "spec only"),
 ("RL", "narrows pass@k at large k (E R19, contested)", "last stage, measure pass@k"),
 ("memory optimisation", "padding trim + arena limits −60% RSS (E); grad ckpt no gain at d=128 (E)", "TRAINING_SPEC §4"),
 ("backprop alternatives", "rejected for pretraining (I, variance)", "untested here"),
]
INFER = [
 ("KV cache", "only attention layers; shared-first across loops implemented", "E4e2 queued"),
 ("speculative decoding", "loop self-speculation exact (E tests); acceptance not yet measured on trained loop models", "bench.py"),
 ("adaptive computation", "KL convergence halting implemented; lookahead gate implemented", "E4p"),
 ("external memory", "Engram tables offloadable (E R17, R15)", "C5"),
 ("quantization", "4-bit tables ≈ fp32 (E R15)", "spec"),
 ("test-time scaling", "loops alone: steeper slope but lose at matched compute (E R21)", "needs token-adaptive depth"),
]
DATA = [
 ("corpus", "TinyStories (toy); FineWeb-Edu/DCLM-class for 1B+ (spec)"),
 ("tokenizer", "byte (toy) / BPE-4k TinyStories (3.96 B/token) / BPE-131k spec"),
 ("filtering, dedup", "keep F_opt ∝ C^0.25 → ~25–30% at 9B (E R32); classifier ensembles (E R30); held-out Goodhart guard (E R31,R32)"),
 ("synthetic", "rephrase/megadocs (E 1.48–1.80×)"),
]
EVAL = [
 ("language modelling", "bpb (tokenizer-independent) — implemented"),
 ("reasoning", "k-hop chain probes failed (F001–F003); addition E8 acc ~0.29 for loop and no-loop; varchain E6 running"),
 ("recall", "MQAR configs E2a–c pending; held-out name-consistency MC set (in-context retrieval) built, decontaminated (171 items)"),
 ("downstream beyond bpb", "held-out TinyStories cloze MC (347 items, 13% contaminated items removed); ds_eval.py post-queue (R31/R40: bpb ≠ capability)"),
 ("coding/math/IF/tool use/multilingual/factuality", "not measurable at toy scale; harness spec pending for 50M+"),
]


def render():
    L = ["# Knowledge map (auto-rendered from research/knowledge_map.py)\n",
         "Tags: E = evidence (ref), I = inference, H = hypothesis, ? = unknown. Refs: R* = literature (research/notes/evidence_log.md), E*/F* = own experiments.\n",
         "## Architecture\n", "| idea | mechanism | param-eff | compute-eff | memory-eff | context | reasoning | stability | weaknesses | refs |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in ARCH]
    for title, rows, hdr in (("Training", TRAIN, "| aspect | finding | status/refs |"), ("Inference", INFER, "| aspect | finding | status |")):
        L += [f"\n## {title}\n", hdr, "|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
    for title, rows in (("Data", DATA), ("Evaluation", EVAL)):
        L += [f"\n## {title}\n", "| aspect | status |", "|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "KNOWLEDGE_MAP.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    render()
