# Knowledge map (auto-rendered from research/knowledge_map.py)

Tags: E = evidence (ref), I = inference, H = hypothesis, ? = unknown. Refs: R* = literature (research/notes/evidence_log.md), E*/F* = own experiments.

## Architecture

| idea | mechanism | param-eff | compute-eff | memory-eff | context | reasoning | stability | weaknesses | refs |
|---|---|---|---|---|---|---|---|---|---|
| Gated DeltaNet (linear attn, delta rule) | O(1) state, delta-rule overwrite, gating | ≈ (E R8) | linear in L (E) | O(1) state (E) | weak exact recall alone (E R9) | ? | good (E R8) | no addressable recall | R8,R9,E3b |
| 3:1 GDN:attention hybrid | few attention layers give recall | + (E E3b vs E3a −0.067 bpb toy) | + (E) | KV in 1/4 layers (E) | good (E R8) | ? | good | now the baseline (Qwen3.5) | R8,R10,E3a,E3b |
| Fine-grained MoE (+shared) | sparse FFN experts | + stored-param capacity (E) | active ≪ total (E) | all experts resident | — | + via loops (E R4) | aux-free balancing (E DeepSeek) | CPU/batch-1 bandwidth bound (I) | R4,E4c |
| Looped core (block loop) | shared weights iterated R times | r^0.46 (E R2) | worse at iso-FLOP (E R4,R21) | KV ×R unless shared (E R14) | — | + manipulation, − memorisation (E R1,R3) | explosion w/o LTI injection (E R20) | latency ×R; recipe-sensitive (E F005, E4g) | R1-R5,R20,R21,E4* |
| LTI stable injection (Parcae) | h←Āh+B̄e+R(h,e), ρ(Ā)<1 | 770M≈1.3B (E R20) | same | same | — | ? | + (E R20) | ? | R20,E4o |
| Token-adaptive depth (lookahead labels) | decider trained on measured CE gain | — | + slope 2.74 vs 1.79 (E R21) | — | — | +3.4 pts (E R21) | avoids halting collapse (I vs R16) | post-training only in R21 | R16,R21,E4p |
| Engram n-gram memory | hashed n-gram table, gated fusion | iso-param win (E R17; E4j −0.059 bpb toy) | iso-FLOP win (E R17) | offloadable, prefetchable (E R17) | + NIAH 84→97 (E R17) | + BBH 5 (E R17) | ? | knowledge collapses if removed (E R17); byte-level confound (?) | R17,E4j |
| Product-key memory | learned key lookup | + iso-token (E R7,R15) | ≈ iso-time (E R15) | SSD decode ok, prefill bad (E R15) | — | — | ? | retrofit fails (E R15) | R7,R15 |
| Shared KV across loops (MELT) | single KV per layer updated by gate | — | — | 3.98× less KV (E R14) | ≈ (E R14) | ≈ (E R14) | ? | fixed R (E R14) | R14,E4e2 |

## Training

| aspect | finding | status/refs |
|---|---|---|
| optimizer | Muon (1 state) + AdamW for 1-D/embeddings | implemented; not yet A/B tested here (?) |
| schedule | WSD 1−sqrt cooldown | implemented; used in all runs |
| loop sampling | fixed R > uniform R at toy scale (E E4g vs E4b); Poisson(R), curriculum queued (E4m,E4n) | F005 |
| distillation | DPT: 2× data-equivalent pass@16, hurts ICL/induction; skip 15% lowest-entropy tokens (E arXiv:2509.01649) | E5a,E5b queued |
| synthetic data | rephrase 1.48×, megadocs 1.80× data efficiency, ~30% synthetic optimum (E arXiv:2603.18534, Kang'25) | spec only |
| RL | narrows pass@k at large k (E R19, contested) | last stage, measure pass@k |
| memory optimisation | padding trim + arena limits −60% RSS (E); grad ckpt no gain at d=128 (E) | TRAINING_SPEC §4 |
| backprop alternatives | rejected for pretraining (I, variance) | untested here |

## Inference

| aspect | finding | status |
|---|---|---|
| KV cache | only attention layers; shared-first across loops implemented | E4e2 queued |
| speculative decoding | loop self-speculation exact (E tests); acceptance not yet measured on trained loop models | bench.py |
| adaptive computation | KL convergence halting implemented; lookahead gate implemented | E4p |
| external memory | Engram tables offloadable (E R17, R15) | C5 |
| quantization | 4-bit tables ≈ fp32 (E R15) | spec |
| test-time scaling | loops alone: steeper slope but lose at matched compute (E R21) | needs token-adaptive depth |

## Data

| aspect | status |
|---|---|
| corpus | TinyStories (toy); FineWeb-Edu/DCLM-class for 1B+ (spec) |
| tokenizer | byte (toy) / BPE-4k TinyStories (3.96 B/token) / BPE-131k spec |
| filtering, dedup | spec (classifier filter + MinHash) |
| synthetic | rephrase/megadocs (E 1.48–1.80×) |

## Evaluation

| aspect | status |
|---|---|
| language modelling | bpb (tokenizer-independent) — implemented |
| reasoning | k-hop chain probes failed (F001–F003); dense-supervision probe pending |
| recall | MQAR configs E2a–c pending |
| coding/math/IF/tool use/multilingual/factuality | not measurable at toy scale; harness spec pending for 50M+ |
