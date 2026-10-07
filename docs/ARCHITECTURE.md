# Natsu-Uta Architecture Specification (v0.1, session 1 — candidate stage)

Status: **candidate design, not validated at scale.** Every design decision is tagged [E] evidence,
[I] inference, or [H] hypothesis, and lists the experiment that tests it.

## 1. Design thesis

At a fixed **stored** parameter budget (≈9B), capability comes from three channels that scale
with different resources. Natsu assigns each channel its cheapest substrate.

| channel | what limits it | substrate in Natsu | evidence |
|---|---|---|---|
| knowledge storage | bits/param (≈2 bits/param, Allen-Zhu & Li 2024, arXiv:2404.05405, L2) | sparse MoE experts + product-key memory (stored, rarely-touched params) + retrieval/tools (zero params) | [E] R7 memory layers > dense with 2× FLOPs on factual tasks |
| knowledge manipulation / reasoning depth | sequential depth, not params | **weight-shared looped core** iterated R times; R is chosen at inference | [E] R1, R3, R5; [E] φ=0.46 (R2) |
| context access | KV memory, recall | 3:1 Gated-DeltaNet : attention hybrid; attention never looped (shared KV across loops) | [E] R8, R9; [E-weak] MELT (arXiv:2605.07721, L1) |

Key [I]: under an iso-*parameter* constraint the R2 scaling law favours looping. Effective capacity is
N_once + r^φ·N_rec. With 75% of the parameters in the looped core and r=3, φ=0.46, that gives ≈1.4–1.5× effective
parameters, paid for in FLOPs. Dense looping has an FFN-expressivity bottleneck, which looped-MoE removes
through routing divergence (R4), so **the looped core must be MoE**.

## 2. Topology (main candidate C3 "Natsu-9B")

```
tokens ─ embed(131k) ─ PRELUDE (2 blocks, unshared, dense FFN)
        ─ CORE (12 blocks, weights shared, looped R ∈ {1..R_max}, train R_max=3)
              per loop r: x ← Inj([x, e_prelude]) + loop_emb[r]          (Huginn-style re-injection)
                         gate_r(x) ∈ (0,1) per token  (soft at train, hard skip at inference)
              block = mixer(pattern g,g,g,a) + MoE-FFN(56 routed exp., top-6, 2 shared, per-loop router bias)
        ─ CODA (2 blocks, unshared; last FFN = product-key memory 512² slots, top-32)
        ─ head (untied) + 1 MTP head
```

Exact counts from `scripts/size_candidates.py`, which instantiates the real model on the meta device:

| candidate | stored | non-embed | active/token (1 pass) | fwd GFLOP/token @8k (all loops) | eff. depth |
|---|---|---|---|---|---|
| C0 dense 3:1 hybrid (Qwen3.5-9B-like reproduction) | 8.26B | 7.19B | 7.73B | 16.1 | 32 |
| C1 dense looped (4+24×2+4) | 8.30B | 7.22B | 7.76B | 27.3 | 56 |
| C2 looped-MoE, no memory | 9.19B | 8.39B | 2.42B | 11.5 | 40 |
| **C3 looped-MoE + PKM + gate + MTP (main)** | **8.93B** | 8.13B | **2.37B** | **11.4** | 40 |
| C4 MoE + PKM, no loop (control) | 8.91B | 8.11B | 2.36B | 4.9 | 16 |

The C3 vs C4 contrast isolates the loop at identical stored params. C3 vs C0 is the headline comparison:
same stored params and 29% fewer FLOPs per token at R=3.

## 3. Components and their differences from prior art

| component | prior art | Natsu delta | why it matters at 9B | test |
|---|---|---|---|---|
| Gated DeltaNet mixer | Yang et al. 2024; Qwen3.5 | none (pure-PyTorch chunkwise kernel, verified vs recurrence to 1e-8) | O(1) state, cheap to loop | tests/test_deltanet.py |
| 3:1 hybrid | Qwen3.5, Kimi Linear | attention layers inside the loop **share one KV cache across loops** ([H], MELT-like) | removes the R× KV multiplier | E6 (planned) |
| looped core | Huginn, Ouro, R4 | loop core is MoE **with learned per-loop router bias** to promote routing divergence | [H] raises φ | E4 (planned) |
| learned depth | Ouro (entropy-regularised exit), MoR | per-token soft gate on loops r≥1 with FLOP penalty; GDN skip = identity state update (a=1, β=0), which keeps the recurrent state exact | skipping tokens is free for GDN layers | E3/E5 |
| knowledge memory | Memory Layers at Scale | placed **outside** the loop (coda), so memory lookups are not repeated R times | knowledge capacity without R× FLOPs | E7 (planned) |
| self-speculation | LoopSpec (training-free) | **train-time shallow→deep KL self-distillation** so loop-1 is a good draft | spec decoding with no extra params | E3c, E5 |
| MTP head | DeepSeek-V3, Qwen3.5 | none | second draft source | — |

### Novelty statement (honest)
No single component is new. The distinct, testable claim is the **combination under an iso-parameter
constraint**:
(a) loop only the cheap-state layers, plus MoE, with a per-loop router bias;
(b) shared-KV attention inside the loop;
(c) knowledge memory outside the loop;
(d) self-distilled shallow loops serving as built-in drafts;
(e) a per-token learned loop budget.

The web/arXiv searches this session (queries in `research/taxonomy.py`, P11) found prior work for each
part individually (R4, R5, R6, R7, MELT) but no work combining looped-MoE with memory layers. That is a
**candidate gap** and is not yet confirmed: L3 reading of R4 and MELT related-work sections is pending.

## 4. Known weaknesses / risks
1. Loops multiply latency per token by about R (mitigation: self-spec, adaptive depth). [E] R2: training FLOPs grow with r.
2. MoE on CPU / small batch is memory-bandwidth bound; the 9B stored params must be resident (4-bit ≈ 4.5 GB).
3. GDN recall gap (R9) means attention layers must not be pruned. Their KV is the long-context memory cost.
4. Truncated BPTT through loops must not be used (R2: φ drops).
5. PKM training instability and under-utilisation are known issues. Mitigations are query batch-norm and a high top-k.

## 5. Size ladder (same code, different config)
| stage | d | core×R | experts | params | where |
|---|---|---|---|---|---|
| toy | 128 | 2×3 | 0–8 | 0.8–1.5M | this sandbox (CPU) |
| 10M | 256 | 4×3 | 16 | ~10M | sandbox (slow) |
| 50M / 100M / 300M | 512 / 768 / 1024 | 8×3 | 32 | — | 1 GPU |
| 1B | 1536 | 12×3 | 48 | — | 8 GPU |
| 9B | 3072 | 12×3 | 56 | 8.9B | cluster |

## 6. Session-1 revision (after toy experiments E3/E4 and F005)
- [E, toy] At iso-param, the **non-looped hybrid MoE (E4c) is Pareto-best**. Looped variants are 0.08–0.10 bpb worse (F005), partly
  due to a re-injection init defect (now fixed: identity init).
- Knowledge memory component changed **PKM → Engram** (R17: iso-param, iso-FLOP win at 27B; prefetchable addresses).
  PKM is kept as an option.
- **Main line is now dual**: C4 (hybrid + MoE + Engram, no loop) and C3 (looped). C3 continues only if E4g–i bring loops within 0.01 bpb
  of E4c (pre-registered rule in REPORT_S1 §4).
- Implemented since v0.1: `loop_kv="shared_first"` (core attention KV reused across loops, cache size independent of R) and `Engram`
  (hashed 2/3-gram tables, gated fusion, causal conv, decode state, prefetchable addressing).

## 7. Session-2 revision (v0.3): decision state after E4g–E4p, E5, E6, E8, F007, F008
**Main line: C4 = 3:1 GDN:attention hybrid + fine-grained MoE + Engram (no loop in the trained core), trained with KD where a teacher exists.**

| component | status | evidence |
|---|---|---|
| hybrid GDN+attn | keep | −0.067 bpb vs attention-only (E3) |
| fine-grained MoE + shared expert, aux-free bias | keep | E4c; analytic 3.1× fewer FLOPs than C0 at 9B |
| Engram (hashed 2/3-gram, gated, conv) | keep **conditionally**: independent 1.3B dense report found no iso-param gain (R40); 10M 2×2 {MoE,dense}×{Engram,none} queued; downstream eval at ≥50M required before committing 1.6B params; delayed-memory schedule under test (H14.1) | −0.054 bpb, 2 seeds; survives BPE (−0.014); possible harm to in-context retrieval (E6, weak) |
| Engram v2 VIP rows | under test (E4s) | R24 failure mode (1) |
| looped core | **demoted at toy scale with our recipe; NOT refuted at scale** | 3-seed gain −0.013 (t≈2) at 1.73× FLOPs; substitutes with Engram (F007, F010); unique depth ≫ loops on addition (E8e 0.484 vs E8b 0.285). Counter-evidence at 100M–1T-token scale: R47 LOOM (per-loop routers + residual scaling; near-iso-FLOP gains at 700M), R48 (looped MoE ≈ 2× larger MoE on reasoning). R47 recipe under toy test (E4y/E4y2); decisive test = 50M+ ladder |
| shared-first KV across loops | adopted for any loop variant | E4e2: within noise, 3× less core KV |
| exit-trained lookahead gate | adopted for any loop variant | E4p: every depth usable (R1…R6 ≈ 1.51) |
| digit-position (Abacus) embedding | under test (E8g/h) | R36 |
| MTP head | keep | R34 (teacherless/MTP avoids Clever-Hans failures) |

Why loops are not dropped: R33 says loops help reasoning while hurting perplexity, and our probes so far measure perplexity or are positionally bottlenecked (R36).
The decision is re-opened if E8h > E8g on 6-digit generalisation, or if E6d/E8d show complementarity on reasoning probes.
The deployed loop form is then "C4 + exit-trained gated loop over the last k core blocks with shared-first KV", trained as a second stage.

## 8. Session-2 revision (v0.4, 2026-10-07): Engram recipe and routing
| component | v0.4 status | evidence |
|---|---|---|
| Engram output path | **paper-faithful path is the ICL default**: varchain 3/3 transitions at steps 400–600 vs 1/3 (v1, none) | F013, E6Le ×3 |
| Engram table lr | **lr ×5 is the strongest LM lever found** for v1 (−0.029, 1 seed, replication queued). The paper path is hurt by ×5 on LM (+0.007) but helped on ICL (1000 → 600 steps) → interaction | E4pq/pr/pp, E6Lg, R67 |
| Engram recipe for 10M | **chosen automatically by a pre-registered rule** (`scripts/pick_engram_recipe.py`): best LM mean among recipes with ≥2 seeds and an ICL arm transitioning ≥2/2 | REPORT_S2 §9 |
| frequency-adaptive table lr | under test (H15.2, E4pt) | R67 FAL |
| factorised table / VIP rows | not adopted at toy scale (+0.016…+0.031 / n.s.) | H14.7, E4s2 |
| delayed memory (N5) | rejected | E4u |
| per-loop routers / per-loop MoE LoRA | not adopted (n.s. at R3). If loops are used: residual scaling + LTI (better R1/R6 robustness) | E4y/y2/y3 |
| routing collapse across loops | **kept on purpose under RAM-bound deployment**: loop r>0 expert reads become cache hits | route_diag, INFERENCE_SPEC v0.3 |
| retrieval channel | kNN-LM measured (−0.13/−0.22 bpb, corpus-inflated); retrieval-distilled Engram (G1c) under test, so deployment needs no datastore | knnlm, NEXT_GEN G1c |
| Engram on algorithmic tasks | v1 interferes with addition (E8i 0.176 vs 0.297); paper-path retest E8j queued | E8i |
