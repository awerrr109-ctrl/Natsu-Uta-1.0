# Session 1 Research Report — Natsu-Uta (2026-10-05)

Deliverables covered at session-1 depth: A (review), F (toy benchmark), G (efficiency), H (novelty),
I (failures), J (next-gen). Specs: `ARCHITECTURE.md`, `TRAINING_SPEC.md`, `INFERENCE_SPEC.md`, `TARGET_ANALYSIS.md`.
Tags: **[E]** evidence, **[I]** inference, **[H]** hypothesis.

## 1. Honest accounting of the search (principle #28)

| item | number | meaning |
|---|---|---|
| hypotheses in taxonomy | 35 across 11 problem areas | each with support **and** refute queries |
| queries executed | arXiv 244, Crossref 244, OpenAlex 98 (free daily budget exhausted), GitHub 410 (5 star strata) | table `queries_done` |
| unique papers harvested | **32,381** | metadata (+abstract when available), title-dedup |
| unique repos harvested | **14,816** | stars 0–1 … >1000, incl. new/abandoned |
| L1 on-topic estimate | ~9.5k papers / ~9.5k repos | keyword relevance — **not reading** |
| refute-tagged papers | 10,011 | from failure/negative-result queries |
| **read at L2** | **~20 sources** | `research/notes/evidence_log.md`, table `reads` |
| **read at L3-partial** | 3 (Looped-MoE, MELT, Engram) | targeted sections |

The 10k/10k target was met **as a provenance-tagged, triaged corpus**, not as reading. Reading is ~20 sources.
Constraints: OpenAlex budget and arXiv 429 throttling on day 1; Semantic Scholar 429 without a key; GitHub issues/PRs not
yet harvested. Reading order for next sessions: `research/notes/l2_queue.md` (support and refute per hypothesis).

## 2. Review: what the evidence says (A)
**Depth via looping.** [E] k×L looped ≈ kL layers on reasoning, but looping hurts memorisation (R1). φ=0.46 recurrence
exponent; truncated BPTT lowers it and hyperconnections raise it to 0.65 (R2). Ouro 1.4B/2.6B ≈ 12B (7.7T tokens) via knowledge
*manipulation* (R3). **Dense loops scale worse than non-looped; looped-MoE beats dense base, but non-looped MoE is best at
iso-FLOP** (R4). Layer-loop R=2 MoE (Loopie) beats a compute-matched 30B-A3B (R5). Shared KV across loops ≈ free (MELT, R14).
Counter: learned halting collapses to 1 step (R16, 0-star repo).

**Knowledge.** [E] ~2 bits/param (R12). Memory layers help at iso-token but little at iso-time; retrofit fails (R15, 0-star repo).
**Engram n-gram memory beats strict iso-param, iso-FLOP MoE at 27B, including reasoning** (R17); ~20–25% of the sparse budget
should go to memory; addresses are prefetchable.

**Mixing.** [E] 3:1 GDN:attention is now industry standard (Qwen3.5-9B already uses it) → baseline, not novelty (R8). Exact
recall only via attention KV (R9).

**Training.** [E] Distillation wins when a teacher exists (R18). RLVR narrows pass@k at large k (R19, contested).

## 3. Toy experiments (F) — CPU, ~0.8M params, 1.64M byte-tokens, single seed

Pareto table (auto, `docs/generated/pareto_toy.md`). Val bpb, lower is better:

| run | design | bpb (best R) | stored params | train FLOPs | inf MFLOP/tok | Pareto |
|---|---|---|---|---|---|---|
| **E4c** | hybrid GDN:attn + MoE, no loop | **1.4870** | 0.82M | 7.3e12 | 1.49 | **yes** |
| E3b | hybrid dense, no loop | 1.4917 | 0.82M | 9.0e12 | 1.84 | |
| E3a | attention-only dense | 1.5588 | 0.76M | 8.1e12 | 1.64 | yes (fewest params) |
| E4d | "Natsu toy": loop-MoE + gate + MTP + self-distill | 1.5647 (R=2) | 0.87M | 1.0e13 | 2.03 | |
| E4b | loop-MoE (R=3, uniform R) | 1.5748 (R=2) | 0.85M | 1.0e13 | 2.03 | |
| E4a | loop-dense | 1.5869 (R=2) | 0.86M | 1.4e13 | 2.72 | |

Findings:
1. [E, toy] **Hybrid GDN:attention beats attention-only by 0.067 bpb** at near-iso-param (E3b vs E3a). This is consistent with the literature.
2. [E, toy] **MoE ≥ dense** in both settings (−0.005 no-loop, −0.012 looped). This is consistent with R4.
3. [E, toy] **Looping hurt by 0.08–0.10 bpb at iso-param**, and loop counts R>2 got monotonically worse, so there was no test-time scaling.
   → **The main thesis is NOT supported at toy scale.** Root-cause analysis in F005 found a real defect: random-init re-injection
   destroyed the residual stream (cos≈0). It is fixed with identity init, and three ablations are queued (E4g/h/i) with a
   pre-registered decision rule.
4. [E, toy] Natsu extras (self-distill + MTP + gate) recover part of the gap (−0.010 vs E4b). The learned gate skips only 5% at
   threshold 0.5, so there is no meaningful learned budget yet (consistent with the R16 warning).
5. [E] Engineering: GDN chunk kernel = recurrence (1e-8). Cached decoding is exact for all variants. Loop self-speculation reproduces target
   greedy output exactly. Peak training RSS 655–842 MB on the 1 GB box.

## 4. Updated design decision (§16, decision rule)
Pre-registered: if the identity-init ablations (E4g–i) do not bring looping to within noise (≤0.01 bpb) of E4c at iso-param,
**the 9B main line becomes C4 = hybrid 3:1 GDN:attention + fine-grained MoE + Engram memory (no loop)**. Looping is then kept only
as an optional *test-time-compute* module, to be evaluated on reasoning probes at the GPU stage. This matches R4's iso-FLOP
ranking and R17's iso-param result, and now has our own toy evidence too.

| candidate | status after session 1 |
|---|---|
| C0 dense 3:1 | baseline (Qwen3.5-like) |
| C1 dense looped | **rejected** (R2, R4, E4a) |
| C2 looped-MoE | weakened (E4b) — pending E4g–i |
| C3 looped-MoE + memory | weakened — pending E4f (Engram + loop) |
| **C4 MoE + Engram, no loop** | **promoted to co-main line** (E4c Pareto-best; R17) |

## 5. Novelty (H) — explainable differences, stated conservatively
No component is new. Gaps found where no prior work turned up in our searches: (a) looped-MoE + prefetchable n-gram memory placed before
the loop; (b) self-distilled shallow loops as built-in drafts; (c) per-loop router bias; (d) a 9B *resident* model + SSD-resident
Engram tables (C5). (a)–(c) are now weakened by our own toy results. (d) is the most promising, and has supporting evidence (R15 NVMe decode, R17 prefetch).

## 6. Failures (I) — `experiments/failures/`
F001–F003 synthetic reasoning probes (shortcuts, then under-budget). F004 ops (git clean during runs). **F005 looping hurts at
toy scale** (re-injection init defect + regime).

## 7. Efficiency (G)
Axis definitions in `TARGET_ANALYSIS.md` §4. Measured: on 1 GB RAM, training uses −60% RSS via padding trim + arena limits; E4c
reaches the best bpb with the **lowest training FLOPs (7.3e12) and lowest inference FLOPs (1.49 MFLOP/tok)**, i.e. MoE is a free win at
this scale. Analytic 9B: C4 needs 4.9 GFLOP/token vs 16.1 for the dense C0 (3.3× fewer), at equal stored params.

## 8. Next-generation (J)
1. C5: 9B resident (hybrid + MoE) + large SSD Engram table, with prefetch.
2. Test-time compute through search/verification with C4 rather than through loops, unless E4g–i reverse F005.
3. Hyperconnections + identity-init re-injection if loops survive.
4. Retrieval/tools as the knowledge channel (closed-book parity with frontier models is not claimed).

## 9. Late result (added at end of session 1) — Engram at toy scale
| run | design | stored params | bpb | train FLOPs | inf MFLOP/tok |
|---|---|---|---|---|---|
| **E4j** | hybrid + MoE + **Engram** (no loop) | 0.825M | **1.4278** | 7.8e12 | 1.58 |
| E4z | hybrid dense, **2× params, 2.3× FLOPs** (8 layers) | 1.616M | 1.4394 | 1.8e13 | 3.61 |
| E4c | hybrid + MoE (no Engram) | 0.821M | 1.4870 | 7.3e12 | 1.49 |

- [E, toy, single seed] At iso-param with E4c, adding Engram (paid for by shrinking the experts) gives **−0.059 bpb**, the largest single effect of the session.
  E4j also **beats a model with 2× the params and 2.3× the inference FLOPs (E4z)**.
- [I] Strongly consistent with R17. At byte level, local n-gram statistics are a large part of the loss, so the size of this effect may shrink with a
  BPE tokenizer. **This must be replicated (3 seeds, BPE tokenizer, 10M stage) before it is believed.**
- Decision update: **C4 (hybrid + MoE + Engram) is the main line going into session 2.**
