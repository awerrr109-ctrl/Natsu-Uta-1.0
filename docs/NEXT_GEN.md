# J. Next-Generation Architecture (v0.1, session 2)

This document proposes what comes after C4. Every item states the evidence that motivates it, the bottleneck it addresses, and the cheapest falsifying experiment.
Tags: [E] measured/cited, [I] inferred, [H] hypothesis.

## Starting point (C4, current main line)
3:1 GDN:attention hybrid + fine-grained MoE (48 experts) + Engram table (ρ = 77%, inside R17's optimum), no trained loop. Analytic 9B cost: 2.57B active, 5.5 GFLOP/token, 0.40 GB KV @32k.

## Bottlenecks that remain (ranked by expected impact on 9B practical quality)
| # | bottleneck | evidence | why parameters alone can't fix it |
|---|---|---|---|
| B1 | closed-book knowledge at 9B | TARGET_ANALYSIS §2: MMLU-Pro/SimpleQA are the hardest axes at 9B | knowledge ∝ stored bits; 9B stored is fixed |
| B2 | multi-step reasoning on verifiable tasks | GPQA gap −11.9 vs target | depth/compute per answer, not params |
| B3 | quality per training token | R35: KD gain fades at 9B token budgets; R32: filtering law | data, not architecture |
| B4 | in-context retrieval during early training | E6 (weak), R37/R38 | an optimisation-dynamics issue |

## Proposals

### G1. Two-tier memory: resident C4 + off-device editable Engram (addresses B1)
- **What**: keep the 9B *resident* budget, but let the Engram table grow beyond it on host/SSD (R17: 100B-param table offloaded with negligible overhead, because addresses depend only on token ids).
  The table is also an **editable, removable knowledge store** (R39 ENGRAFT: 0.84 exact answer on 100 invented facts, collateral KL 0.013).
- **Why 9B-specific**: the "9B" constraint is about resident/active compute. A prefetchable table adds knowledge without adding FLOPs or VRAM.
- **Known failure modes**: phrasing locality (R39: facts only fire on the same n-grams); composition fails (4/83).
- **Cheapest test**: E9 (queued); then H14.7 (factorised table) at toy iso-param. Table-only vs full-FT vs FFN-only fact writing, measured on train-phrasing acc, held-out phrasing acc and collateral bpb.
- **Prior art check (R43–R46)**: context-dependent read-out exists (MoME, FactorEngram); factorised tables (FactorEngram, TN-gram) cut table size several-fold at equal quality. **H14.7**: a factorised table in C4 could free ~1B of the 1.6B table params for experts.
- **Next-gen fix [H, partly novel]**: a *content-addressed* path (learned keys) next to the exact n-gram path, evaluated on paraphrase robustness of written facts. That evaluation is absent in R43–R46. Add a second Engram head addressed by a learned product-key over a pooled context embedding (PKM-style, R7/R15) next to the exact-n-gram path. Exact path = cheap, prefetchable; semantic path = phrasing-robust. Falsifier: held-out-phrasing acc in E9 does not improve.

### G1b. Three-store knowledge partition (from R55)
- weights: skills and frequent knowledge; Engram table: frequent local patterns, editable overlay; retrieval datastore: long-tail facts.
- R55: ~1 retrieval token per parameter captures a median 91% of the retrieval gain, so a ~9B-token datastore (~18 GB) suffices for a 9B model.
- Open question: which documents go to pretraining vs datastore (R55 suggests explicit partitioning). Test at the 50M ladder: pretrain on 50% and retrieve over the other 50%, vs pretrain on 100%.

### G2. Verifier-coupled test-time compute instead of trained loops (addresses B2)
- **Evidence**:
  - Loops at toy scale: −0.013 bpb (t ≈ 2.0, 3 seeds) at 1.73× FLOPs; substitutes with Engram (F007, F010); no gain on addition (E8b).
  - R27: compute-optimal TTS with a policy-matched verifier gets 100–1000× FLOP efficiency vs much larger models on math.
- **What**: C4 + sampling (maj@N / best-of-N) + **executable verifiers** (code tests, math checkers) and a small PRM trained on C4's own samples (R27: PRMs do not transfer across policies).
- **Measured so far**: implemented (`generate.sample_n/majority_vote/best_of_n/tts_accounting`, `tts_eval.py`); curves pending (post_s2).
- **Loop role**: a cheap draft for self-speculative decoding (implemented, exact). **Re-opened by R47/R48**: looped MoE with per-loop routers and residual scaling shows iso-FLOP gains at 0.7–1.7B and ~2× total-param efficiency on reasoning at trillion-token scale. Our toy negatives may be scale- and recipe-specific. The 50M+ ladder must include C4-loop-LOOM vs C4 at iso-FLOP; the G2 vs loop choice is made there, not at toy scale.

### G3. Delayed / staged memory training (addresses B4)
- **H14.1 / N5**: ramp the Engram branch in after induction circuits form (`train.engram_delay`). Falsifiers are pre-registered (E6Lb/E6Lc/E4u).
- If confirmed, the general principle is: **introduce lookup shortcuts only after the circuits they could pre-empt have formed.** This applies to any memory/retrieval module.

### G4. Teacher-sunk-cost distillation plus data rephrasing (addresses B3)
- KD helps in the under-trained regime (E5a −0.035, E5b −0.022 on top of Engram), and its value at 9B is bounded by R35. Use it only from an existing strong teacher, concentrated in mid/post-training and reasoning traces.
- Data: keep ~25–30% of web at 9B compute (R32), rephrase lower buckets (R26, R30), held-out Goodhart guard (R31).

### G5. Stored-vs-active rebalancing (re-opened by R42)
- R42: iso-total MoE beats dense given more tokens, with optimal E depending on the memory/compute budget. C4's E=48 fine-grained is outside the validated range.
- **X1 (ladder stage 1B)**: C4 vs C4 with half the experts and a larger active set, at iso-stored params and iso-training-FLOPs.

## What is explicitly NOT proposed (and why)
- Trained deep loops as the main capability lever: the toy evidence is weak/negative (F005 updates 1–4, F007, F010). R33's reasoning gains are not reproduced here.
- Aggressive web filtering at 9B: R32's law says it is wrong for 1e23-FLOP budgets.
- Forward-only / non-backprop pretraining: gradient variance grows with dimension (taxonomy H6.4 refute list). It stays a fine-tuning-only option.


## G1c. Retrieval-distilled Engram (H15.1, session 2)
- **Evidence**: on toy scale, kNN gives −0.13/−0.22 bpb at 1/4 tok/param (inflated by TinyStories redundancy) and stays partly additive with Engram (E4j+kNN 1.249).
  But R60/R61 report that kNN-LM perplexity gains fail to transfer to generation/QA and cost 2.17× latency plus a datastore (~244 B/token fp16 at d=128).
- **Proposal**: build kNN-mixed targets offline from a frozen checkpoint, then DPT-distil them into the student, with the Engram table lr ×5 (F013 path).
  Deployment needs no datastore and no latency.
- **Difference from MemDec (R60)**: the target is the base model's own token-indexed memory, not a separate decoder. That gives zero extra inference FLOPs, and the memory stays editable (G1).
- **Falsifier**: the student's bpb gain is < 30% of the online kNN gain on the same valid slice → the table cannot absorb context-indexed knowledge; keep the datastore (G1b) instead.
- **Prereqs**: JL/int8 store result (post_s2b); self-match exclusion; target cache format (top-16 value u16 + weight f16 = 64 B/token).
