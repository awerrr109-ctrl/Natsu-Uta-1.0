# Evidence log (L2/L3 reading notes)

Format: **[ID] source** — level — what it shows — controlled? — implication for Natsu — tags
Tags: [E] evidence, [I] inference, [H] hypothesis. Every item here was actually read by the
researcher at the stated level (abstract/summary/README = L2; full text sections = L3).

## Session 1 (2026-10-05)

### Looping / recurrent depth (H1.1, H11.x)
- **[R1] Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers", ICLR 2025, arXiv:2502.17416** — L2.
  [E] k-layer looped L times ≈ kL-layer model on synthetic reasoning (addition, p-hop induction, math); in LM, looped models competitive on *reasoning* downstream tasks; reasoning vs memorization dichotomy (loops help reasoning, hurt memorization/perplexity). Proposed looping-based regularization.
  → [I] at iso-PARAMETER, loops are the cheapest way to add reasoning depth; knowledge must come from elsewhere (memory/MoE/retrieval).
- **[R2] Schwethelm et al., "Iso-Depth Scaling Laws for Looped LMs", arXiv:2604.21106 + repo kschwethelm/looped-lm-scaling (L2 README)** — L2.
  [E] Fit L = E + A(N_once + r^φ N_rec)^-α + B D^-β over 116 runs; **φ = 0.46**. Truncated BPTT → φ 0.35–0.38 (worse loop despite lower val loss); hyperconnections → φ 0.65–0.66. At r=4, 410M looped ≈ 580M non-looped but costs the training compute of 1B non-looped.
  → [I] **Under a fixed 9B STORED-parameter budget the relevant comparison is iso-param, where looping is a pure gain (r^0.46 effective capacity multiplier on the looped share), paid for in compute.** With r=4 on a 75% looped share: N_eff ≈ 0.25N + 4^0.46·0.75N ≈ 1.68N (≈15B-equivalent loss for 9B params). [H] φ can be pushed higher by hyperconnections + per-loop routing.
  → Design rule: do NOT use truncated BPTT through loops (it hides a worse loop).
- **[R3] Zhu et al., "Ouro: Scaling Latent Reasoning via Looped LMs", arXiv:2510.25741** — L2.
  [E] 1.4B/2.6B looped models trained on 7.7T tokens match up to 12B SOTA models on many benchmarks; entropy-regularized learned depth allocation; advantage is from *knowledge manipulation*, not knowledge capacity.
  → consistent with R1. Caveat: 7.7T tokens — heavy overtraining; gains may partially reflect data scale. [unknown] iso-token comparison details not read at L3.
- **[R4] Lee et al., "Sparse Layers are Critical to Scaling Looped LMs", arXiv:2605.09165** — L2 (summary of HTML).
  [E] FineWeb-Edu 10B tokens, 37M–305M active. Dense looped scales *worse* than base; **Looped-MoE scales better than base** (iso-FLOP α 0.077 vs 0.076; OLMES 39.6 vs 38.7 vs looped-dense 37.4). Mechanism: **routing divergence** — 25–53% of tokens get disjoint experts across passes of the same layer. Early exits at loop boundaries far better (ppl 50.2 vs 55.4 base at 10% FLOP saved).
  → [I] Natsu's looped core must use MoE FFNs; we add an explicit learned per-loop router bias to *encourage* divergence (not in R4 — our delta; [H] untested).
  → caution: effect sizes are small (≈1 OLMES point) and only ≤305M active.
- **[R5] Loopie, "Loop the Loopies!", arXiv:2607.16051** — L2 (targeted Q&A over HTML).
  [E] MoE 20B-A2B and 6B-A0.6B, *layer-loop* R=2 (each layer applied twice consecutively), 3.5T tokens, beats compute-matched Qwen3-30B-A3B after ~600B tokens; no dynamic halting; R>2 avoided for throughput; inference-time scaling not studied.
  → [I] two looping topologies exist: block-loop (Huginn/Ouro/R4) and layer-loop (Loopie). Layer-loop keeps one KV per layer-iteration pair too. Open question H11.5.
- **[R6] LoopSpec arXiv:2609.17184; LoopCD arXiv:2610.02185** — L1/L2 (search snippets only).
  [E-weak] training-free self-speculative decoding from early loops exists (LoopSpec), and contrastive decoding using early loop states (LoopCD).
  → Our loop-speculative decoder is therefore **not novel by itself**; our delta is *training-time shallow↔deep self-distillation* to raise acceptance ([H], to be measured in E3c/E5).

### Memory layers (H1.4)
- **[R7] Berges et al., "Memory Layers at Scale", arXiv:2412.09764** — L2.
  [E] Product-key memory layers up to 128B memory params, 1T tokens, base up to 8B: outperform dense models with >2× compute and MoE matched for compute+params; gains largest on factual tasks.
  → [I] complements loops (R1/R3: loops add manipulation, not knowledge). Natsu stores knowledge in PKM-style memory in the (non-looped) coda/prelude → knowledge/FLOP decoupling. Caveat for 9B budget: memory params *count* against the 9B stored budget, so the trade is "memory params vs FFN params" not free capacity.

### Hybrid sequence mixing (H3.1, H3.2, H11.3)
- **[R8] Qwen3.5-9B model card (HF)** — L2.
  [E] 9B, 32 layers, layout 8×(3×(GDN→FFN) + 1×(Gated Attn→FFN)), d=4096, FFN 12288, vocab 248k, MTP, 262k ctx. Reported: MMLU-Pro 82.5, GPQA-D 81.7, LiveCodeBench v6 65.6, IFEval 91.5, HMMT Feb25 83.2, AA-LCR 63.0, BFCL-V4 66.1, TAU2 79.1.
  → **This is the direct 9B baseline to beat on the Pareto frontier**, and it already uses the 3:1 GDN:attention hybrid. A Natsu design that is "GDN hybrid" alone has **zero architectural novelty vs Qwen3.5**; differentiation must come from loops+MoE+memory+training.
- **[R9] "What Attention Recalls and Recurrence Controls in Hybrid LMs", arXiv:2609.04434** — L2.
  [E] In Qwen3.5/Falcon-H1, exact retrieval survives only through attention KV (64–98%) and collapses to 0 through recurrent state; recurrent state carries mode (language, persona). Failure modes of rec-only: DRM false recall, conjunction errors.
  → [I] Attention layers are non-negotiable for recall; recurrent state is a "prior". [H] For looping: loop the GDN+MoE layers (cheap state, no KV growth), keep attention layers mostly un-looped or share KV across loops (H11.6).
- **[R10] 3:1 hybrid convergence (Qwen3.5, Kimi Linear)** — L1 (news/blog snippets) — [E-weak] industry convergence on 3 linear : 1 full attention.

### Comparison target (Claude Opus 4.8) — see docs/TARGET_ANALYSIS.md
- **[R11] morphllm.com/claude-benchmarks (third-party aggregation, verified 2026-06-09)** — L2.
  [E, vendor-reported via 3rd party] Opus 4.8 (2026-05-28): SWE-bench Verified 88.6, SWE-bench Pro 69.2 (vendor scaffold), GPQA-D 93.6, OSWorld 83.4, Terminal-Bench 2.1 74.6, 1M ctx, 128k output, $5/$25 per MTok. Parameter count, architecture, training data: **unknown**.

## Contradictions / open tensions found
1. R2 (φ=0.46: looping is compute-inefficient) vs R3/R5 (looped models beat bigger models). Resolution [I]: R2 is iso-*compute*, R3/R5 report iso-*param* or wall-clock/compute-matched with MoE. Both can be true. Our constraint is params → looping favored, but we must report FLOPs honestly.
2. R4 says dense loops scale worse than base even iso-param? R4's iso-param claim is only for Looped-MoE. [unknown] whether dense-looped < base at iso-param in R4 → need L3 reading.
3. R1 "loops hurt memorization" vs R7 "memory layers help facts" → complementary, untested combination at scale (searched: "looped transformer combined with memory layers product key" → no direct hit in top-10 web results; RLT / MELT unrelated). **Candidate gap.**

### Low-star repos read at L2 (principle #4) — both changed the design
- **[R15] re133/sparse-memory-lm (★0, pushed 2026-10-05)** — L2 README (REPORT.md in German not yet read).
  [E] 21M Llama + product-key table (16.8M rows = 6.4B params, 3 layers share one table) on 500M Wikipedia tokens: val PPL 19.96 ≈ dense ~114M (iso-TOKENS). **At iso-training-TIME the 1M-row model was only 3% better than no table.** Bigger tables keep helping (~1.4× equiv. size per 4× rows). 4-bit table ≈ fp32 (19.98 vs 19.96). Decoding with the table on **NVMe at 114–138 tok/s**; prefill from NVMe collapses (1.5–6.5k tok/s). **Retrofitting a table onto frozen Qwen3.5-0.8B did NOT add facts** (fact cloze identical on seen and unseen articles, MMLU −2.7). Single seed; dense LR untuned (author notes this favours the table).
  → [I] Big consequence for the 9B budget: memory-table params are "cheap" params that **need not live in RAM/VRAM** for decoding. **[H-new H12.1] Count the 9B budget as dense+experts only, and put a large PKM table (e.g. 16–64M rows) on SSD**, or else keep a strict 9B total and size the table inside it (C3). Both variants are now tracked (C3 strict, C5 "9B resident + external table").
  → [I] the table must be trained from scratch (retrofit failed), so it is a pretraining-time decision.
- **[R16] zyberg2091/trm-halting-targets (★0)** — L2 README.
  [E] TRM-style recursive model on 4/8-digit addition, 3 seeds × 3 halting targets: learned halting **collapses to step-1 stopping** (first with soft targets), and once everything stops at step 1, **forced extra steps hurt accuracy in 155/161 checkpoints**. At 8 digits one supervision step beats the best five-step result in 6/6 graded-target comparisons.
  → **counter-evidence for Natsu's soft depth gate** (H2.2): a FLOP penalty plus a soft gate is exactly the setup prone to collapse. Mitigations to test: (a) random-R training keeps every depth supervised (our loop_sampling=uniform; R16 has no such thing), (b) gate penalty only in P2 anneal, (c) exact-match-style targets. Added to state as risk + experiment E5.

### L3-targeted reads (sections queried directly) — session 1
- **[R4 L3-partial]** Iso-FLOP (compute-optimal at 1e18) test-loss ranking: **MoE (non-looped) > Looped-MoE > Base > Looped**. Full-depth PPL: MoE 42.1, Looped-MoE(8×2) 44.5, Looped 45.4, Base 47.9 (note that Looped beats Base in this PPL table despite worse scaling, so the two sources are not fully consistent). OLMES: Looped-MoE 39.6 with 216M stored vs Base 38.7 with 246M.
  → [E] **Contradiction with the naive thesis: at iso-FLOP, a non-looped MoE beats a looped MoE.** The looped-MoE advantage is (a) fewer *stored* params and (b) better early-exit trade-offs. [I] For us, a 9B-stored budget means the fair comparison is C3 (looped-MoE, 8.9B) vs C4 (MoE no loop, 8.9B). C4 has *2.3× lower FLOPs/token* and may be as good or better per FLOP. **Looping must earn its FLOPs via test-time scaling (more loops → better on hard problems), which R4 did not measure.** E4b vs E4c is exactly this test at toy scale.
  → R4 cites MoEUT (Csordás et al.) as prior looped+MoE; criticises its iso-unique-param comparisons as giving 4× compute. Concurrent: Parcae (spectral-norm loop stabilisation), MoR. **No memory-layer + looping combination is cited** → candidate gap still open (now at L3-partial).
- **[R14 L3-partial] MELT arXiv:2605.07721**: single KV per layer shared across loops, updated by a gated momentum h_t = z⊙h_{t−1} + (1−z)⊙x_t, plus a W=50 per-loop sliding window. Converted from Ouro with 32K samples (320M tokens). Matches or slightly beats Ouro with 3.98× less KV (AIME24 50.4 vs 50.2). Limitations: fixed loop count at inference, no GQA, sequential KV updates during training.
  → [E] supports H11.6 (shared KV across loops costs about nothing). Our design is simpler (reuse the last loop's KV, or keep only loop-R KV) and compatible with variable R; MELT's gate is a stronger baseline to compare against. Status of H11.6 → "supported by prior work; our variant untested".

### Memory / distillation / RL (session 1, L2 + targeted Q&A)
- **[R17] Engram, Cheng et al. (DeepSeek) arXiv:2601.07372 + repo deepseek-ai/Engram** — L2 + targeted L3 Q&A.
  [E] Hashed suffix n-gram (N≤3, 8 hash heads) embedding tables. The hidden state gates the retrieved memory (query = h, key/value = memory), followed by a depthwise causal conv (k=4). Inserted at layers 2 and 15 of 27B. **U-shaped sparsity-allocation law: ~20–25% of the sparse budget in memory, 75–80% in MoE.** Engram-27B beats a strictly iso-param, iso-FLOP MoE: MMLU +3.4, BBH +5.0, ARC-C +3.7, HumanEval +3.0, MATH +2.4, MQ-NIAH 84.2→97.0. 262B tokens. Mechanism: relieves early layers of static reconstruction ("effectively deeper"). Deterministic addresses allow prefetch from host memory. Failure mode: suppressing Engram at inference collapses factual tasks (TriviaQA → 29%).
  → [I] **Engram is a better fit for Natsu than PKM**: (1) iso-param *and* iso-FLOP evidence at 27B (PKM evidence R7 is similar, but R15 shows PKM wins little at iso-time); (2) addresses are known from token ids → SSD/host offload without the PKM prefill collapse seen in R15; (3) "effectively deeper" is complementary to loops (loops add depth; Engram frees depth). **Design change: C3 knowledge memory switches PKM → Engram after the prelude** (implemented `model.Engram`, tested cache-exact and prefetchable). Allocation rule adopted: 20–25% of the sparse (MoE+memory) budget.
  → [H-new H12.2] Engram + looped-MoE: the loop core never re-reads the memory (it is before the loop), so memory cost is paid once while its benefit ("freed early layers") is multiplied across R loops. Untested anywhere we found → E4f.
- **[R18] Busbridge et al., "Distillation Scaling Laws", ICML 2025, arXiv:2502.08606** — L2.
  [E] With an existing teacher (our case), distillation beats supervised learning up to a compute level that grows with student size. If the teacher must be trained just for one student, supervised learning is preferable. Author summary: choose a teacher only slightly more capable than the student's target rather than the largest one (capacity gap).
  → [I] For 9B: distill from an existing open strong model (e.g. a ≥30B MoE). The benefit fades at very large token budgets, so KD is front-loaded (P1), not used through P2. [unknown] crossover token count for 9B → needs L3.
- **[R19] Yue et al., "Does RL Really Incentivize Reasoning Capacity…", NeurIPS 2025, arXiv:2504.13837** — L2 (abstract-level).
  [E] RLVR models beat their base at small k but **base models surpass RL models at large pass@k**: RL narrows the reasoning boundary rather than expanding it. A 2026 follow-up ("Curriculum RL can incentivize…", arXiv:2606.22317, L1) claims curriculum RL does expand it → **contested**.
  → [I] RL is not the lever for *new* capability at 9B. Distillation and pretraining data are. RL is kept for elicitation and calibration, and is evaluated with pass@1 and pass@k.

## Session 2

### Looping recipe (triggered by E4g reversal of F005)
- **[R20] Parcae, Prairie et al. arXiv:2604.12946 (UCSD/Together)** — L3-partial.
  [E] Prior looped models suffer residual-state explosion and loss spikes. Fix: the residual stream as a discretised LTI system,
  h_{t+1} = Ā h_t + B̄ e + R̄(h_t, e), with A = Diag(−exp(logA)) and ZOH Ā = exp(ΔA) → spectral radius < 1. Prelude output normalised.
  T ~ Poisson(μ_rec), sampled per sequence. **770M Parcae ≈ 1.3B Transformer (CORE)**; at 1.3B, +2.99 CORE. Limitation: latency grows with μ_rec.
  → implemented as `reinject_mode="lti"` (tested cache-exact) → E4o. Parcae samples depth (Poisson) and still beats parameter-matched
  Transformers, so our uniform-R failure (F005) is more likely due to toy scale and the uniform distribution plus a bad injection than to sampling per se → E4n tests Poisson(R).
- **[R21] TaH2 "Improving Test-Time Scaling with Adaptive Looped Transformers" arXiv:2609.35748** — L2.
  [E] **Existing looped transformers have steeper accuracy/compute slopes but underperform non-looped baselines at matched compute**,
  because many tokens do not benefit from extra iterations. An iteration decider trained by lookahead depth supervision (online labels: "does
  one more iteration improve this token?") gives slope 2.74 vs 1.79, +3.4 points over the baseline peak at matched compute, and a gain growing with max depth
  (+2.8 at depth 2 → +3.9 at depth 8).
  → [I] This is the missing piece of our depth-gate design (E4d gate skipped only 5%). **Replace the FLOP-penalty gate with lookahead
  supervision**: target_t = 1[CE_t(r+1) < CE_t(r) − margin]. Implementable cheaply because R loops are already computed in training.
  This is also the fix for R16's halting collapse: targets come from measured benefit, not a cost penalty.
- **[R22] "Stability and Generalization in Looped Transformers" arXiv:2604.15259; Fixed-Point Reasoners arXiv:2606.18206; "Stabilizing
  Extrapolation…" arXiv:2606.29983** — L1 (titles/snippets). Extrapolation beyond trained depth needs fixed-point-style training. Our E4g
  degrades at R=6, which is consistent. Queued for L2.

### Token-indexed memory: robustness across scale/tokenizer (H13.3, H13.6)
- **[R23] Over-Tokenized Transformer (Huang et al., ICML 2025, arXiv:2501.16975)** — L3-partial.
  [E] Input-side hierarchical n-gram vocabulary (summed 1..n-gram embeddings). **Loss is log-linear in input vocab: −0.015 per 4× (L = 2.6754 − 0.0256·log10 m)**.
  OE-12.8M 400M ≈ 1B baseline (2.5×). **The gain is about constant from OLMoE-1.3B to 7B** even as embedding share drops. Scaling the *output* vocab hurts small (underfitting) models.
  → [I] Strong cross-scale evidence that token-indexed memory is not a toy artefact. Supports C4 at 9B. Also a **design rule: grow the input side
  (Engram/OE), keep the output vocab moderate** for 9B.
- **[R24] X-gram (arXiv:2604.21724)** — L2.
  [E] At 0.73B, X-gram 48.5 vs Engram 47.2 vs MoRT 45.8 vs Retoken 45.0. At 1.15B, +2.3–3.2 over Engram, reaching baseline quality with 57% of the data.
  **Failure modes of naive n-gram memory: (1) Zipfian under-training: most long-tail rows stay cold; (2) parallel fixed slots collapse into
  redundant subspaces; (3) injecting into Q/K is fragile, while value-stream/residual injection is better.**
  → [H] Engram v2 in Natsu: frequency-aware hashing (reserve dedicated rows for frequent n-grams, share buckets for the tail) +
  per-head distinct conv kernels. Planned E7 (toy can test (2) via head count, (1) via slot count).
- Refute check H13.6: no paper found showing the n-gram memory gain *vanishing* with larger tokenizer vocab. R23 shows the opposite.
  [unknown] at byte level our effect may be inflated → B1/B2 test this directly.

### Test-time scaling (P4, principle #13)
- **[R27] "Can 1B LLM Surpass 405B LLM? Compute-optimal TTS" arXiv:2502.06703 (ICLR'25 track)** — L3-partial.
  [E] Optimal TTS depends on policy size and difficulty: <7B → search (beam/DVTS) on hard problems, BoN on easy ones; 72B → BoN everywhere.
  3B + TTS > 405B on MATH-500 and AIME24; 7B-distill + TTS > o1 / R1 on MATH-500 and AIME24, at **100–1000× fewer FLOPs** than 405B.
  **Failures**: PRMs do not generalize across policies (OOD → worse than majority vote); on the hardest set (AIME24), TTS < distillation from strong reasoners;
  gains shrink as the policy gets stronger; math-only evaluation.
  → [I] For the 9B target, the inference-time efficiency multiplier vs frontier models can plausibly reach 100–1000× **on verifiable domains (math/code)**,
  but only with a policy-matched verifier (train our own PRM/ORM on our own samples) and only when the base reasoning comes from distillation.
  Verifiers from executable feedback (code tests, math checkers) avoid PRM OOD issues → tool-integrated verification (L1: Kang et al.).

### Own results, session 2 (BPE confound check)
- **B1 vs B2 (BPE-4k, 0.79M non-embedding, 819k tokens ≈ 3.2MB text)**: MoE no-loop 1.1449 bpb → +Engram (iso non-embed) **1.1312 (−0.014)**.
  Byte level was −0.059 (E4j vs E4c). [E] Engram's gain **survives the BPE tokenizer but shrinks ~4×**, consistent with the byte-level
  inflation hypothesis (H13.6) and with R23 (gain is log-linear in effective input vocab; BPE already captures part of local n-gram statistics).
  Note: BPE runs see ~2× more bytes of context and ~same bytes trained (819k tok × 3.96 ≈ 3.2MB vs 1.6MB), so the absolute bpb is not comparable across tokenizers; only within-tokenizer deltas are.

### Loop stability / adaptive recursion (H13.2, H13.5)
- **[R28] Stability and Generalization in Looped Transformers (Labovich, arXiv:2604.15259)** — L2.
  [E] Theory + chess/sudoku/prefix-sum experiments: **without recall (= input re-injection) a looped network has countable fixed points and cannot be strongly input-dependent**. Recall + *outer normalization* gives reachable, input-smooth fixed points and stable backprop. Internal recall placement is competitive and better on sudoku.
  → [I] **Tension with our E4g** (no re-injection was best at toy LM). Resolution: E4g uses R=3 (not a fixed-point regime), and the prelude output is already in the residual stream. R28's claim is about extrapolation to many iterations. Prediction: E4g fails at large R (consistent: R=6 1.552) while LTI/recall variants (E4o) should degrade less at R=6. This is a pre-registered test.
- **[R29] Mixture-of-Recursions (arXiv:2507.10524)** — L2.
  [E] Token-level routers over recursion depth; attention only among tokens still active; KV cached only for active tokens; a KV-sharing variant reuses
  first-recursion KV (= our `loop_kv=shared_first`). New Pareto frontier at 135M–1.7B at equal training FLOPs. [unknown] failure modes (not in abstract).
  → [I] Independent support (with R21) that **token-adaptive depth is the condition for loops to win at matched compute**, and that shared-first KV is viable.
- **B3 (BPE, loop R=3 fixed, iso non-embed with B1)**: R1 1.2204 / R2 1.1504 / **R3 1.1392** / R4 1.1452 / R6 1.1750.
  vs B1 1.1449: loop −0.006 bpb at 1.43× inf FLOPs. **Under BPE, Engram (−0.014 at 1.04× FLOPs) beats looping (−0.006 at 1.43×) by ~3.4× in gain-per-FLOP terms.**
  Both effects shrink from byte→BPE (Engram 0.059→0.014, loop 0.030→0.006). The ranking is preserved. Single seed; BPE deltas ≈ 2–5× typical seed noise (seed runs pending).

### Data axis (P5): quality vs quantity vs diversity at long horizons (principle #15)
- **[R30] Nemotron-CC (arXiv:2412.02595)** — L3-partial.
  [E] FineWeb-Edu/DCLM remove ~90% of data and are ~80% near-duplicates, which hits diminishing returns after ~4 epochs. Classifier *ensembling* raises the HQ share 9→25%;
  rephrasing low-quality data gives +1.5 avg; dropping heuristic filters on HQ data gives +18% yield. At 8B/1T, HQ subset MMLU 52.8 vs DCLM 47.2 / FW-Edu 51.2;
  **8B at 15T tokens (7.2T Nemotron-CC) MMLU 70.3 vs Llama-3.1-8B 65.3.** → aggressive filtering is wrong for long horizons.
- **[R31] "The data-quality illusion" (Saada et al. 2025)** — L2.
  [E] Classifier-based filtering improves downstream tasks but can make LM loss on the HQ set *worse* (U-shaped in selection fraction). It "removes the bad"
  rather than imitating the good, and acts mainly as alignment to benchmark-like style.
- **[R32] BETR (arXiv:2507.12466)** — L2.
  [E] Benchmark-targeted selection gives 2.1× compute multiplier over DCLM (4.7× over unfiltered). **Optimal kept fraction F_opt ∝ C^0.25: 3% at 1e20 FLOPs → 30% at 1e23.**
  Goodhart: optimising for the core suite *lowers* held-out non-core tasks.
- → [I] **Data spec for Natsu-9B (≈ 6.4e22–1e23 training FLOPs at 2.6B active × ~4–6T tokens)**: keep ~25–30% of the web (R32 law),
  ensemble classifiers (R30), rephrase the rest (R30, R26 megadocs 1.8×), distill (R25). Evaluate only on a **held-out benchmark set never used for selection**
  (R31, R32 Goodhart). The toy-stage filter question is moot (TinyStories). Recorded in TRAINING_SPEC v0.2.

### Reasoning vs memorization; objective (P2/P4, principles #13, #15) — session 2
- **[R33] Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers" (ICLR 2025, arXiv:2502.17416)** — L3-partial (HTML §3–4).
  [E] At 1B LM scale, k⊗L looped models have **worse perplexity and worse closed-book QA (memorization)** than the iso-FLOP kL⊗1 baseline, but close most of the
  iso-param→iso-FLOP gap on open-book QA / math word problems, and **beat the 24-layer baseline on reasoning primitives with 24/k× fewer params**.
  Downstream accuracy scales ~log(effective depth). A cosine-similarity regulariser between successive layer blocks (λ=10, cos≥0.98) reproduces the inductive bias
  without weight sharing.
  → [I] **Reframes F007**: our substitution result is measured on bpb, which R33 says is exactly the metric where loops look worst. Engram (memorization-type lookup)
  and loops (reasoning-type depth) can be substitutes on bpb but complements on reasoning. Prediction (pre-registered): on E8 addition and E6 varchain,
  gain(loop | Engram) ≥ 0.5·gain(loop | no Engram), i.e. not strongly sub-additive. Tested by the 2×2 cells E6a/E6b/E6c/E6d and E8a/E8b/E8c/E8d (queue s2e).
  Also: the λ-cosine regulariser is a cheap 9B option ("loop bias without loop FLOPs") → logged as H13.7.
- **[R34] Bachmann & Nagarajan, "The pitfalls of next-token prediction" (ICML 2024, arXiv:2403.06963)** — L2.
  [E] Teacher-forced NTP can fail to *learn* the correct predictor on a simple path-star planning task (Transformer and Mamba both fail ≈ chance) via the
  "Clever Hans" shortcut. Teacherless multi-token training (dummy-token inputs) fixes it in preliminary experiments.
  → [I] Supports keeping MTP (impl) and argues for a teacherless/MTP fraction in P1. Our gen_chain v1/v2 failures (F001/F002: root shortcut) look like
  the same Clever-Hans mechanism at toy scale. [unknown] whether it matters for natural-text bpb; tested only on synthetic graphs.

### Own results, session 2 (E4n, E5 distillation)
- **E4n (Poisson-R loop sampling, mean 3)**: R1 1.5419 / R3 **1.5029** / R6 1.5175. Flat across depth (robust), but best is 1.5029, worse than E4c no-loop 1.4870.
  Ranking of loop recipes at toy scale: E4g fixed (1.4574) < E4p exit-trained (1.4607) < E4m curriculum (1.4701) < E4n Poisson (1.5029) < E4h uniform (1.5291).
  [E own] Stochastic depth training buys robustness at a cost proportional to how often shallow R is sampled. Exit training (E4p) is the only recipe that gets both.
- **E5a/E5b (DPT logit distillation from E4z, w=0.5, T=1, 15% lowest-entropy tokens hard-label only)**, identical data/steps to E4c/E4j:
  | student | no KD | +KD from E4z | Δ |
  |---|---|---|---|
  | MoE (0.82M) | E4c 1.4870 | E5a **1.4522** | −0.035 |
  | MoE+Engram (0.83M) | E4j 1.4278 | E5b **1.4063** | −0.022 |
  Teacher E4z (1.62M dense, 2× params) = 1.4394. **Both students beat the teacher**; E5b beats it by 0.033.
  [I] At 1.6M tokens per run, the student is data-starved, not capacity-starved. Soft targets add information per token, as the "2× data-equivalent" claim
  in R25 predicts. Student > teacher is consistent with KD-as-regularisation (born-again networks). Engram + KD are **partly additive** (−0.059 then −0.022 more),
  unlike Engram + loop (F007). Cost: the teacher forward adds ~0.8× student training FLOPs (3.61e6 vs 3·1.49e6 per token), so KD's training-FLOP multiplier
  ≈1.8× for −0.035, and the teacher's own training cost is extra. At inference, KD is free; this is the axis that matters for the 9B deployment target.
  Single seed; seed runs are pending, so deltas ≥0.02 are likely real (B-series noise estimate ~0.003–0.007, to be confirmed).
  **New Pareto best (toy, inference FLOPs): E5b 1.4063 at 1.58e6 FLOP/token.**
- **[R35] Busbridge et al., "Distillation Scaling Laws" (ICML 2025, arXiv:2502.08606)** — L3-partial (section headers + §5 claims). **Refutation read for E5.**
  [E] Student CE depends on the teacher only through the teacher's CE L_T (a power law with a capacity-gap regime). **Distillation beats supervised learning
  only up to a student-compute level that grows with student size; with enough student tokens, supervised ≥ distillation.** If the teacher must be trained
  for one student, supervised is generally preferable. Students can beat teachers (weak-to-strong) in some isoFLOP cases.
  → [I] Our E5 regime (0.8M params, 1.6M tokens ≈ 2 tokens/param, heavily *under*-trained) is exactly where KD should win most. **The −0.035 does NOT transfer
  to Natsu-9B at 4–6T tokens (~500 tokens/param)**; R35 predicts the gain shrinks toward zero there. The 9B KD plan is therefore justified only with an
  **existing** teacher (frontier/open-weight, teacher cost sunk) and mainly for post-training and reasoning traces (R25 pass@k), not as a pretraining
  multiplier. TRAINING_SPEC P1 KD share is kept, but its claimed efficiency multiplier is downgraded to "regime-dependent, ≤1.x at 9B" [I].
  Control E5c (born-again, same-size teacher E4c) is queued: if E5c ≈ E5a, the toy gain is regularisation, not capacity transfer.

### Own results, session 2 (E8 addition probe, depth-hungry task)
- **E8a (no loop) vs E8b (loop R=3 fixed), 5-digit addition, answer exact-match acc (teacher-forced, 256 val problems)**:
  E8a acc **0.297** (answer loss 0.603). E8b R1 0.094 / R2 0.270 / **R3 0.285** / R4 0.285 (answer loss 1.037 at R3). Both 0.0 on 6-digit (no length generalisation).
  [E own] **Looping did not help addition at iso-param**, despite this being the depth-hungry task R33 predicts loops excel on. Answer loss is *worse* with loops
  (1.04 vs 0.60), and E8b took 8.2× the wall time (4587 s vs 557 s; partly CPU contention with my foreground work).
  [I] Contradiction with R33 (loops ≈ iso-FLOP deep model on addition). Candidate explanations: (a) our 2-layer core × 3 = 6 effective layers is enough for
  the 1–5-digit regime, so neither needs more depth; acc ~0.29 for both suggests a shared bottleneck (data/steps), not depth; (b) R33 trains far longer to
  convergence; (c) carry propagation needs position-wise algorithms (index hints/abacus embeddings), not depth. Test: E8e (6-core unlooped, iso-FLOP) separates
  "depth helps" from "loop helps"; tts_eval (greedy vs maj@N) runs after the queue (scripts/post_s2.sh).
