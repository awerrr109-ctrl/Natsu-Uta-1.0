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
  Teacher E4z (1.62M dense, 2× params) = 1.4394. **Correction: only E5b beats the teacher** (by 0.033); E5a (1.4522) is still 0.013 worse than the teacher, though it closes 73% of the E4c→E4z gap at half the params.
  [I] At 1.6M tokens per run, the student is data-starved, not capacity-starved. Soft targets add information per token, as the "2× data-equivalent" claim
  in R25 predicts. E5b > teacher is explained by Engram + KD stacking (E4j alone is already 1.4278 < teacher 1.4394), so it is not evidence of weak-to-strong KD by itself. Engram + KD are **partly additive** (−0.059 then −0.022 more),
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
  (1.04 vs 0.60), at 1.71× inference FLOPs/token and 2.05× lower training throughput (1110 vs 2278 tok/s). Wall-clock (4587 s vs 557 s) is NOT usable: no foreground job ran then,
  so the gap is most likely sandbox suspension between turns. Throughput (tok/s) from the logs is used as the time metric from now on.
  [I] Contradiction with R33 (loops ≈ iso-FLOP deep model on addition). Candidate explanations: (a) our 2-layer core × 3 = 6 effective layers is enough for
  the 1–5-digit regime, so neither needs more depth; acc ~0.29 for both suggests a shared bottleneck (data/steps), not depth; (b) R33 trains far longer to
  convergence; (c) carry propagation needs position-wise algorithms (index hints/abacus embeddings), not depth. Test: E8e (6-core unlooped, iso-FLOP) separates
  "depth helps" from "loop helps"; tts_eval (greedy vs maj@N) runs after the queue (scripts/post_s2.sh).
- **[R36] McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (Abacus, NeurIPS 2024, arXiv:2405.17399)** — L2.
  [E] Transformer arithmetic failure is largely positional: the model can't track each digit's position within a number. Adding an embedding of the digit's
  position relative to the start of the number fixes it, and **only then do input injection and recurrent (looped) layers add further gains**.
  20-digit training reaches 99% on 100-digit addition.
  → [I] **Resolves the E8 contradiction with R33 as a testable hypothesis**: our loop showed no gain because the bottleneck is digit alignment, not depth.
  Implemented `digit_pos` (+ random offset for length generalisation; cache-exact test added). E8g (no loop) vs E8h (loop) are queued (s2g).
  Prediction: both jump well above 0.30 acc; E8h > E8g on d6 (length generalisation). If E8h ≤ E8g, loops lack value even with positions fixed at this scale.

### Own results, session 2 — seed noise (s1 replicates; s2 replicates still queued)
| config | seed 0 | seed 1 | |Δ| |
|---|---|---|---|
| E4c MoE no-loop | 1.4870 | 1.4735 | 0.0135 |
| E4j + Engram | 1.4278 | 1.4246 | 0.0032 |
| E4g loop R3 | 1.4574 | 1.4691 | 0.0117 |
- [E own] Seed spread (data order + init) is **~0.003–0.014 bpb**, larger than the 0.003–0.007 I assumed earlier. Consequences (2-seed means):
  - Engram: E4c 1.4803 → E4j 1.4262, **−0.054** (≫ noise) — robust.
  - loop: E4c 1.4803 → E4g 1.4633, **−0.017** (≈1.3–1.5× noise) — weak; the earlier −0.030 was partly a lucky seed pairing.
  - E4p vs E4g (0.003), E4e2 vs E4g (0.005): **inside noise → "no difference"**, not rankings.
  - E5a −0.035 vs E4c seed 0, −0.021 vs the 2-seed mean: probably real, but needs E5a seeds. E5b −0.022 vs E4j (Engram seed spread is small): likely real.
  - BPE deltas (Engram −0.014, loop −0.006) need replicates before being called; loop-under-BPE is within noise.
- **E4i (fixed R3 + identity-init concat reinjection)**: R3 1.4843, R1 3.08, R6 2.54. **Reinjection makes depth extrapolation explode** at fixed R
  (cf. R28/R20: unconstrained injection is unstable; LTI (E4o) is the pre-registered fix, still pending). E4i is worse than E4g (1.4574/1.4691) → at fixed R, no reinjection is best.
- **E4e2 (shared-first KV across loops)**: R3 1.4620 vs E4g 1.4574/1.4691 → **no measurable quality loss (inside noise), and the KV cache of core attention
  layers shrinks by R× (3×)**. Supports R14/R29. Adopted as the default for loop variants in C4-loop.
- **E6 varchain (dependency-chain probe)**: answer loss E6a no-loop **0.542** < E6b +Engram 0.917 < E6c loop R3 0.914 (R2 0.957). Sequence EM ≈ 0 for all.
  [E own] On this in-context lookup+arithmetic probe, **both Engram and the loop hurt** at iso-param (they take params from the FFN/MoE: expert_mult 0.296→0.17
  for Engram; loop shares weights). [I] varchain needs in-context retrieval (attention), not n-gram statistics or depth; Engram rows are useless
  for random variable values, and the parameters taken from experts cost capacity. This is the first task where Engram is negative: **a real failure mode for C4.**
  It matches R17's finding that Engram helps knowledge and reasoning but its value depends on recurring local patterns.
  **Caveat (curve check)**: train loss E6a 1.097@800 → 0.476@1199 is a sharp late drop (a phase transition, likely retrieval circuit formation); E6b was *ahead*
  at step 800 (1.020) and then plateaued (0.868); E6c 1.291@800 → 0.885. The ranking is therefore a **transition-timing** effect in one seed and a 1200-step
  budget, not a capacity verdict. Downgraded to [weak E]. Needs longer runs or seeds before "Engram hurts retrieval tasks" can be claimed.
- **Pareto with honest KD accounting (scripts/pareto.py v2)**: charging teacher forward + teacher training, E5b costs **3.14e13 train FLOPs vs E4j 7.78e12 (4.0×)**
  for −0.020 bpb vs the E4j 2-seed mean. At **iso-train-FLOP**, E4z alone (1.77e13, 1.4394) is worse than E4j, so the KD route is only Pareto on the
  *inference* axis. With a sunk-cost teacher (the 9B plan), the marginal training cost is the teacher forward only (~1.8×). Consistent with R35.

### H14.1 — does token-indexed memory delay in-context retrieval? (from E6 anomaly)
- **[R37] Edelman et al., "The Evolution of Statistical Induction Heads" (arXiv:2402.11004)** — L2. [E] Models pass uniform → in-context unigram → *sudden*
  transition to the bigram (induction) solution; **the presence of the simpler unigram solution may delay formation of the final solution.**
- **[R38] Bietti et al., "Birth of a Transformer: A Memory Viewpoint" (arXiv:2306.00802)** — L2. [E] Global bigrams are learned fast (weight matrices act as
  associative memories); the induction head for in-context bigrams develops slowly; this depends on data-distribution properties.
- → [I] Mechanism for E6: Engram is an explicit *global n-gram* associative memory. It makes the "global statistics" solution cheaper and faster, which
  plausibly reduces gradient pressure on the induction circuit (R37 "simple solution delays complex one"). E6b was ahead at step 800 and then plateaued
  while E6a transitioned. This is a **known pattern in a new place**. I found no paper testing it for hashed n-gram memories (Engram/OE/X-gram report
  NIAH gains at scale; R17 NIAH 84→97). That is a counter-signal: at scale, retrieval improves.
- [H] **H14.1 test + fix, "delayed memory"**: ramp the Engram branch in only after the retrieval circuit forms (`train.engram_delay=[0.4,0.6]`).
  Pre-registered: (a) E6Lb (Engram) < E6La on varchain at 2400 steps confirms the harm; (b) E6Lc (delayed) ≈ E6La confirms the mechanism;
  (c) E4u (delayed, LM) keeps ≥ 70% of Engram's −0.054 bpb, so the fix is cheap. If (a) fails, E6 was a timing artefact and H14.1 is dropped.
- **Refutation check for H14.1 (R17 re-read, L3-partial on this point)**: Engram-27B reports **Variable Tracking 77.0 → 89.0** and MQ-NIAH 84.2 → 97.0 (RULER),
  where variable tracking is the closest public analogue of our varchain probe. [E] At 27B and long training, n-gram memory *improves* in-context variable tracking.
  → [I] If E6L confirms harm at toy scale, the two results are reconciled by **training stage**: early on, memory delays the induction circuit (R37/R38);
  after the circuit forms, memory frees attention for global context (R17). That predicts a delayed-memory schedule (H14.1 fix) gets both. It is also
  consistent with the toy runs being stopped right around the transition. Absent E6L harm, H14.1 is dropped and the varchain result is a timing artefact.
- **3-seed result (s2 replicates)**: E4c 1.4870/1.4735/1.4772 (mean 1.4792, sd 0.0070); E4j 1.4278/1.4246/1.4261 (mean 1.4262, sd 0.0016).
  **Engram −0.053 bpb, Welch t ≈ 12.8 (n=3 each).** The strongest result of the project at toy scale. Engram also reduces seed variance by 4.4× (sd 0.0070 → 0.0016) [E own].

### External/editable memory (P7) — low-star repo finding (principle #4)
- **[R39] fulvian/engraft-ngram (51★, pushed 2026-09-30)** — L2 (README with result files). "ENGRAFT": write facts into an Engram-style n-gram table
  of a deployed model (Qwen3.8-Flash-Next, 320M-row table) by training **only table rows**, shipped as a removable overlay.
  [E, their files] 100 invented facts: exact answer 0.841 (base 0.005); collateral mean KL 0.0131 on neutral text; capacity to 300 facts with no ceiling,
  damage growing sublinearly. **Failure mode: facts fire only when the prompt contains the same n-grams. Other languages/phrasings get the base model's
  answer exactly.** Composition of two facts mostly fails (4/83 both right).
  → [I] For Natsu-9B, the Engram table is also an **editable, removable knowledge store** (continual learning without touching weights). This is a
  9B-specific advantage: knowledge can be updated after training at the cost of a row overlay. The phrasing-locality failure is structural (addresses = exact token n-grams).
  [H] **E9 probe (queued s2j)**: write 64 invented facts into the trained E4j by (a) table-only, (b) full fine-tune, (c) FFN/MoE-only.
  Measure train-phrasing acc, **held-out-phrasing acc** (R39's failure mode), and collateral bpb on TinyStories. Prediction:
  (a) has the lowest collateral damage but the worst held-out phrasing; (b) the opposite.

### Issues/PRs as evidence (principle #4) — first pass over key repos' issue trackers
- **[R40] deepseek-ai/Engram issue #20 (8 comments)** — L2. Independent practitioner report. Llama-2-style **1.3B dense**, Engram at layers 2 and 10, 32k rows/layer,
  RedPajama 1T tokens, 250k steps. **(a) iso-param (FFN shrunk to keep 1.3B): no loss gain and no eval gain; (b) Engram added on top (more params):
  loss gain but eval gain only on HellaSwag.** Earlier at 30k steps: lower loss but *worse* PIQA/SIQA/ARC-e/HellaSwag.
  → [E, refuting at 1.3B dense, single uncontrolled report] This is the strongest counter-evidence to C4's core component found so far. Differences from
  R17 and from us: dense FFN (R17's iso-param win is vs MoE), only 32k rows/layer (≈ 2×32k×d params, small), layer placement, unknown gate/conv details.
  Our toy uses MoE and is iso-param vs MoE, matching R17's setting, not the issue's. [I] **The Engram win may be conditional on an MoE backbone and on
  table size (R17's U-shaped sparsity allocation law).** Also, the "lower loss but worse evals" pattern matches R31 (data-quality illusion) in spirit:
  bpb is not capability.
  Consequences: (1) the 10M stage (s2i) compares C4 vs no-Engram at iso-param *with MoE*, which is the right control. A dense control (L10c) is added.
  (2) A downstream eval at ≥ 50M is required before committing 1.6B of the 9B budget to the table. (3) Our toy bpb wins must not be reported as capability wins.
- **[R41] sandyresearch/parcae issues #4, #8, #10** — L2. #4 "conclusions questionable" (micro-experiments of a few dozen steps; the author answers that
  §5.1 is 1.3B / 100B tokens and the micro runs stop at divergence). **#8: 66% of training tokens were padding; #10: ~36% of tokens never seen, ~36% seen
  twice (per-rank shuffle with different seeds).** Both closed as fixed. [E] Parcae's published numbers were produced with a data pipeline that wasted
  ~2/3 of compute on padding and ~1/3 on duplicates (fixed later; whether the paper's numbers were re-run is unknown). → Treat Parcae's absolute
  scaling-law constants as [weak E]. #10 led directly to F009 in our own code.
- **3-seed loop result**: E4g R3 1.4574/1.4691/1.4731 (mean 1.4665, sd 0.0082) vs E4c mean 1.4792 → **loop −0.013 bpb, t ≈ 2.0 (n=3)** at 1.73× inference
  FLOPs. Marginal. Engram's −0.053 (t ≈ 12.8) is 4× larger at 1.06× FLOPs. The single-seed −0.030 claimed in session 1 was 2.4× too large (F008).

### N2 test (E4q vs E4r): memory-conditioned depth decider
| run | R1 | R2 | R3 | R6 | gate th0.3 (bpb, skip) | th0.5 |
|---|---|---|---|---|---|---|
| E4q loop+Engram+lookahead, **gate sees Engram gate value** | 1.4739 | 1.4288 | 1.4229 | 1.4609 | 1.4367 @ 39% | 1.4641 @ 87% |
| E4r same, **without** memory feature | 1.4671 | 1.4260 | 1.4223 | 1.4583 | 1.4356 @ 42% | 1.4564 @ 87% |
| E4j Engram, no loop (3-seed mean) | 1.4262 | | | | | |
- [E own] **N2: no effect.** E4q ≈ E4r on every row (Δ ≤ 0.008, within seed noise, and E4r is slightly better). The Engram gate scalar adds no information to
  the depth decider at this scale. The pre-registered AUC comparison (gate_eval) runs post-queue; unless it shows ≥ +0.03 AUC, N2 is refuted.
- [E own] **Loop + Engram, even with exit training, does not beat Engram alone**: best R3 1.4223 vs E4j 1.4262 (Δ −0.004, inside noise) at 1.6× FLOPs;
  any gated (cheaper) setting is *worse* than E4j at 1.06×. Confirms F007 with a stronger loop recipe. Exit training makes R1 usable (1.467 vs E4k's 1.622).
- **R17 re-read (L3-partial, §3 allocation law + Table 1)**: [E] U-shaped loss vs allocation ratio ρ (MoE share of sparse params). Optimum **ρ ≈ 75–80%**, stable
  across 2e20 and 6e20 FLOPs. Gain at the 10B regime: **Δ = 0.0139 nats** (1.7248 → 1.7109). Engram-27B vs MoE-27B at 262B tokens: Pile loss 1.960 → 1.950 (−0.010),
  MMLU +3.0, CMMLU +4.0, BBH +5.0. Dense-4B (same activated params) is far behind both (Pile 2.091), so **the comparison that matters is vs MoE, not vs dense.**
  → [I] (1) Natsu-9B C4 puts 1.611B of 6.99B sparse params in the table → **ρ = 77%, inside R17's optimum.** (2) The expected 9B gain is ~0.01–0.014 nats/token,
  i.e. **≈ 0.003–0.0045 bits per byte** (128k vocab, ~4.5 bytes/token). [Unit-corrected] That is **~12× smaller than our byte-level toy gain (0.053 bpb)
  and ~3× smaller than our BPE-4k toy gain (0.014 bpb)**. The gain shrinks with tokenizer granularity and scale, as R23 predicts (log-linear in effective input vocab).
  Toy numbers must not be extrapolated. A ~0.004 bpb gain at 9B is only worth 1.6B params because R17 shows it converts to +3–5 points on MMLU/BBH. (3) R40 (dense 1.3B, no iso-param gain) is consistent with R17: Engram's iso-param win is defined
  against MoE, and R17 never claims an iso-param win against dense. The 10M 2×2 tests exactly this.

### MoE vs dense at iso-STORED params (refutation check on TARGET_ANALYSIS §6 "C4 ≈ 4–6B dense")
- **[R42] Ludziejewski et al., "Joint MoE Scaling Laws: MoE Can Be Memory Efficient" (ICML 2025, arXiv:2502.05172)** — L3-partial (§4).
  [E] 280+ runs up to 2.7B active / 5B total. **Total-parameter-matched MoE beats dense at the same training compute** (validated at 1.1B total, E=2,4),
  even when the dense model is over-trained. Rule of thumb: at fixed total params, an MoE with **E ≤ 8** beats compute-optimal dense if trained on E× more tokens.
  More experts mean a higher optimal tokens/param ratio and a lower LR (Finding 4: LR ∝ E^−0.25). Diminishing returns at very large E.
  → [I] **My §6 projection was too pessimistic in one direction and unsupported in another.** R42 supports MoE ≥ dense at iso-total for small E (≤8)
  *given more tokens*. C4 uses 48 fine-grained experts, outside R42's validated E range (their E counts full-size experts; fine-grained granularity is
  Krajewski et al. 2402.07871, which finds the MoE–dense gap widens with scale). So "C4 ≥ C0 at iso-stored params, given enough tokens" is **supported [weak E]**,
  not "C4 ≈ 4–6B dense". Revised: quality parity with C0 at 9B stored is plausible if C4 trains on ≥ C0's token budget (cheap, since C4 is 3.1× cheaper per token).
  Action: LR scaling `lr ∝ E^-0.25` added to TRAINING_SPEC; TARGET_ANALYSIS §6 corrected.

- **E4s (Engram v2 VIP rows)**: 1.4337 vs E4j 1.4262 → worse, but **invalid**: a key-order bug in the VIP builder misrouted most VIP hits (F011).
  Rerun E4s2 with fixed keys (true VIP hit rate on validation: 93% of 2-grams, 55% of 3-grams) is queued.

### Novelty check for NEXT_GEN G1 "semantic address path" (principle #24) — Engram-family 2026 follow-ups found in our own corpus
- **[R43] MoME (arXiv:2609.15126)** — L2. Context-aware memory: each token's row becomes a mixture of M slots chosen by a gate over the hidden state.
  Beats Value-Embedding/Bigram/STEM at iso-param and iso-FLOP (sub-billion). Addresses polysemy, but the address is still the token id.
- **[R44] FactorEngram (arXiv:2609.35578)** — L2. Factorised n-gram memory: sparse coefficients over a shared basis dictionary, with basis-level contextual gating.
  Gains on LM and downstream at 340M/1B. Addresses "monolithic slot + scalar gate" and collision-only sharing.
- **[R45] Tensorized Engram (TN-gram, arXiv:2606.08347)** — L2. CP-factorised n-gram embeddings that share factors across orders. Matches or beats Engram with far fewer params.
- **[R46] Tokenizer-Agnostic Engram (arXiv:2607.29065)** — L2. Polynomial byte hashing gives hash equivalence across tokenizers. Comparable performance.
- → [I] **G1's "semantic address" is only partly novel.** Context-dependent *read-out* (MoME, FactorEngram) is done. What remains untested in this family:
  (a) a **content-addressed** (learned-key, PKM-style) path *alongside* the exact n-gram path, so that facts fire under paraphrase (R39's failure mode).
  None of R43–R46 evaluates paraphrase robustness of written facts. (b) **Parameter efficiency**: R44/R45 make tables much smaller at equal quality.
  For a 9B budget where the table is 1.6B (18%), a 3–10× smaller table at equal quality would free ~1–1.4B params for experts. That may matter
  more than any new address path. → New candidate: replace the hashed table in C4 by a factorised (R44/R45-style) table. Logged as H14.7; test after E9.
- Process note: the four relevant 2026 papers were already in our harvested corpus but had not been read. The L2 queue had not surfaced them because
  P7 queries pre-dated them. This is the case for targeted title queries over the existing DB before writing a novelty claim. Now a rule (see RESEARCH_PLAN).
- **E6 2×2 complete (varchain answer loss, 1 seed, 1200 steps)**: plain E6a **0.542** | loop E6c 0.914 | Engram E6b 0.917 | loop+Engram E6d 0.988.
  The R33-derived prediction (loop complementary to Engram on reasoning) is untestable here: every modification is worse than plain, and E6a is an outlier
  that went through its transition before step 1200 (curve check). No conclusion until E6L (2400 steps, 2 seeds) lands.

### E8 addition: depth vs loop vs memory (answer exact-match acc on 1–5 digits, 256 problems; one seed)
| run | params | eff. depth | acc | answer loss | 6-digit loss |
|---|---|---|---|---|---|
| E8a 2-core MoE | 0.82M | 4 | 0.297 | 0.603 | 5.32 |
| E8b 2-core × 3 loops | 0.82M | 8 | 0.285 | 1.037 | 4.91 |
| E8c 2-core + Engram | 0.83M | 4 | 0.242 | 1.080 | 5.25 |
| E8d loop + Engram | 0.83M | 8 | 0.258 (R3) / 0.270 (R4) | 1.048 | 5.49 |
| **E8e 6-core unlooped (iso-FLOP with E8b)** | 1.60M | 8 | **0.484** | 0.621 | 6.89 |
- [E own] **Depth with fresh parameters helps addition a lot (+0.19 acc); depth from loops does not (−0.01).** At iso-FLOP, unique layers beat shared
  layers on this task. That is the opposite of R33's claim ("k-layer looped L times ≈ kL-layer model on addition"). Engram hurts addition (−0.055 acc).
  Arithmetic over random digits has no reusable n-gram statistics, and the table took parameters from the MoE (expert_mult 0.296 → 0.17).
- [I] Candidate explanations for loop ≠ depth: (a) R33 trains to convergence, while our 1500 steps are a fixed short budget; looped models may need more steps.
  (b) R36: positional bottleneck. Both are tested by E8g/E8h (Abacus) and, for (a), a longer E8b would be needed.
  (c) E8e also has 2× params, so it is iso-FLOP, not iso-param. The cleanest single-variable comparison is E8b vs E8e = shared vs unshared weights at the same depth.
- Consequence for the design: **for reasoning-like algorithmic tasks, iso-FLOP unique depth > loops at toy scale.** This supports C4's choice (no loop) and
  G2 (spend extra inference compute on sampling/verification, not loops).
- **H13.7 (R33 block-cosine regulariser) on addition**: E8f (6-core + block_cos λ=1, k=2) acc **0.418** vs E8e (6-core, no reg) **0.484**; 6-digit loss 5.62 vs 6.89.
  [E own, 1 seed] Pulling layers toward a "loop-like" weight structure **costs in-distribution accuracy (−0.066)** and slightly improves the 6-digit loss
  (5.62 vs 6.89, still at chance accuracy). That is the same direction as weight sharing (E8b: worse in-dist, best 6-digit loss 4.23–4.91).
  [I] The "loop inductive bias" at toy scale trades in-distribution fit for marginally better extrapolation loss. It is not a free reasoning gain.
  H13.7 is not adopted. E4t (LM) is pending.

### Looped MoE at scale — refutation of our "loops don't pay" conclusion (from the 2026 loop papers in our corpus)
- **[R47] LOOM, "Looping Beyond Twice: A Scalable Recipe for Looped MoE" (arXiv:2610.01153)** — L3-partial. [E] Looped MoE fails to scale because of
  (1) hidden-state variance growth across loops and (2) **expert selection collapse: shared routers pick nearly the same experts every loop.**
  Fixes: residual scaling + embedding re-injection every loop + **per-loop routers** + a "Looping Residual". **Near-iso-FLOP at 700M: 5 loops,
  ppl 18.36 → 16.54, zero-shot 38.84 → 39.53.** 1.7B / 60B tokens (non-iso-FLOP) peaks at 9 loops.
- **[R48] "Scaling Laws for Looped Mixture of Experts" (arXiv:2609.40316)** — L2. [E] A joint law over recurrence × sparsity; **sparsity raises the
  effective-parameter gain from looping**. Downstream: sparsity ~3× active-param efficiency, recurrence ~2× total-param efficiency *on reasoning*.
  At trillion-token scale and matched training compute, a law-sized looped MoE matches a ~2× larger non-looped MoE on reasoning benchmarks.
- **Contradiction with our results (F005/F007/E8)**: our toy loop gain is −0.013 bpb (t≈2) and loops lose to unique depth on addition.
  [I] Differences that could explain it: (a) our loops used a **shared router + per-loop bias only** (`moe_loop_bias`). R47 says per-loop *routers* are
  needed, and a bias is a much weaker form; (b) no residual scaling (R47's variance fix); (c) scale and token budget (R47 gains appear at 100M–1.7B, 60B tokens;
  our 0.8M/1.6M tokens is ~4 orders of magnitude smaller); (d) we did not use embedding re-injection with residual scaling (E4i used concat without scaling, E4o LTI).
  **I therefore downgrade the conclusion "loops don't pay" to "loops don't pay at toy scale with our recipe."** That is weaker, and honest.
- Test (cheap, queued): `route_diag` measures cross-loop expert overlap on E4g/E4p/E4k/E8b (post_s2). If Jaccard ≫ chance, we reproduce R47's collapse.
  Next [H]: implement R47's per-loop routers + residual scaling (`moe_loop_router=True`, `loop_res_scale`) and rerun E4g-style. Pre-registered:
  if 3-seed gain stays < 0.02 bpb, loops stay demoted at toy scale and the question moves to the 50M+ ladder (GPU).
- **H13.7 on LM**: E4t (E4z + block-cos λ=1) **1.4355** vs E4z 1.4394 (single seed each) → −0.004, within seed noise (sd ≈ 0.007). No detectable effect on LM bpb.
  Combined with E8f (−0.066 acc on addition), **H13.7 is rejected** at toy scale.
- **[R49] "Beyond the Training Horizon" (arXiv:2609.33144)**, **[R50] "Shared Weights, Selected Computations" (arXiv:2609.39892)** — L2. Mechanistic loop studies
  (length generalisation on polynomial iteration, FSM composition, KG traversal; per-loop computation is selected by the entering hidden state).
  Both find loop mechanisms degrade at greater depths and errors compound, with limited self-correction. Neither tests iso-FLOP in-distribution accuracy
  vs unique layers, so they neither confirm nor refute E8e > E8b. Consistent with our E4g/E4o R6 degradation.
- **E5c born-again control (KD from a same-size teacher E4c, same recipe as E5a)**: **1.4578**. E4c 3-seed mean 1.4792 (sd 0.007); E5a (2× teacher) 1.4522.
  [E own, 1 seed] **~79% of the KD gain (−0.021 of −0.027 vs the E4c mean) does not need a larger teacher.** Teacher capacity adds only −0.006 (within noise).
  [I] At toy scale, KD acts mainly as regularisation / label smoothing with sample-specific targets (born-again effect), not capacity transfer.
  Consistent with R35 (KD helps mostly in the under-trained regime). Consequence: the "≥ 2× teacher" assumption in TRAINING_SPEC is unnecessary at small scale.
  **A cheap self-distillation round (train → distil into a fresh copy) is a candidate data-efficiency lever** in the data-limited regime, and it costs no
  larger model. At 9B (data-rich, R35), still expected ≈ 0.

### R36 test (Abacus digit positions) — E8g / E8h (1–5 digit addition, 1500 steps, 1 seed)
| run | acc (in-dist) | answer loss | 6-digit loss | 6-digit acc |
|---|---|---|---|---|
| E8a no loop | 0.297 | 0.603 | 5.32 | 0 |
| **E8g no loop + Abacus** | **0.430** (+0.133) | 0.634 | **4.40** | 0 |
| E8b loop R3 | 0.285 | 1.037 | 4.91 | 0 |
| E8h loop R3 + Abacus | **0.066** (−0.219) | 1.198 | 4.61 | 0 |
| E8e 6-core (iso-FLOP with loops) | 0.484 | 0.621 | 6.89 | 0 |
- [E own] **Abacus helps the non-looped model (+0.13 acc, better OOD loss) and makes the looped one much worse.** Pre-registered prediction
  ("both above 0.30; E8h > E8g on 6-digit") **FAILED on both counts**. R36's "positions fixed → loops add gains" does not reproduce here.
- Curve check: E8h's train loss was flat at ~1.5 from step 250 to 1000 and only started dropping at 1250 (E8g drops from 750). The looped model is
  **slower to escape the plateau**, and Abacus delays that escape further. At 1500 steps E8h is mid-transition, so this is a budget-sensitive result,
  like E6. R36 trains far longer (one GPU-day).
- [I] Pattern across E6, E8b, E8h: at fixed short budgets, **looped models enter algorithmic phase transitions later** than non-looped ones.
  This is a confound for every toy loop comparison here. A fair loop test needs either training to convergence or an iso-loss (not iso-step) protocol.
  Recorded as F012. Abacus itself is adopted as a cheap option for digit tasks (+0.13 acc at no FLOP cost).
- **E4l (loop R3 + shallow→deep self-distillation 0.5)**: R1 1.5039 / R2 1.4687 / **R3 1.4661** / R4 1.4735 / R6 1.5025.
  vs E4g 3-seed mean R1 1.683 / R3 1.4665 / R6 1.5508. [E own] Self-distillation **leaves R3 unchanged** (Δ −0.0004) and makes shallow and deep
  depths far more usable (R1 −0.18, R6 −0.048). It is the second "anytime" recipe (with E4p's exit training), at ~1.3× extra training compute
  (an extra shallow forward/backward). For the self-speculative draft role (INFERENCE_SPEC), E4l/E4p are the right training recipes: draft quality at R1 matters.

### Train-to-test compute allocation (P4, principle #12 "9B + more inference compute")
- **[R51] "Test-Time Scaling Makes Overtraining Compute-Optimal" (T² laws, arXiv:2604.01411)** — L2. [E] Jointly optimising model size, training tokens and
  inference samples (pass@k) under a fixed end-to-end budget: **when inference cost is counted, the optimal pretraining shifts deep into the overtraining
  regime**, far outside standard scaling suites. Validated by pretraining heavily overtrained models; the effect survives post-training.
  → [I] **Direct support for the project thesis** (a fixed small model + more test-time samples instead of a bigger model). It also pins the 9B token budget:
  overtrain well past Chinchilla (spec: 4–6T tokens ≈ 450–650 tok/param for a 9B; ~2000 tok/param of *active* for C4). This also argues for MoE (C4):
  per-sample inference cost is proportional to active params (2.6B), so T²'s optimum moves further toward "many cheap samples".
- **[R52] "When More Thinking Hurts: Overthinking in TTS" (arXiv:2604.10739)** — L2 (refutation side). [E] Marginal returns of extra reasoning tokens
  diminish quickly; extended reasoning abandons correct answers; optimal length varies with difficulty. → [I] G2 must allocate per-question
  (adaptive N / early stop on vote agreement), not a uniform N. Spec: majority-vote early stopping once the leading answer's margin exceeds a threshold.
- **E5a seed 1**: 1.4379 → E5a 2-seed mean **1.4450** vs E4c 3-seed mean 1.4792: **KD −0.034** (2 seeds, ≈ 5 sd of E4c's spread). KD gain confirmed at toy scale.
- **E6La (varchain, no loop, 2400 steps, seed 0)**: answer loss 0.041, **sequence EM 0.844** (long20 0.781). Steps-to-acc ≥ 0.2 at step 1600.
  The 1200-step E6a (EM 0.016) was mid-transition, which confirms the caveat. E6Lb (Engram) is running; it is the H14.1 test.

### Self-improvement in post-training (principle #15 "self-improvement loops")
- **[R53] Self-Verified Distillation (arXiv:2605.26132)** — L2. [E] A post-trained model generates candidate solutions to *unlabeled* seed questions, filters
  them by a 3-stage self-verification cascade (cycle-consistency, factuality, correctness; unanimous judge votes), and trains on the survivors.
  **Qwen3-4B: +16.7 pass@1 math (AIME26/HMMT), +11.1 science (GPQA-D/HLE), +8.3 code (LCB v5/v6)**, with no teacher and no tools.
  More samples and a larger verification budget give better data.
  → [I] For Natsu-9B post-training this is a teacher-free lever, and it composes with G2: the same adaptive sampling + verification machinery used at
  inference (`adaptive_vote`, verifiers) generates the training data. Risk: self-confirmation and collapse (refute queries in P5; R31-style Goodhart on
  the verification judge). Guard: hold out benchmarks from the seed pool; track pass@k as well as pass@1 (R19: RLVR narrows pass@k).
- **[R54] FROST (arXiv:2609.29988)** — L2. Online synthetic-data filtering by gradient alignment with real data. Filters 20–30% of synthetic data and improves the real task.
  → [I] A candidate for filtering the rephrased/synthetic share in pretraining (TRAINING_SPEC §3). Real-anchored, no external verifier.

### H14.1 test, part (a) — E6La vs E6Lb (varchain, 2400 steps, seed 0)
| run | answer loss | sequence EM | long20 EM | train loss @1200 / @2400 |
|---|---|---|---|---|
| E6La MoE (no Engram) | **0.041** | **0.844** | 0.781 | 0.574 / 0.038 |
| E6Lb MoE + Engram (iso-param) | 0.891 | **0.000** | 0.000 | 0.962 / ~0.79 |
- [E own, 1 seed, huge effect] **With twice the steps the Engram model still never forms the in-context retrieval circuit**, while the plain model solves the task
  (EM 0.84). This is not a timing artefact. Pre-registered condition (a) "E6Lb < E6La" was confirmed on seed 0 only. **RETRACTED as over-claim (F014): E6La seed 1 also fails to transition (EM 0.008); status: not established, 1/2 vs 0/1 transitions.**
  The size of the gap (EM 0.84 vs 0.00) is far beyond any seed effect seen so far.
- [I] Mechanism candidates: (1) R37/R38: the global-statistics shortcut (n-gram table) removes the gradient pressure that builds induction circuits.
  (2) Capacity: Engram took parameters from the experts (expert_mult 0.296 → 0.17). E8i-style control for varchain is needed (added below).
  (3) The Engram gate injects noise for random variable tokens, i.e. hash rows keyed on random values carry no signal and interfere.
- **Design impact (serious)**: the C4 core component may impair in-context learning at small scale. R17 reports RULER VT 77 → 89 at 27B, so the effect may
  reverse with scale or depend on training stage. **This makes N5 (delayed memory, E6Lc) and the 50M-ladder downstream/ICL evals decisive** for keeping Engram.

### P15 knowledge channel — first reads
- **[R55] "To Memorize or to Retrieve: Scaling the Interaction Between Pretraining and Retrieval" (arXiv:2604.00715)** — L2. [E] OLMo-2-based LMs, 30M–3B,
  up to 100B DCLM tokens, varying datastore size and source. **Retrieval gains are front-loaded: a median 91% of the largest improvement comes from a datastore of
  ~1 retrieval token per model parameter.** Small models gain more in gold-answer perplexity; larger, more pretrained models gain more in *accuracy*.
  Retrieval from already-seen data preserves most of the gain. Conclusion: explicitly partition data into "internalise" vs "externally access".
  → [I] For Natsu-9B, a datastore of ~9B tokens already captures most of the retrieval benefit, which is cheap (≈ 18 GB of text + an index). This is far smaller
  than the frontier assumption of trillion-token datastores. Combined with Engram's editable table (G1) and R39's phrasing locality, the knowledge channel
  design becomes: **parametric (weights) for skills, Engram table for frequent local patterns, retrieval datastore (~1 tok/param) for long-tail facts**.
  The "partition data between internalisation and external access" principle maps onto C4's three stores. Added to NEXT_GEN G1.

### Session 2 — refutation reads triggered by H14.1(a)
- **[R17b] Engram paper re-read (arXiv:2601.07372 §2–3, primary source)** — L3 for the implementation details. [E] Output `Y = SiLU(Conv1D_dil=maxN(RMSNorm(Ṽ))) + Ṽ`,
  conv zero-init "to strictly preserve the identity mapping", table Adam lr ×5 and no wd. Suppressing Engram at inference: TriviaQA keeps 29%, reading
  comprehension keeps 81–93%, so context-grounded tasks live in attention. → [E own] our v1 differs on 4 of 5 items (F013). H14.1(a) is downgraded to "v1-only" until E6Le.
- **[R56] Bietti et al. 2023, "Birth of a Transformer: A Memory Viewpoint" (arXiv:2306.00802)** — L2. [E] Global bigrams are learned first (FFN as associative memory),
  and the induction head forms later and more slowly. → [I] An n-gram table is a *faster* global-bigram learner, so it removes the residual loss that drives
  induction-head formation on mixed tasks. Supports mechanism (1) of H14.1. Predicts that a delay (E6Lc) helps only if the induction head forms during the delay window.
- **[R57] Singh et al. 2025, "Strategy Coopetition…" (arXiv:2503.05631, ICML)** — L2. [E] ICL and the in-weights hybrid (CIWL) share sub-circuits: ICL cannot emerge
  quickly on its own and needs CIWL's slow development, yet CIWL later replaces it. → [I] Refutation pressure on N5: a *full* delay of the in-weights channel might
  also slow ICL, because the cooperative half is removed. Prediction: E6Lc (delayed Engram) ≥ E6La is not guaranteed. A partial-scale ramp
  (engram_scale 0.1→1) is the variant consistent with coopetition. Will be added as E6Lf if E6Lc and E6Le are both negative.

### H14.1(a) final at n=2 seeds — REFUTED (F014)
| run | seed | answer loss | seq EM | long20 EM | transitioned |
|---|---|---|---|---|---|
| E6La no Engram | 0 | 0.041 | 0.844 | 0.781 | yes |
| E6La no Engram | 1 | 0.773 | 0.008 | 0.000 | no |
| E6Lb + Engram v1 | 0 | 0.891 | 0.000 | 0.000 | no |
| E6Lb + Engram v1 | 1 | 2e-5 | **1.000** | **1.000** | yes (train loss 0.05 @1000) |
- [E own] Transition count 1/2 vs 1/2. "Engram blocks in-context retrieval" is **not supported**, and the session's strongest negative evidence against C4's memory disappears.
  This is consistent with R17 (Engram helps VT/NIAH at 27B) and R17b (reading comprehension does not depend on Engram).
- [I] The real phenomenon is **phase-transition timing variance**. Steps-to-transition is the metric that matters for small-budget design decisions; mean EM is not.
- Consequence: N5 is demoted (its motivating harm is gone). E6Lc/E4u keep running only as cheap data (E6Lc adds two more transition samples with Engram).
  The 3rd seeds (s2m) stay, because n=3 per arm gives a transition-rate estimate. E6Le (paper path) stays, because F013 is a real divergence independent of H14.1.

### Reads on transition-time variance (follow-up to F014)
- **[R58] Zucchet et al. 2025, "The emergence of sparse attention: impact of data distribution and benefits of repetition" (arXiv:2505.17863)** — L2. [E] Emergence time of
  sparse attention follows power laws in task structure, architecture and optimiser. Repetition (in-context or cross-sample) speeds emergence a lot. It is confirmed on in-context
  associative recall, where emergence time grows with #pairs and vocabulary. → [I] varchain at n_vars=8, n_steps=12 sits near the edge of our 2400-step budget, so seeds straddle it
  (F014). Design lever for small-budget training: control emergence time through data repetition and burstiness, instead of reading architecture effects off single runs.
- **[R59] "What Happens During the Loss Plateau?" (arXiv:2506.13688, NeurIPS 2025)** — L2. [E] During the plateau there is a partial solution, repetition bias and representation
  collapse (hidden states nearly parallel). The bottleneck is slow attention-map learning, and intervening on attention changes plateau length. → [I] A cheap diagnostic for our runs
  is the hidden-state cosine during the plateau, which separates "about to transition" from "stuck". Candidate E6Lg: varchain curriculum (n_steps 4→12), which by R58 should shrink the
  seed spread. Not queued (budget); listed in next_experiments.

### s2h tail + post_s2 results (session 2)
**H14.1 / N5 final**
- E6Lc (delayed Engram, varchain) seeds: s0 no transition (EM 0.000), s1 transition at step 1000 (EM 0.992). Transition counts: no-Engram 1/2, Engram 1/2, delayed 1/2. → (b) is uninformative, since all arms are the same.
- **E4u (LM, Engram ramped in at 40–60%)**: bpb **1.4770** vs E4c 1.4792 (no Engram) and E4j 1.4262 (Engram from step 0). It keeps **4%** of the gain, while ≥70% was pre-registered. → **(c) fails, N5 rejected.**
  [I] A token-indexed memory needs the whole run to fill. On LM data the table is the fast learner (R56), and delaying it throws away its main advantage. Together with F014, Engram from step 0 stays.

**N2 gate AUC (gate_eval)**: E4q (memory-conditioned) AUC 0.675, E4r (without) 0.655. Skipping 40% of loop compute costs +0.013 bpb in both. → [E] Consistent with F010: the memory feature adds ≤0.02 AUC, not useful.

**Route diagnostics (route_diag, R47 collapse test)**: same-token expert-set Jaccard across loops is **0.55–0.88**, against chance 0.14, and load cosine is 0.79–0.997 (E4g/E4p/E4k/E8b).
- [E own] **Per-loop routing collapse is confirmed**: loops re-pick mostly the same experts, so loops add depth without adding expert diversity (R47's claim, reproduced at toy scale).
- The per-loop bias alone does not prevent it. E4y/E4y2 (separate per-loop routers) are the direct test and are queued in s2l.

**Self-speculation with loops (bench)**: R1 draft / R3 verify gives exact outputs in every case.
- Accept rate: E4g 0.64, E4p 0.83, E4l (self-distilled) **0.91**, which is 4.65 tokens per target call.
- [E] Self-distillation is what makes the shallow draft agree with the deep model. That is a second benefit of E4l beyond its bpb.
- Wall-clock per token on CPU is still worse (0.015 vs 0.0117 s/tok), because at d=128 the per-call overhead dominates. The speedup claim waits for the ≥50M GPU stage.

**Downstream MC (ds_eval)**: cloze 0.245–0.29 (n=200, SE≈0.031) and names 0.36–0.48 (n=171, SE≈0.038), chance 0.25.
- [E] Every 0.8M model is near chance on cloze. Names is above chance for all, but with no significant between-model differences (the E3a attention-only model is lowest at 0.36).
- Uninformative at this scale, as expected. The harness stays ready for 50M.

**TTS curves (tts_eval, 5-digit addition)**: greedy accuracy is 0 for every model. A small check on 48 problems by digit count:

| model | 1 digit | 2 digits | 3 digits | 5 digits |
|---|---|---|---|---|
| E8e (6-core) | 0.40 | 0.46 | 0.19 | 0 |
| E8a | 0.48 | 0.08 | 0.06 | 0 |

- **Floor effect, an eval design flaw (F015)**: 5 digits is above every model's ability, so there is no TTS signal.
- Rerun at 2–3 digits is queued (post_s2b).

**kNN-LM (G1b / R55), datastore = E4c's own hidden states on train tokens**:

| model | store tokens | λ=0 | λ=0.1 | λ=0.25 | λ=0.5 |
|---|---|---|---|---|---|
| E4c | 0.8M (≈1 tok/param) | 1.5008 | 1.4387 | 1.3947 | **1.3732** |
| E4c | 3.2M (≈4 tok/param) | 1.5008 | 1.4031 | 1.3335 | **1.2793** |
| E4j (Engram) | 0.8M | 1.4548 | 1.3994 | 1.3639 | **1.3576** |

- [E own] Retrieval gives **−0.13 bpb at 1 tok/param and −0.22 at 4 tok/param**. That is 2.5–4× the Engram gain, and still improving at the largest λ, so the λ grid must be extended.
- The base bpb (1.50) is on a different valid slice than the training eval (1.479), so compare within this table only.
- **Leakage audit** (fg run): 39% of validation 16-grams, 6.6% of 32-grams and 0.4% of 64-grams also appear in the first 3.2M train tokens. TinyStories is highly redundant, so this gain is mostly *near-duplicate recall*. That is legitimate retrieval of seen data (R55), but it overstates the gain on low-redundancy corpora. **Do not extrapolate the magnitude to 9B web data.**
- [E own] **Engram and kNN are partly substitutes**: the Engram advantage shrinks from 0.046 (λ=0) to 0.016 (λ=0.5, 0.8M store). Both exploit local repetition. The 3.2M E4j row is still pending.
- Against R55's "91% of the gain at ~1 tok/param": here 1 tok/param gives 0.128 of 0.222 (58%) and the gain has not saturated at 4 tok/param. That contradicts R55 on this corpus, likely because of the redundancy.
- **kNN update (E4j, 3.2M store)**: λ 0/0.1/0.25/0.5 → 1.4548 / 1.3618 / 1.2967 / **1.2494**. Engram advantage over E4c: 0.046 (λ=0) → 0.016 (0.8M, λ=0.5) → **0.030 (3.2M, λ=0.5)**.
  [E own] Engram + kNN is the best combination so far (1.249). The two are **partly additive, not pure substitutes**: about ⅓–⅔ of the Engram gain survives retrieval.
  [I] For C4 this is the three-store design (weights / Engram / datastore) from NEXT_GEN G1b, now with a first toy-scale measurement. The open cost is datastore RAM/disk:
  3.2M keys × 128-d fp16 = 781 MB, i.e. **~244 B per stored token**. At 9B tok/param scale that is ~2.2 TB uncompressed, so G1b needs PQ/low-dim keys (next: key-dim ablation).

### Refutation reads for G1b (kNN-LM)
- **[R60] Memory Decoder (arXiv:2508.09874)** — L2. [E] A small transformer (124M) is pretrained to imitate the kNN-LM distribution (KL to the cached kNN distribution + LM CE) and plugged into GPT-2.
  On Wikitext-103 it reaches 13.36 PPL vs kNN-LM 15.62 vs in-context RAG 18.46, with 1.28× latency vs kNN-LM 2.17×. **kNN-LM often *degrades* knowledge-intensive QA (NQ, HotpotQA)**, while MemDec improves it.
  There is no datastore at deployment.
  → [I] (1) Refutes "perplexity gain ⇒ task gain" for kNN-LM. Our −0.22 bpb is a perplexity number, and ds_eval cannot measure the task side at 0.8M.
  (2) **Gap / H15.1 "retrieval-distilled pretraining"**: MemDec distils kNN into a *separate* decoder. Distilling the kNN-mixed distribution into *the model's own Engram table* is not found in the corpus.
  Mechanism: the table is token-indexed like kNN keys are context-indexed. The DPT machinery (E5) is reused with teacher = (1−λ)p_LM + λp_kNN of a frozen checkpoint.
  Deployment would then need neither a datastore nor extra latency.
  Cost (**corrected**; the first estimate of 80 TFLOP was wrong by ~10³): exact kNN for 1.6M queries × 0.8M keys × 128-d is ≈3×10¹⁷ FLOP, infeasible on CPU.
  With an IVF index (1024 lists, 8 probes, ≈6.4k candidates per query) it is ≈1.6M × 0.95 MFLOP ≈ 1.5 TFLOP, i.e. minutes. **ANN is a prerequisite for G1c and for G1b at scale.**
  Trap to handle: the store contains the training windows themselves, so self-matches must be excluded (positions within ±seq of the query).
  Not queued yet: the queue is full for ~15 h, and the JL-compression result (post_s2b) decides the store format first.
- **[R61] "kNN-LM Does Not Improve Open-ended Text Generation" (Wang et al., OpenReview 3FNrGv5MKb)** — L1 (PDF blocked by CAPTCHA, abstract-level only). The title-level claim agrees with R60:
  PPL gains do not transfer to generation quality. Logged as refutation evidence and taken into G1b's risk list.
