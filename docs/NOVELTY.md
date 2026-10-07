# Novelty analysis (deliverable H) — explainable differences, session 2

Rule: a contribution counts only if (a) the difference from the closest prior work is stated, (b) the bottleneck it targets is named,
(c) a falsifying experiment exists. Prior-art status comes from targeted searches (taxonomy P11/P13) and the reads table.
"Not found" means "not found in our searches", not "does not exist".

## N1. Memory-before-loop: Engram placed before a weight-shared loop core  [status: **REFUTED on bpb (F007)**; reasoning-probe 2×2 pending (E6d/E8d)]
- Closest prior art: Engram (R17, non-looped), Over-Encoding (R23, input side), looped MoE (R4), MoR (R29). No looped model with
  token-indexed memory was found (searched P11/P13 queries; R4 and MELT related-work sections, L3-partial).
- Bottleneck: in looped models the early "static reconstruction" work (R17 mechanism) is repeated or wasted inside every loop pass. Memory computed
  once before the loop frees all R passes.
- Why 9B-specific: under a stored-param cap, both components are parameter-cheap ways to buy capacity: memory = cheap stored rows, loop = depth
  without params.
- Falsifier: if E4k (loop+Engram) − E4g (loop) ≥ E4j (Engram) − E4c (no-loop) − noise, the effects are additive or synergistic; if smaller, they are substitutes.

## N2. Memory-conditioned depth: use the Engram *hit signal* as a feature for the per-token loop decider  [status: **no effect (F010)**; AUC check pending; parked]
- Observation chain: (i) loops only beat non-looped models at matched compute when depth is token-adaptive (R21, R29); (ii) Engram's gate value
  g_t = σ(⟨q(h_t), k(m_t)⟩) measures how well a static n-gram explains the current token (R17 mechanism; our `Engram.forward`);
  (iii) tokens that are predictable from local n-grams should need fewer loops.
- Proposal: the depth decider sees [h_t, g_t, entropy of the memory read]. It is trained with lookahead labels (R21-style; implemented `gate_mode=lookahead`).
- Difference from prior work: TaH2 / MoR routers see only the hidden state. Here a *retrieval-confidence signal from a separate parametric memory*
  conditions compute allocation. The analogy is neuroscience-style complementary learning systems (fast lookup vs slow deliberation).
- Bottleneck: router quality (R16: halting collapses; R21: needs good labels). The memory signal is a cheap, already-computed feature.
- Falsifier: router AUC for "loop helps" with vs without g_t features; compute saved at equal bpb.
- Cost: d+2 → 1 linear per loop. Negligible.

## N3. Anytime-tax curriculum (fixed-R → sampled-R)  [status: **failed pre-registered test (E4m 1.4701 vs E4g 1.4574)**; superseded by exit training (E4p), which is prior art (deep supervision / LayerSkip)]
- Evidence that motivates it is our own (F005 update): uniform-R costs ~0.07 bpb at toy scale; fixed-R wins but has no anytime property.
- Prior art: Huginn and Parcae train with sampled R from the start. No fixed→sampled curriculum was found.
- Falsifier: E4m at R=3 within 0.01 of E4g *and* R=1/R=2 close to E4h.

## N4. Budget accounting under a stored-parameter constraint with offloadable memory (C5)  [status: I, analytic]
- `docs/generated/efficiency_9b.md`: C4 at 9.1B stored keeps 1.6B in an Engram table that is prefetchable (R17) and SSD-decodable (R15 for PKM).
  Resident 4-bit weights are 3.75 GB vs 4.6 GB for a Qwen3.5-9B-like model. FLOPs/token are 3.1× lower, KV 2.7× lower, decode bytes 3.2× lower.
- This is an engineering consequence of known components, not a new mechanism; it is claimed as a design point on the Pareto frontier.

## N5. Delayed conditional memory (memory ramped in after in-context circuits form)  [status: DEMOTED — motivating harm refuted (F014: Engram 1/2 vs no-Engram 1/2 transitions); kept as a low-priority option]
- **Difference from prior art**: Engram/OE/X-gram train the memory from step 0. Induction-head work (R37/R38) shows simple global-statistics
  solutions delay in-context circuits. Nobody (found) schedules a token-indexed memory to avoid this.
- **Bottleneck addressed**: the possible early-training conflict between global n-gram memory and in-context retrieval (E6, weak evidence),
  reconciled with R17's late-stage gains (RULER VT 77→89).
- **Why 9B-relevant**: at 9B the memory table is ~1.6B params (18% of the budget). If it slows induction formation, the cost is paid on every
  in-context task. The fix costs nothing at inference.
- **Status update (F013/F014)**: the motivating evidence is now weaker. (i) The E6La/E6Lb outcome is seed-bimodal (E6La s1 also failed), and (ii) our Engram v1
  was not paper-faithful (no internal residual, random conv init). N5 is tested only if ≥3-seed transition counts still favour no-Engram, under the faithful path too.
- **Falsifier**: E6Lb ≥ E6La (no harm) → N5 is unnecessary; E6Lc ≉ E6La → wrong mechanism; E4u loses > 30% of the LM gain → too costly.

## N6. KD × memory additivity (soft targets and token-indexed memory are complementary)  [status: weak E, E5b]
- E5b: Engram −0.054 and then KD −0.022 more (vs Engram + loop: sub-additive, F007). Not novel as components. The measured complementarity under a
  fixed param budget is, to my search, unreported. R35 bounds its value at high tokens/param.

## Not novel (explicitly)
GDN hybrid 3:1 (Qwen3.5), MoE with shared experts, MTP, shared-first KV (MoR/MELT), LTI injection (Parcae), lookahead depth labels (TaH2),
loop self-speculation (LoopSpec), DPT distillation, 8-bit Adam. These are implemented as components and credited.
