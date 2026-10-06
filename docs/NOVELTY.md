# Novelty analysis (deliverable H) — explainable differences, session 2

Rule: a contribution counts only if (a) the difference from the closest prior work is stated, (b) the bottleneck it targets is named,
(c) a falsifying experiment exists. Prior-art status comes from targeted searches (taxonomy P11/P13) and the reads table.
"Not found" means "not found in our searches", not "does not exist".

## N1. Memory-before-loop: Engram placed before a weight-shared loop core  [status: H, toy test E4k running]
- Closest prior art: Engram (R17, non-looped), Over-Encoding (R23, input side), looped MoE (R4), MoR (R29). No looped model with
  token-indexed memory was found (searched P11/P13 queries; R4 and MELT related-work sections, L3-partial).
- Bottleneck: in looped models the early "static reconstruction" work (R17 mechanism) is repeated or wasted inside every loop pass. Memory computed
  once before the loop frees all R passes.
- Why 9B-specific: under a stored-param cap, both components are parameter-cheap ways to buy capacity: memory = cheap stored rows, loop = depth
  without params.
- Falsifier: if E4k (loop+Engram) − E4g (loop) ≥ E4j (Engram) − E4c (no-loop) − noise, the effects are additive or synergistic; if smaller, they are substitutes.

## N2. Memory-conditioned depth: use the Engram *hit signal* as a feature for the per-token loop decider  [status: H, new]
- Observation chain: (i) loops only beat non-looped models at matched compute when depth is token-adaptive (R21, R29); (ii) Engram's gate value
  g_t = σ(⟨q(h_t), k(m_t)⟩) measures how well a static n-gram explains the current token (R17 mechanism; our `Engram.forward`);
  (iii) tokens that are predictable from local n-grams should need fewer loops.
- Proposal: the depth decider sees [h_t, g_t, entropy of the memory read]. It is trained with lookahead labels (R21-style; implemented `gate_mode=lookahead`).
- Difference from prior work: TaH2 / MoR routers see only the hidden state. Here a *retrieval-confidence signal from a separate parametric memory*
  conditions compute allocation. The analogy is neuroscience-style complementary learning systems (fast lookup vs slow deliberation).
- Bottleneck: router quality (R16: halting collapses; R21: needs good labels). The memory signal is a cheap, already-computed feature.
- Falsifier: router AUC for "loop helps" with vs without g_t features; compute saved at equal bpb.
- Cost: d+2 → 1 linear per loop. Negligible.

## N3. Anytime-tax curriculum (fixed-R → sampled-R)  [status: H, E4m running]
- Evidence that motivates it is our own (F005 update): uniform-R costs ~0.07 bpb at toy scale; fixed-R wins but has no anytime property.
- Prior art: Huginn and Parcae train with sampled R from the start. No fixed→sampled curriculum was found.
- Falsifier: E4m at R=3 within 0.01 of E4g *and* R=1/R=2 close to E4h.

## N4. Budget accounting under a stored-parameter constraint with offloadable memory (C5)  [status: I, analytic]
- `docs/generated/efficiency_9b.md`: C4 at 9.1B stored keeps 1.6B in an Engram table that is prefetchable (R17) and SSD-decodable (R15 for PKM).
  Resident 4-bit weights are 3.75 GB vs 4.6 GB for a Qwen3.5-9B-like model. FLOPs/token are 3.1× lower, KV 2.7× lower, decode bytes 3.2× lower.
- This is an engineering consequence of known components, not a new mechanism; it is claimed as a design point on the Pareto frontier.

## Not novel (explicitly)
GDN hybrid 3:1 (Qwen3.5), MoE with shared experts, MTP, shared-first KV (MoR/MELT), LTI injection (Parcae), lookahead depth labels (TaH2),
loop self-speculation (LoopSpec), DPT distillation, 8-bit Adam. These are implemented as components and credited.
