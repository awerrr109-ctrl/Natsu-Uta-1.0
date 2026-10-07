# F013 — Natsu's Engram v1 output path was not the paper's (found by re-reading the primary source after H14.1(a))

- **trigger**: E6Lb (MoE+Engram) sequence EM 0.000 vs E6La 0.844 on varchain. That contradicts R17, where Engram *improves* RULER VT (77→87–89) and MQ-NIAH (84→97).
  Before blaming the idea (H14.1), check whether we implemented the idea. Primary source re-read (arXiv:2601.07372 §2, §3 config).
- **divergences found** [E, code vs paper]:
  | item | paper | Natsu v1 |
  |---|---|---|
  | output | `Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ` (internal residual) | `Y = SiLU(Conv1D(Ṽ))` (no residual, no norm) |
  | conv init | **zero** (module ≈ identity on Ṽ at start) | random N(0,1/k) |
  | conv dilation | max N-gram order (3) | 1 |
  | table optimiser | Adam, **lr ×5**, no wd | base Adam lr, no wd |
  | gate | σ(RMSNorm(h)·RMSNorm(k)/√d) | same (OK) |
- **why it matters for ICL** [I]: with random conv init and no residual, Engram injects an un-gated-in-practice, position-mixed random signal from step 0
  into the residual stream right after the prelude, on every token, including random variable values whose hash rows carry no signal.
  With zero-init + residual, the signal is exactly `gate·v`, so the gate can switch it off for unpredictable tokens. That is mechanism (3) in the H14.1 note.
- **status**: hypothesis, not yet a failure of the idea. The H14.1(a) result stands as a property of **Engram v1** only. Every claim that says "Engram"
  in REPORT_S2/NOVELTY is a claim about v1 until E6Le/E4pp report.
- **fix**: `engram_paper=True` (model) + `train.engram_lr_mult=5` (separate Adam group). Default stays v1 so all past runs remain reproducible.
  Decode-consistency test `test_engram_paper_cache` (dilated conv state of span (k-1)·dil).
- **pre-registered**: E6Le seq EM ≥ 0.42 (½·E6La) → the ICL penalty was an implementation artefact; adopt the paper path and rerun E6Lc-style questions on it.
  E6Le ≈ 0 → the penalty is a property of n-gram memory at this scale (H14.1 survives the faithful implementation). E4pp: bpb ≤ 1.4212 adopt; ±0.005 neutral.
- **lessons**: (1) after a surprising negative result on a published component, first diff the implementation against the paper's equations and
  *training config*, not just its block diagram. (2) Optimiser details (lr multipliers, init) are part of the method. Added to the RESEARCH_PLAN checklist.

- **results so far**: E6Le (varchain, s0) transitioned at step 600, EM 1.000, on the seed where v1 failed. **E4pp (LM)**: bpb 1.4336 vs v1 1.4262 ± 0.0016 → +0.0074 (worse).
  Neither "paper = better" nor "v1 = fine" holds across both tasks. Ablation E4pq/E4pr (s2n) separates the output path from lr ×5. **C4 keeps v1 for LM until the ablation reports.**
- **varchain 3 seeds**: paper path 3/3 transitioned at steps 600/600/400 vs 1/3 for v1 and 1/3 without Engram → the implementation divergence was the cause of the ICL-side discrepancy with R17.
