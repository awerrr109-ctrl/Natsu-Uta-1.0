# F005 — E4: looped models are WORSE than non-looped at iso-param on TinyStories (toy scale)

- **hypothesis** (H1.1 / H11.1, main thesis): at ≈iso-stored-params (0.82–0.87M), a looped core (R=3) beats the same
  non-looped model on language modeling, and looped-MoE beats looped-dense (routing divergence, R4).
- **implementation**: byte-level TinyStories, 800 steps × 16 × 128 = 1.64M tokens, AdamW 3e-3, WSD; looped runs use
  uniform loop sampling R~U{1..3} + Huginn-style concat re-injection (Linear(2d→d), *random init*).
- **observed** (val bpb, lower is better; single seed):

  | run | params | eval R | bpb | train FLOPs |
  |---|---|---|---|---|
  | E4c hybrid MoE, no loop | 0.82M | 1 | **1.4870** | 7.3e12 |
  | E3b hybrid dense, no loop | 0.82M | 1 | 1.4917 | 9.0e12 |
  | E3a attention-only dense | 0.76M | 1 | 1.5588 | 8.1e12 |
  | E4d "Natsu toy" (loop-MoE+gate+MTP+self-distill) | 0.87M | 2 | 1.5647 | 1.0e13 |
  | E4b loop-MoE | 0.85M | 2 | 1.5748 | 1.0e13 |
  | E4a loop-dense | 0.86M | 2 | 1.5869 | 1.4e13 |

  Additional observations:
  - Within looped runs: **R=2 is best, and R=3 (the trained max) and extrapolation R=4,6 are monotonically worse.** There is no test-time scaling.
  - MoE > dense in both the looped (−0.012) and non-looped (−0.005) settings → consistent with R4's "MoE helps".
  - Natsu-toy extras (self-distill + MTP + gate) help the loop model (−0.010 vs E4b), but it is still 0.078 bpb behind no-loop.
  - Depth gate: threshold 0.5 skips only 5% of loop-work; 0.7 skips 24% at +0.03 bpb → there is no meaningful learned budget yet.
- **why it failed — candidate causes (ranked by prior)**:
  1. **Re-injection init bug [confirmed defect]**: `inj = Linear(2d→d)` with random init maps x to a vector with cos(x, inj([x,e])) ≈ 0 and
     a 3× smaller norm (measured: 11.3 → 3.6). Every loop r≥1 begins by *destroying* the residual stream; the model must learn to
     undo it. Fix: identity init (inj([x,e]) = x at step 0), now implemented.
  2. **Uniform loop sampling** spends 1/3 of the steps at R=1, so the deep path gets fewer updates. Huginn used a heavy-tailed
     distribution over many loops with long training. At 800 steps the cost may dominate.
  3. **Scale/regime**: R2 shows φ≈0.46 and R4 shows dense loops scale worse. At 0.8M params and 1.6M tokens, data-limited models may
     not benefit from depth at all: language modelling at this scale is mostly n-gram statistics (cf. Engram's argument).
  4. Reasoning benefits of loops (R1) appear on reasoning tasks, not bpb; R1 itself reports that looping hurts perplexity/memorisation.
- **lessons**:
  - Weight-shared re-entry points need **identity-preserving init**. This is a general rule for any added "merge" layer.
  - "Natsu thesis = loops help at iso-param" is **not supported** by the first controlled toy test. It must not be
    reported as working.
  - The R=2 > R=3 inversion is the signature of an under-trained deep path, the same symptom R16 saw with learned halting.
- **next hypotheses / experiments** (queued as s1e):
  - E4g fixed R=3, no re-injection (isolates cause 1 + 2)
  - E4h uniform R, identity-init re-injection (isolates cause 1)
  - E4i fixed R=3, identity-init re-injection
  - if all are still worse than E4c → accept that **at toy scale, loops do not pay for LM bpb**, keep C4 (MoE + Engram, no loop) as
    the main line, and test loops only on reasoning probes at the GPU stage. Decision rule written down **before** seeing the results.

## UPDATE (session 2) — the "loops hurt" conclusion was WRONG in its generality
- **E4g** (loop-MoE, **fixed R=3 training, no re-injection**, same 0.82M params, same data): val bpb by eval-R =
  R1 1.694 / R2 1.485 / **R3 1.457** / R4 1.474 / R6 1.552. **At R=3 it beats the non-looped E4c (1.487) by 0.030 bpb.**
- Therefore the F005 deficit came from the training recipe (uniform R ~ U{1..3} and/or random-init concat re-injection), not from
  looping per se. E4h (uniform R + identity-init re-injection) and E4i (fixed R + identity re-injection) separate the two causes.
- New observations: (a) a fixed-R model is useless at R=1 (1.694): there is **no anytime property**; (b) it is mildly robust at R=4, not at R=6;
  (c) the loop costs 1.72× inference FLOPs for −0.030 bpb, while Engram (E4j) gave −0.059 at 1.06× FLOPs → **Engram still dominates on the Pareto front.**
- **Lesson**: one failed recipe ≠ a refuted mechanism. Pre-registered ablations caught this. Without them the loop line would have been wrongly dropped.
- **New hypothesis H13.1**: uniform-R training imposes an "anytime tax", since the shared weights must be good at every depth. The tax shrinks with
  scale or with a curriculum (train at fixed R, then fine-tune with sampled R). This is testable.

## UPDATE 2 (session 2) — cause decomposition
| run | injection | loop sampling | best bpb (R) |
|---|---|---|---|
| E4b | concat, random init | uniform U{1..3} | 1.5748 (R2) |
| E4h | concat, **identity init** | uniform | 1.5291 (R2) |
| E4g | none | **fixed 3** | **1.4574 (R3)** |
| E4c | (no loop) | — | 1.4870 |
- Identity init recovers 0.046 bpb (it is a real defect, now fixed). **Uniform-R sampling costs ~0.07 bpb more** and is the dominant cause.
  Uniform R still lacks depth scaling (R2 best, R3 worse). The "anytime tax" (H13.1) is large at toy scale.
- Remaining isolations queued: E4i (fixed R + identity init), E4m (fixed→uniform curriculum), E4n (Poisson mean R), E4o (Parcae LTI).

## UPDATE 3 (session 2) — curriculum and lookahead gate
| run | recipe | R1 | R2 | R3 | R4 | R6 | best |
|---|---|---|---|---|---|---|---|
| E4g | fixed R=3 | 1.694 | — | **1.4574** | — | 1.552 | 1.4574 |
| E4m | fixed R=3 for 50%, then U{1..3} | 1.5213 | **1.4701** | 1.4796 | 1.5069 | 1.5887 | 1.4701 (R2) |
| E4p | fixed R=3 + lookahead gate (TaH2-style, intermediate exits through coda) | 1.5132 | 1.4655 | **1.4607** | 1.4696 | 1.5071 | 1.4607 |
- **E4m (H13.1 curriculum)**: the anytime tax is reduced but not removed. Best is 1.4701 vs E4g 1.4574 (+0.013), R1 improves massively (1.694→1.521), but
  depth scaling vanishes again (R2 ≥ R3). Pre-registered "within 0.01 of E4g at R3": **fails** (+0.022 at R3).
- **E4p**: exit training makes *every* depth usable (R1 1.513, R6 1.507 — the most depth-robust run so far) at a cost of +0.003 at R3 vs E4g. This is the
  first loop recipe that is both anytime and near-best. Gate quality (gate_eval): **AUC 0.662** for "loop helps by >0.02 nats" (35% of tokens).
  Skip trade-off: th=0.3 → 33% tokens skip loops, bpb 1.4620 (+0.011 vs full 1.4513 on the gate_eval split) at 0.86× FLOPs (1.49 vs 1.73 rel. to R1);
  th=0.5 → 81% skip, +0.044 bpb at 0.66×.
  → gain per FLOP of the gated loop vs no loop (E4c 1.4870 at 1.0×): −0.025 bpb at 1.49× vs Engram −0.059 at 1.06×. **Engram still dominates.**
- **Lesson**: intermediate-exit training (E4p) is the right way to get anytime loops (better than sampled R). A 0.66 AUC gate is weak:
  the per-token "loop helps" signal is mostly unpredictable from hidden states at this scale. N2 (E4q vs E4r) tests whether Engram's gate value adds information.

## UPDATE 4 (session 2) — seed noise and reinjection
- Seed spread is 0.003–0.014 bpb. The 2-seed loop gain is **−0.017** (E4c mean 1.4803 → E4g mean 1.4633), not −0.030. E4p/E4e2 vs E4g are within noise.
- E4i (fixed R + identity-init concat reinjection): R3 1.4843, but R1 3.08 and R6 2.54. Unconstrained reinjection is unstable off the trained depth.
  The pre-registered LTI test (E4o) is still pending (requeued).
- **Lesson (methodological, also see F008)**: I ranked loop recipes at 0.003–0.01 resolution on one seed. Only differences ≥ ~0.02 (≥1.5× the observed seed spread) are claims from now on.

## UPDATE 5 (session 2) — E4o: Parcae-style LTI reinjection (pre-registered R28/R20 test)
| run | R1 | R2 | R3 | R4 | R6 | R6−R3 |
|---|---|---|---|---|---|---|
| E4g no reinjection (3-seed mean) | 1.683 | 1.495 | 1.4665 | 1.4801 | 1.5508 | +0.084 |
| E4i concat identity-init | 3.083 | 1.613 | 1.4843 | 1.583 | 2.536 | +1.05 |
| **E4o LTI (Ā=exp(−dt·exp(logA)), B init 0)** | 1.718 | 1.487 | **1.4535** | 1.4715 | 1.5726 | +0.119 |
- **Pre-registered prediction (R28 notes): "LTI degrades less at R=6 than E4g". FAILED**: E4o's R6−R3 gap (+0.119) is larger than E4g's (+0.084).
  LTI does fix the *explosion* of naive concat injection (E4i +1.05 → +0.119), so it is the right injection form if any is used.
- R3 1.4535 is the best single-seed loop number, 0.013 below the E4g 3-seed mean, i.e. ~1.6 sd. One seed, so not a claim.
- [I] Depth extrapolation at toy scale is limited by training at a fixed R=3 (a single iteration count), not by injection stability. R28's fixed-point
  argument applies to models trained toward a fixed point (many iterations or a fixed-point objective), which we do not do.
