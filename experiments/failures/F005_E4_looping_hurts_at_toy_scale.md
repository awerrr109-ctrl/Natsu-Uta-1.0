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
