# F007 — N1 "memory-before-loop" complementarity: REFUTED at toy scale (byte level)

- **hypothesis (N1, docs/NOVELTY.md)**: Engram before a looped core is complementary: gain(loop | Engram) ≥ gain(loop | no Engram).
- **implementation**: identical recipe (fixed R=3, no reinjection, 1.64M byte tokens), iso-param (~0.82–0.83M), one seed.
- **pre-registered falsifier**: E4k − E4g ≥ E4j − E4c − noise means additive.
- **observed** (val bpb):

  | | no loop | loop R=3 |
  |---|---|---|
  | no Engram | E4c 1.4870 | E4g 1.4574 (−0.030) |
  | Engram | E4j **1.4278** (−0.059) | E4k 1.4327 (−0.054 vs E4c) |

  gain(loop | no Engram) = −0.030; gain(loop | Engram) = **+0.005** (worse, and 1.6× FLOPs). gain(Engram | loop) = −0.025 vs gain(Engram | no loop) = −0.059.
  → **Strongly sub-additive. The two mechanisms are substitutes at this scale.** E4k is dominated by E4j on every Pareto axis.
- **interpretation [I]**: both buy "effective depth" for the same thing, local/static pattern reconstruction. Engram does it by lookup (R17's mechanism claim),
  the loop does it by recomputation. Once the lookup handles it, a 0.8M model trained on 1.6M tokens has little left that benefits from extra depth.
  This agrees with R17's "Engram effectively deepens the network": deepening twice does not help.
- **what this does NOT refute**: loops for *reasoning* tasks (R1/R3 claim loop gains on manipulation, not LM loss), loops at larger scale/tokens,
  token-adaptive loops (E4q/E4r: compute only where needed, and the Engram gate as a router feature is exactly designed for the substitute case: skip loops where memory already explains the token).
- **lesson**: the mechanism story ("memory frees depth, loop adds depth → complementary") was plausible but wrong in the measured regime. Freed depth is only useful if
  there is depth-hungry work left.
- **next**: (1) N2 (E4q vs E4r) becomes *more* relevant: if they are substitutes per token, memory confidence should predict where loops are useless;
  (2) test on a reasoning probe (E8 addition) where depth-hungry work exists; (3) main line stays C4 (no loop), decision unchanged.
