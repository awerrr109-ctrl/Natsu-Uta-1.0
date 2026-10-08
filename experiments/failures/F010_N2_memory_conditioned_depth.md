# F010 — N2 "memory-conditioned depth decider" shows no effect (pending AUC)

- **hypothesis (N2, docs/NOVELTY.md)**: the Engram gate value (how much the n-gram memory "explains" a token) predicts where extra loops are useless, so it should improve the per-token depth decider.
- **implementation**: `gate_mem_feat=True` appends the Engram gate scalar to the lookahead-gate input (E4q); identical run without it (E4r).
- **expected**: better skip/quality trade-off for E4q at the same skip fraction, and higher gate AUC.
- **observed**: E4q 1.4367 @39% skip vs E4r 1.4356 @42%; full-depth R3 1.4229 vs 1.4223. No difference (E4r is marginally better).
- **why (I)**: (1) F007 already showed loop and Engram are substitutes, so after Engram the loop gain is ~0 for nearly all tokens. There is nothing left
  for a decider to discriminate. (2) The hidden state already contains the Engram contribution (it is added to the residual), so the scalar is redundant.
- **lessons**: a decider feature cannot help when the decided quantity (loop gain) is ~0. Test the existence of the effect first, then the router.
- **next**: N2 is parked. It only makes sense where loops have a real per-token gain (reasoning probes E8g/h, if positive).
