# F008 — Method: seed noise was assumed rather than measured

- **hypothesis**: toy-run noise is ~0.003–0.007 bpb (assumed from the B-series spread, never measured on the same config).
- **implementation**: seed-1 replicates of E4c/E4j/E4g (queue s2c).
- **expected**: |Δ| ≤ 0.007.
- **observed**: E4c 0.0135, E4g 0.0117, E4j 0.0032.
- **why**: at 1.6M tokens and 800 steps, data order dominates (MemmapLoader seed = data order). Engram runs vary less, plausibly because the table reduces dependence on which local patterns are seen early.
- **lessons**:
  1. Effects < 0.02 bpb on one seed are not claims.
  2. Earlier statements affected: loop gain (−0.030 → −0.017 2-seed); "E4p ≈ E4g"; BPE loop effect (−0.006, within noise); ranking among loop recipes.
  3. Statements surviving: Engram (−0.054, 2-seed), hybrid vs attention (−0.067), KD on E4j (−0.022; Engram spread is small).
- **next**: s2 replicates (third seed) are queued; E5a seed 1 queued (s2h). Reports now give 2-seed means where available.
