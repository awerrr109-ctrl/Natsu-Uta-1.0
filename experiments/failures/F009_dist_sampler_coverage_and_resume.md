# F009 — Distributed sampler: with-replacement coverage and resume re-seeding (found by reading a third-party issue)

- **hypothesis (implicit)**: per-rank random windows (`rng(seed*1000+rank)`) are "good enough" sharding; resuming with `seed=start` is fine.
- **implementation**: `train_dist.ShardedTokens` (session 2), verified only for "runs and resumes" on CPU gloo ×2.
- **expected**: each epoch covers the corpus.
- **observed (analysis, no large run done)**: sampling with replacement covers only 1−1/e ≈ 63% unique windows per epoch-equivalent, and ranks overlap.
  This is exactly the failure reported in sandyresearch/parcae issue #10 (~36% never read, ~36% read twice). Resume re-seeded with `seed=start`,
  which draws a new order, so pre/post-resume data are not a single epoch.
- **why**: the test checked mechanics (loss decreases, resume loads) but not **data coverage**, which is invisible in a loss curve.
- **fix**: epoch-exact permutation shared by all ranks plus a rank-strided slice, with saved `(epoch, cursor)`. Unit test `test_sharded_tokens_epoch_exact`
  asserts every window is seen once per epoch across 4 ranks and that resume is exact. It runs as the s2j queue gate.
- **lessons**: (1) issues of *other* projects are a source of bug classes for our own code (principle #4: "Issueに重要情報があるrepo").
  (2) Data-pipeline tests must assert coverage, not just "it runs".
- **affected results**: none. All toy results use the single-process MemmapLoader (random windows with replacement; for ≤1 epoch of 118M tokens at
  1.6M tokens/run, the coverage issue is irrelevant there). The 9B training spec relies on train_dist, so the fix matters for scale.
