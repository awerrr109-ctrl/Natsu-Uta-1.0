# F017 — 10M stage OOM-killed on the 985 MB sandbox; the picker ran on an incomplete rule

- **observed**: s2i launched all four 10M runs (L10a–d); each was SIGKILLed after model build, at the first step (batch 8, seq 256).
  The retry at batch 4 × accum 2 was also killed (3-step probe).
- **measurement** (`scripts/mem_profile.py`, one fwd+bwd step, L10a, threads 1):
  | setting | build | fwd peak | bwd peak |
  |---|---|---|---|
  | batch 4, dense MoE | 199 MB | 660 | 710 |
  | batch 4, token-dispatch MoE | 267 | 681 | 729 |
  | batch 2, token-dispatch MoE | 267 | 483 | 518 |
  | batch 4, GDN chunk 64 | 267 | 682 | 697 |

  Activations dominate: ~115 MB per sample of 256 tokens at d=320 with 6 layers. The dense-einsum MoE is *not* the cause: token dispatch saves nothing at E=8, k=2.
  Adam state (≈160 MB) plus the eval batch push batch 4 over the ~800 MB usable.
- **fix**: batch 2 × accum 4 (same tokens/step) and eval batch 2. Token-dispatch MoE (`moe_sparse`) is kept as an option (exactness-tested), since it matters at larger E.
  Future option, already listed in principle 9: activation recomputation (checkpoint the GDN chunk loop).
- **picker deviation**: `pick_engram_recipe.py` chose **v1_lr20** (3 seeds, best LM) although it had no ICL arm. An ICL arm was "required only when defined", which is a loophole.
  The rule is now "ICL arm required": an E6Lj ×2 check runs before the 10M stage. If E6Lj is < 2/2, the 10M Engram configs revert to v1_lr5.
- **lessons**: (1) run a memory probe of the exact config before queueing a new scale; the 0.8M runs never exceeded 600 MB, so this was never tested.
  (2) Decision rules with optional clauses get gamed by missing data. Make required evidence explicit.
- **probe at batch 2 × accum 4** (3 steps, incl. eval): passes, **peak RSS 820 MB** at 619 tok/s, so ~85 min per 10M run. The margin to the 985 MB limit is small, so nothing else may run in the foreground during s2t (fg_guard enforces this).
  If any 10M run is killed, the next step is activation checkpointing of the core blocks (≈ −40% activation memory), not a smaller batch.
- Note: `grad_ckpt=True` (per-block activation recomputation) already exists in model.py and was not enabled for L10. It is the documented 1 GB-RAM mode (principle 9).
  Caveat found in code review: with loops + `loop_kv=shared_first`, the checkpoint branch skips setting `blk.mix.last_kv`, so it is safe only for non-looped configs (all L10 cells). To fix before any looped 10M+ run.
  Planned measurement: peak RSS at batch 4 with grad_ckpt (mem_profile) once s2t is done; if it is below 600 MB, later stages use batch 4 + grad_ckpt (≈1.3× compute for 2× batch).
