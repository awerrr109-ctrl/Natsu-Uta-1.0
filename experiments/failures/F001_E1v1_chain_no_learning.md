# F001 — E1 v1: k-hop pointer chasing never learned (all models ≈ 50%)

- **hypothesis**: H1.1 / H4.1 — looped weight-shared core (2 layers × R loops) matches a deeper dense model on k-hop variable dereferencing and extrapolates to more hops with more loops at test time.
- **implementation**: `gen_chain` (letters=vars, digits=values, hops 1–4, 4 distractors, statements shuffled); d=128, 1000 steps × 32 seqs (~1M tokens, ~64k scored answer tokens), AdamW 2e-3 WSD. Configs `E1a_dense4_attn`, `E1b_dense8_attn`.
- **expected**: dense8 > dense4 on hops 3–4; both degrade on OOD hops 6/8.
- **observed**: dense4 acc 0.50 / hop6 0.48 / hop8 0.51; dense8 0.49 / 0.52 / 0.57. **No hop dependence** → models use a shortcut (choose among the 3–4 digits in the prompt), not pointer chasing. Evaluating untrained loop counts (loops>1 on a non-looped model) monotonically hurts (dense8 at 8 loops: 0.34) — expected, recorded as a sanity check that the loop path is live.
- **why it failed**: (1) training budget far below the phase transition for induction/pointer-chasing (known to require many steps; loss plateaued at ~0.5 by step 700); (2) the eval metric aggregated all hops so we could not see partial learning; (3) shuffled statements + distractors that themselves point into the chain make 1-hop already a 2-step retrieval.
- **lessons**: ALWAYS evaluate per difficulty level; verify a baseline *can* learn the task before comparing architectures (a "learnability gate"); a flat-across-difficulty accuracy is the signature of a shortcut.
- **next hypothesis**: E1 v2 = curriculum-free easier variant (2 distractors, hops 1–3 train), per-hop eval (1,2,3 in-dist; 4,6 OOD), 2.5× steps. Gate: dense4 must exceed 90% at hop1 before loop comparisons are interpreted.
