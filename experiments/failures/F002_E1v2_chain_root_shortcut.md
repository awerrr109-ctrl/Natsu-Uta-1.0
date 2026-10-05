# F002 — E1 v2 (first attempt): pointer-chasing task had a hop-independent shortcut

- **hypothesis**: same as F001 (looping helps k-hop reasoning), with the easier task variant.
- **implementation**: single chain, 2 distractors, hops 1–3, per-hop eval.
- **expected**: hop1 > hop2 > hop3 > hop4 > hop6 accuracy for a dense 4-layer model.
- **observed** (step 500): hop1 0.80, hop2 0.79, hop3 0.84, hop4 0.80, hop6 0.81. Flat again, but much higher than chance.
- **why it failed**: **task-design bug.** With one chain, the answer is always the digit assigned to the chain root,
  and the root is identifiable without following the chain (it is the digit-valued variable referenced by another
  variable). Distractors sometimes also point at the root. The model learned a 2-step heuristic that is exact for any hop count.
- **lessons**: (1) synthetic reasoning tasks must be *adversarially checked for shortcuts* before training: write
  down the simplest heuristic that could solve them and make sure it doesn't; (2) "flat accuracy across difficulty"
  is a reliable alarm, now confirmed twice (F001, F002); (3) killing the run early saved about 10 minutes of CPU.
- **fix**: `gen_chain` v3 builds `n_chains=2` disjoint equal-length chains with distinct root values and queries one tail.
  The best hop-independent strategy is now a 50% guess between roots.
- **next**: rerun E1v2 a/b/c/d with v3 task.
