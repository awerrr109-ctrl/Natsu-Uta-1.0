# F012 — Loop comparisons at fixed step budgets are confounded by delayed phase transitions

- **hypothesis (R33, R36)**: loops help algorithmic tasks, especially once positional bottlenecks are removed (Abacus).
- **implementation**: E8b/E8d/E8h (loops) vs E8a/E8c/E8g (no loop), 1500 steps. E6c/E6d vs E6a, 1200 steps.
- **expected**: loop ≥ no-loop, and loop > no-loop on 6-digit length generalisation.
- **observed**: every looped run is worse in-distribution. E8h (loop + Abacus) is at 0.066 acc vs 0.430 without the loop. The train curves show looped
  runs sitting on the ~1.5 loss plateau longer (E8h until step ~1250 vs E8g ~750; E6a transitioned between 800 and 1200 while E6c/E6d were slower).
- **why (I)**: an iteration-shared block must learn a computation that works at every pass. The gradient signal through 3 applications of the same
  weights is noisier/conflicting early, so the circuit forms later. With a fixed step budget, the comparison measures *when* the transition happens, not final capability.
- **lessons**: (1) For algorithmic probes, compare at convergence or report steps-to-threshold (iso-loss), not loss at a fixed step.
  (2) Toy-scale claims "loops don't help reasoning" are downgraded to "loops learn algorithmic tasks more slowly at fixed small budgets".
  This is still decision-relevant (training efficiency), but not a capability verdict.
- **next**: add a `steps_to_acc` metric (first eval step with acc ≥ threshold) to final.json; rerun E8b/E8h at 4× steps only if the 50M ladder keeps loops alive.
