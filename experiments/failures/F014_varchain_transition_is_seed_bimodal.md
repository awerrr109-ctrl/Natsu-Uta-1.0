# F014 — The varchain retrieval phase transition is seed-bimodal; a 1-seed "huge effect" was over-claimed

- **claim being checked**: H14.1(a) "Engram prevents the in-context retrieval circuit". Seed 0: E6La EM 0.844 vs E6Lb 0.000, called "far beyond any seed effect".
- **observed** [E own]: **E6La seed 1 (no Engram, same config)**: final answer loss 0.773, seq EM **0.008**, long20 EM 0.000. It never crossed the transition in 2400 steps
  (train loss 0.85 @1600, 0.75 @2200; seed 0 was 0.57 @1200 and 0.038 @2400).
- **why the earlier claim was wrong**: on a task with a sharp phase transition, the outcome per seed is ~binary. The seed-to-seed spread is then the *whole*
  range (0 ↔ 0.84), so a single seed cannot separate "Engram blocks the transition" from "this seed did not transition in budget". F008 measured seed noise
  on bpb (a smooth metric) and that number was wrongly carried over to a bimodal metric.
- **current status of H14.1(a)**: E6La transitions 1/2, E6Lb 0/1 (s1 running). **Not established.** The answer-loss comparison is also weak: E6Lb_s0 0.891 vs E6La_s1 0.773.
- **update (E6Lb seed 1 done)** [E own]: **E6Lb s1 (with Engram) transitioned fully**: answer loss 2e-5, seq EM **1.000**, long20 EM **1.000**, train loss 0.05 already at step 1000
  (earlier than either E6La seed). Transitions: **no-Engram 1/2, Engram 1/2**. **H14.1(a) is refuted at this scale** (best seed of each arm: Engram 1.000 ≥ 0.844).
  If anything, the Engram seed that transitioned did so faster and generalised to long20 better (1.000 vs 0.781). With n=2 per arm that is not a claim.
- **fix (method)**: on transition tasks, report (#seeds transitioned / #seeds, median steps-to-transition via `steps_to`) and use ≥3 seeds per arm before any claim.
  Seed-2 replicates of E6La/E6Lb added to s2m. The F013 paper-faithful E6Le is still worth running (it fixes a real divergence regardless).
- **lessons**: "effect far beyond seed noise" must be justified with seed noise *measured on the same metric*. For thresholded/bimodal metrics, count outcomes.
