# F003 — E1 v3 (two-chain pointer chasing): unlearned within 2.4M-token CPU budget

- **hypothesis**: a 4-layer dense attention model (0.76M params) learns 1–3 hop two-chain pointer chasing within 2500 steps (2.4M tokens, 160k scored answer tokens). This was the learnability gate before testing loops.
- **implementation**: `gen_chain` v3, n_chains=2, seq≤80, AdamW 2e-3, WSD.
- **expected**: hop1 ≥ 0.9.
- **observed**: chance at all hops (final hop1 0.51, hop2 0.49, hop3 0.48, hop4 0.50, hop6 0.49). Loss fell 0.46 → 0.34–0.36, but that is learning the *format* tokens (EOS), not the answer. Mid-training wobble at step 1500 (0.56) did not persist.
- **why it failed** [I]: the shortcut is gone (as intended), so the model must learn induction-style variable binding. Induction-head formation is a known phase transition that needs far more steps at small width. Budget: 1 answer token per ~15 input tokens, so ~160k supervised bits. The prior shortcut tasks (F001/F002) were "learned" quickly *because* they were shortcuts.
- **lessons**: (1) the CPU budget (~5k tok/s) cannot reach the induction phase transition for this task design. (2) A cheaper reasoning probe is needed where every position is supervised: **dense supervision**. (3) Gate passed in the negative direction: the task is now valid, but the budget is not.
- **next hypothesis**: (a) E1v4 = densely supervised variable-lookup (every statement after the first asks for the value: "a=3;b=a→3;c=b→3"). That turns each hop into its own supervised step, a known speedup for learning compositional tasks (process supervision analogue). (b) Defer the k-hop/looping question to the GPU 10M stage. (c) Use TinyStories bpb (E3/E4) as the session-1 primary signal.
