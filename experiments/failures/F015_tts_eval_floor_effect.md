# F015 — TTS evaluation at 5 digits hit a floor: no test-time-scaling signal

- **intended**: an accuracy-vs-FLOPs curve (greedy@R, maj@N, oracle@N, adaptive-N) for the E8 addition models (principle 13: gain per extra compute).
- **observed** [E]: greedy accuracy is 0 for all six models at 5 digits, and oracle@16 is ≤0.05. Training-time EM (0.30–0.48) was averaged over 1–5 digits (min_digits=1).
  By digit count, E8e scores 0.40/0.46/0.19/0 for 1/2/3/5 digits.
- **why**: the eval took the training *max* difficulty as its only point; the training metric hid that the models fail almost entirely at the hardest length.
- **fix**: tts_eval `--digits 2` and `--digits 3` rerun (post_s2b, after s2j). Rule: before running a TTS curve, check that greedy accuracy is in [0.1, 0.8] for the base model.
- **lesson**: an aggregated training metric over a difficulty mixture says nothing about any single difficulty level. Pick eval difficulty from per-level accuracy.
