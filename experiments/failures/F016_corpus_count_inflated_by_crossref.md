# F016 — Corpus size overstated: raw paper count includes off-topic crossref hits

- **claim being checked**: "harvested 49,665 papers (≥10k target met)" (principle #2/#28).
- **observed** [E, computed from research.sqlite on 2026-10-08]:
  | source | rows | loose ML-keyword titles |
  |---|---|---|
  | arXiv | 8,872 | 6,586 (74%) |
  | OpenAlex | 17,424 | 12,663 (73%) |
  | crossref | 23,369 | 13,434 (57%) |

  Strict LM-specific titles (language model / transformer / LLM / attention / pretrain / token / MoE / quantiz / distillation / state space / …): **19,174**.
  Auto-relevance > 0.5 (L1): **8,551**. Read at L2/L3: **56** in the reads table, plus R56–R69 in the notes; the table lags (see below).
- **why**: crossref full-text search is lexical and returns, e.g., "Embedding in the Cardiff Grammar" for "sparse embedding learning rate".
  In this run arXiv was rate-limited (429s; the harvester backs off after 3 failures) and OpenAlex hit its daily budget, so P16 queries got crossref only.
- **honest numbers to report from now on**: harvested 49,665 raw / **≈19k strictly on-topic** / 8.5k L1-relevant / ~70 read. The ≥10k on-topic target is met (19k), but not by the raw count's margin.
  Reading depth (~70) is the real gap, as already stated in REPORT_S2 §1.
- **fix**: (1) report the strict on-topic count next to the raw count (added to corpus_stats); (2) re-run P16 on arXiv/OpenAlex next session when the budgets reset (queries_done marks crossref only, so they will run);
  (3) sync the reads table with the evidence log (R56–R69 are missing from `reads`).
