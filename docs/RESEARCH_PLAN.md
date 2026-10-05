# Natsu-Uta Research Plan (Session 1 foundation)

## 0. Objective (operational definition)
Build a ~9B-parameter system (stored parameters ≤ 9.5B incl. embeddings) that maximises
practical capability per unit of {training compute, training tokens, inference FLOPs, RAM},
and **measure** where it sits on the quality/cost Pareto frontier versus public models.

"Matching a frontier model" is decomposed in `docs/TARGET_ANALYSIS.md`; it is a set of
axis-wise targets with confidence labels, not one number.

## 1. Problem decomposition
Capability of a fixed-parameter model ≈ f(knowledge stored, knowledge *manipulation*
depth, context access, test-time compute, post-training elicitation). Each maps to a problem:

| id | problem | bottleneck it attacks | taxonomy key |
|---|---|---|---|
| P1 | capability per stored parameter | params | `P1_param_efficiency` |
| P2 | FLOPs where needed | compute | `P2_conditional_compute` |
| P3 | long context sub-quadratically | memory/compute at long L | `P3_sequence_mixing` |
| P4 | convert inference compute into accuracy | fixed params | `P4_inference_time_scaling` |
| P5 | capability per training token | data | `P5_data_efficiency` |
| P6 | train cheaply / in 1 GB | compute + RAM | `P6_training_efficiency` |
| P7 | offload knowledge/compute (retrieval, tools) | params | `P7_memory_retrieval_tools` |
| P8 | post-training elicitation | sample efficiency of RL | `P8_post_training` |
| P9 | deployment compression | RAM/latency | `P9_inference_compression` |
| P10 | cross-field principles | blind spots | `P10_cross_field` |

## 2. Search strategy (Step A–H loop)
Every query lives in `research/taxonomy.py` under (problem → hypothesis → support|refute).
`research/harvest.py` runs it against arXiv, OpenAlex, and GitHub (star-stratified into
5 buckets: 0–1, 2–10, 11–100, 101–1000, >1000, sorted by recency, so low-star/new repos
are deliberately included). Each hit stores provenance (problem, hypothesis, kind, query).
After each cycle, `research/analyze.py` produces the L2 reading queue; reading notes go to
`research/notes/evidence_log.md`; new hypotheses/queries are appended to the taxonomy
(Step H) — the DB dedups so re-runs are cheap and nothing is re-read.

## 3. Evidence levels (honesty contract)
| level | meaning |
|---|---|
| L0 | harvested metadata |
| L1 | auto-scored relevance from title/abstract (NOT reading) |
| L2 | abstract/README read and annotated by the researcher |
| L3 | full paper / code inspected (incl. paper↔code mismatch check) |

Claims in reports are tagged **[E]** evidence (cite L2/L3 source or own experiment id),
**[I]** inference, **[H]** hypothesis. Comparison-target facts are tagged
confirmed / inferred / unknown.

## 4. Literature / repo evaluation criteria
1. scale of evidence (params, tokens) — toy-only results are down-weighted for 9B decisions;
2. controlled comparison (iso-FLOP / iso-param / iso-token?) — uncontrolled claims flagged;
3. existence of negative/ablation results; 4. reproducibility (code, independent replication);
5. hardware realism (does it map to dense matmuls?); 6. relevance to 9B / 1 GB constraints.

## 5. Experiment ladder
toy (≤1M, CPU, synthetic+TinyStories) → 10M → 50M → 100M → 300M → 1B → 9B.
Only toy and (later) 10M fit this sandbox (2 vCPU, ~1 GB RAM). Larger stages are fully
specified (configs + launch scripts) but require GPUs; that boundary is stated, not hidden.
Each run logs loss, bpb, exact-match acc, gnorm, tok/s, RSS/peak RSS, analytic train FLOPs.

## 6. State persistence
`state/PROJECT_STATE.json` (machine-readable) + `state/PROJECT_STATE.md` (human).
Failed experiments: `experiments/failures/*.md` (hypothesis / implementation / expected /
observed / why / lessons / next). Research DB: `research/db/research.sqlite` (committed).
