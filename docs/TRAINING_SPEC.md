# Training Specification (v0.2)

v0.2 changes (session 2): data section rewritten from R30–R32 (filtering fraction law, ensemble classifiers, Goodhart guard);
logit distillation is now [impl] (DPT, R25); synthetic data is quantified (R26); 8-bit optimizer state is [impl] with a measured footprint;
the loop-sampling curriculum options are [impl]. Evidence ids refer to `research/notes/evidence_log.md`.

Status: specification for the 1B and 9B stages; toy-stage parts are implemented and running
(`src/natsu/train.py`). Items marked [impl] exist in code; [spec] are specified but not run here.

## 1. Objective schedule (single objectives are not assumed optimal)
| phase | tokens (9B) | objective | rationale |
|---|---|---|---|
| P0 warmup | 0–2% | NTP, R=1 only | stabilise prelude/coda before loops ([I] loop training is unstable early, cf. DeepLoop L1) |
| P1 main | 2–80% | NTP + 0.3·MTP + loop sampling R~U{1..3} [impl] + shallow→deep KL self-distillation 0.5 [impl] + logit distillation from teacher, DPT-style: the 15% lowest-teacher-entropy tokens get hard-label CE only (`skip_low_entropy=0.15`), T=1 [impl `train.distill`; R25]. Toy: E5a −0.035 / E5b −0.022 bpb, students beat teacher. **R35: gain is an under-trained-regime effect; at 9B (~500 tok/param) KD is used only with an existing (sunk-cost) teacher and mainly for post-training/reasoning traces** | random R makes every depth a valid model (Huginn); KD is the main sample-efficiency lever at ≤9B (Gemma 2/3, Minitron; L1, to be read at L3) |
| P2 anneal | 80–100% | NTP on high-quality + reasoning traces; WSD 1−sqrt cooldown [impl]; depth-gate FLOP penalty switched on [impl] | annealing with HQ data (MiniCPM, DCLM; L1) |
| P3 SFT | — | masked answer-only loss [impl: `masked_ce`] | |
| P4 RL | — | GRPO with verifiable rewards (math/code exec/tool env), loop count as an action-budget | [E-contested] RLVR may only sharpen base-model distribution ("Does RL really incentivize…", L1); hence RL is placed last and measured with pass@k, not pass@1 only |

## 2. Optimizer and parametrisation
- Muon for 2-D matrices (incl. MoE expert tensors, flattened) + AdamW for embeddings/norms/gains/biases [impl `optim.py`]. Moonshot RMS-matching factor 0.2·sqrt(max dim) [impl].
- [E] Muon keeps one state buffer vs Adam's two (33% optimizer-RAM saving on matrices).
- Schedule: WSD with 1−sqrt cooldown, 20% decay [impl]. WSD lets stable-phase checkpoints be branched into multiple cooldowns, which saves research compute.
- Hparam transfer: width/depth µP-style transfer across the toy→1B ladder [spec]; loops count as depth for residual scaling (output projections init std / sqrt(2·n_eff)) [impl].
- MoE balancing: aux-loss-free bias update (DeepSeek-V3) [impl `MoE.update_balance`].
- No truncated BPTT through loops ([E] R2: degrades φ).

## 3. Data (information density first)

### 3.0 Filtering rules (v0.2, from R30–R32)
- [E R32] Optimal kept fraction of web scales as F_opt ∝ C^0.25 (3% at 1e20 FLOPs → ~30% at 1e23). Natsu-9B pretraining is
  ≈ 6·2.6e9 active·(4–6)e12 ≈ 6e22–9e22 FLOPs → **keep ~25–30% of the web pool**, not the ~10% of FineWeb-Edu/DCLM.
  [I] Aggressive filtering is right for the toy/10M–300M ladder stages (small C) and wrong for the 9B run; the ladder must therefore
  re-tune F per stage rather than transfer one filter threshold.
- [E R30] Ensemble several quality classifiers (fastText-DCLM + edu-classifier + own-teacher-scored) and bucket by score; ensembling raised the HQ share 9→25%.
  Do not apply heuristic filters to the top bucket (+18% yield).
- [E R30, R26] Low/mid buckets are **rephrased** by a teacher rather than dropped (rephrase 1.48×, megadocs 1.80× data efficiency; ~30% synthetic optimum).
- [E R31] Classifier filtering mostly removes the bad and aligns style to benchmarks; it can *raise* LM loss on HQ text. [E R32] Benchmark-targeted
  selection Goodharts. → **Guard**: a held-out evaluation suite is frozen *before* any data selection and is never used for selection,
  classifier training, or mixture tuning. Selection may only use the "dev" suite; reported numbers come from the held-out suite
  (`eval_harness.py` takes the suite list from config, so the split is enforced by config review, not by code).
- Dedup: MinHash at document level plus exact paragraph dedup; up to ~4 epochs over the HQ bucket is acceptable (R30: diminishing after ~4).

### 3.1 Mixture (P1)
| component | share P1 | notes |
|---|---|---|
| web (ensemble-classifier buckets; top ~25–30% kept raw, lower buckets rephrased; MinHash dedup) | 45% | of which ≈ 1/3 rephrased (R26, R30) |
| code (permissively licensed, dedup, execution-filtered where possible) | 20% | |
| math & STEM (incl. synthetic rephrasings) | 15% | WRAP 3× claim not reproduced in R26 (1.48× rephrase / 1.80× megadocs); we budget with R26 numbers |
| synthetic reasoning traces from teacher, verified by executor/checker | 10% | only verified traces (avoid model-collapse dynamics, L1 refute list) |
| multilingual | 10% | |
- Tokens: 9B target ~6–8T tokens; overtraining ~40× Chinchilla, like current 9B models (Qwen3.5 unknown, Ouro 7.7T).
- Tokenizer [spec]: byte-fallback BPE, 131,072 vocab, trained on the mixture; byte tokenizer [impl] used for toy stages.

## 4. Memory-efficient training (1 GB research mode) — measured at toy scale
| technique | status | measured effect |
|---|---|---|
| memmap token storage, streaming encode in 4 MB chunks | [impl] | 118M tokens encoded at constant RAM in 8.8 s |
| trim synthetic padding to batch max | [impl] | step time 3 s → 0.3 s, RSS 800 → 330 MB (E1 smoke) |
| glibc arena limits (MALLOC_ARENA_MAX=1) | [impl] | RSS 807 → 326 MB on the same run |
| activation recomputation per block (`grad_ckpt`) | [impl] | at d=128 toy: +250% time, no RSS gain (allocator-dominated); expected to matter only at larger d |
| Muon (1 state) vs AdamW (2 states) | [impl] | −1 fp32 copy of matrix params |
| block-wise 8-bit Adam (int8 m, uint8 sqrt-v, 256-element blocks, fp32 scales) | [impl `optim.Adam8bit`] | state 8,320 B vs 32,768 B fp32 for a 64×64 matrix (3.9× less); quadratic-loss test reaches the same tolerance as AdamW |
| 4-bit optimizer state | [spec] | |
| forward-only (MeZO / forward gradients) | rejected for pretraining [I]: gradient variance grows with dimension (refute list in taxonomy H6.4); kept only for fine-tuning experiments |

## 5. Curriculum
- Loop curriculum: R_max ramps 1→3 over P0–P1 [spec]; sampling modes fixed / uniform / poisson / poisson_R / fixed_then_uniform [impl `sample_loops`].
  [E own, toy] fixed R=3 (E4g 1.4574) beat uniform R (E4b 1.5748, E4h 1.5291) at matched steps; fixed_then_uniform (E4m) and poisson_R (E4n)
  are the queued tests of whether depth robustness can be bought without that cost (F005).
- Context: 4k → 32k → 256k (GDN layers need no RoPE; attention layers use RoPE θ=1e6) [spec].
