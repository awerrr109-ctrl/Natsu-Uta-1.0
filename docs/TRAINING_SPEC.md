# Training Specification (v0.1)

Status: specification for the 1B and 9B stages; toy-stage parts are implemented and running
(`src/natsu/train.py`). Items marked [impl] exist in code; [spec] are specified but not run here.

## 1. Objective schedule (single objectives are not assumed optimal)
| phase | tokens (9B) | objective | rationale |
|---|---|---|---|
| P0 warmup | 0–2% | NTP, R=1 only | stabilise prelude/coda before loops ([I] loop training is unstable early, cf. DeepLoop L1) |
| P1 main | 2–80% | NTP + 0.3·MTP + loop sampling R~U{1..3} [impl] + shallow→deep KL self-distillation 0.5 [impl] + logit distillation from teacher on 30% of batches [spec] | random R makes every depth a valid model (Huginn); KD is the main sample-efficiency lever at ≤9B (Gemma 2/3, Minitron; L1, to be read at L3) |
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
| component | share P1 | notes |
|---|---|---|
| web (FineWeb-Edu-class classifier-filtered + DCLM-style fastText filter, MinHash dedup) | 45% | quality > quantity; dedup at document and paragraph level |
| code (permissively licensed, dedup, execution-filtered where possible) | 20% | |
| math & STEM (incl. synthetic rephrasings) | 15% | rephrasing web (WRAP-style) gives 3× token efficiency [L1 claim, to verify] |
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
| 8-bit / 4-bit optimizer state | [spec] | |
| forward-only (MeZO / forward gradients) | rejected for pretraining [I]: gradient variance grows with dimension (refute list in taxonomy H6.4); kept only for fine-tuning experiments |

## 5. Curriculum
- Loop curriculum: R_max ramps 1→3 over P0–P1 [spec]; sampling R~U{1..R_max} [impl].
- Context: 4k → 32k → 256k (GDN layers need no RoPE; attention layers use RoPE θ=1e6) [spec].
