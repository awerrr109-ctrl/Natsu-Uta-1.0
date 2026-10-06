# Target analysis: what "Claude-4.8-class at 9B" means (operationalised)

The user target "Claude 4.8 equivalent" is interpreted as **Claude Opus 4.8** (released 2026-05-28).
All Claude numbers below are vendor-reported, collected via a third-party aggregator
(morphllm.com, verified 2026-06-09) → status **confirmed-as-reported**, not independently reproduced.

## 1. Known facts about the target

| item | value | status |
|---|---|---|
| SWE-bench Verified | 88.6% | confirmed (vendor) |
| SWE-bench Pro (vendor scaffold) | 69.2% | confirmed (vendor) |
| GPQA Diamond | 93.6% | confirmed (vendor) |
| OSWorld | 83.4% | confirmed (vendor) |
| Terminal-Bench 2.1 | 74.6% | confirmed (vendor) |
| Context / max output | 1M / 128k | confirmed |
| Price | $5 / $25 per MTok | confirmed |
| Parameter count | — | **unknown** |
| Architecture (dense/MoE, attention type) | — | **unknown** |
| Training tokens / data / RL recipe | — | **unknown** |
| Inference compute per answer ("thinking" budget) | adaptive; tokens hidden | **unknown** |
| Model size class | frontier-scale, very likely ≫9B (price point + capability) | **inferred** (no public evidence of size) |

## 2. Decomposition into measurable axes, with the strongest public 9B as anchor

The anchor is Qwen3.5-9B (HF model card, 2026-03), which already uses a GDN 3:1 hybrid.

| axis | target metric(s) | Opus 4.8 | Qwen3.5-9B | gap | where the 9B gap is plausibly closable [I] |
|---|---|---|---|---|---|
| reasoning / STEM | GPQA-D | 93.6 | 81.7 | −11.9 | test-time compute (loops, verifier, search); RL |
| knowledge | MMLU-Pro / SimpleQA | n/a | 82.5 / n/a | ? | **hardest at 9B** — needs memory layers + retrieval/tools |
| coding (agentic) | SWE-bench Verified / Pro | 88.6 / 69.2 | not reported | large | tool use + long context + search; RL on executable envs |
| coding (competitive) | LiveCodeBench v6 | n/a | 65.6 | ? | verifier/execution feedback, best-of-n |
| instruction following | IFEval | n/a | 91.5 | near-saturated | post-training |
| long context | AA-LCR / LongBench v2 | n/a | 63.0 / 55.2 | ? | hybrid attention + KV-free recurrence |
| tool use / agents | TAU2 / BFCL / OSWorld | OSWorld 83.4 | TAU2 79.1, OSWorld-V 41.8 | −41.6 (OSWorld) | agent RL, environment scale |
| multilingual | MMMLU | n/a | 81.2 | ? | data mixture |
| hallucination | SimpleQA / AA-Omniscience | n/a | n/a | ? | calibrated abstention + retrieval |
| latency / cost | tokens/s, $/task | $5/$25 | open weights | 9B wins | the main Pareto advantage of 9B |

## 3. Feasibility statement (honest)

- [E] A 9B open model is currently 12 points behind on GPQA-D and roughly 40 points behind on OSWorld.
- [I] Parametric knowledge scales with stored parameters (memory-layer and MoE literature; R7). A 9B
  model **will not** hold the same amount of closed-book knowledge as a frontier model. The bottleneck is
  bits stored per parameter (≈2 bits/param capacity estimates from the knowledge-capacity literature,
  to be verified at L3 next session). The way around it is to **not store the knowledge**: retrieval and tools.
- [I] Reasoning depth is *not* bounded by params in the same way (R1–R5: looped models gain manipulation
  ability at fixed params). Test-time compute can trade FLOPs for accuracy.
- **Target definition adopted:** match Opus-4.8-class *task success* on reasoning/coding/agentic axes **with tools
  and retrieval allowed and test-time compute reported**, and dominate it on the cost/latency Pareto axes. A
  closed-book knowledge match is explicitly **not** claimed to be reachable at 9B.
- Falsification: if a C3-style model at the 1B stage fails to beat an iso-param dense baseline on reasoning
  benchmarks at ≤3× inference FLOPs, the core thesis (loops+MoE+memory > dense at iso-param) is rejected.

## 4. Efficiency multipliers: definitions (what "10,000×" can and cannot mean)

| axis | definition | reference | realistic claim range [I] |
|---|---|---|---|
| parameter efficiency | quality per stored param | iso-quality dense model params / ours | 1.5–3× (looping φ≈0.46 gives r^0.46 on the looped share; MoE/memory add more) |
| inference efficiency | quality per inference FLOP | same quality frontier model FLOPs / ours | **100–1000×** is plausible *vs frontier models* because they are ≫9B (unknown size → range only) |
| training compute | FLOPs to reach target | frontier training FLOPs / ours | 100–10,000× vs frontier is plausible *because the target model is enormous*; this mostly measures model-size ratio, not method |
| sample efficiency | tokens to reach target | | 2–10× from distillation + filtering (literature range; to verify) |
| memory | RAM to train/serve | | serve 9B in 4-bit at about 5 GB; *research* loop at 1 GB (demonstrated here at toy scale) |

The 10,000×–100,000× figure is achievable **only** as a ratio against a frontier model's training/inference cost,
and only on axes where the 9B model actually matches it. We report every multiplier with its reference model
and axis; no blended "10,000×" number will be claimed.

## 5. Session-2 measured multipliers (toy scale; each with axis, reference and evidence level)
| axis | reference | measured (toy) | how computed | transfer to 9B |
|---|---|---|---|---|
| parameter efficiency | dense 2× params (E4z 1.616M, 1.4394) | E4j (0.825M, 2-seed 1.4262) beats E4z at **0.51× params** → ≥ 1.96× | iso-quality param ratio (lower bound) | [I] R23/R17 show the memory gain persists across 0.4B–27B; size shrinks with BPE (−0.059 → −0.014) |
| inference FLOP efficiency | E4z (3.61 MFLOP/tok) | E4j 1.58 MFLOP/tok at better bpb → **≥ 2.3×** | iso-quality FLOP ratio (lower bound) | [E analytic] C4 vs C0 at 9B: 3.1× fewer FLOP/token at 4k, 3.2× fewer decode bytes |
| sample efficiency | E4c without KD | E5a matches the quality E4c would reach with more data | needs a data-scaling curve | [E R35] KD gain shrinks with tokens/param, so ≤ 1.x at 9B |
| training compute | E4z | E4j reaches better bpb with 0.44× training FLOPs (7.78e12 vs 1.77e13) → **≥ 2.3×** | iso-quality training FLOP ratio | as above |
| memory (research loop) | — | all runs peak ≤ 850 MB RSS on a 985 MB host | measured | 50M+ stages need GPU; 1GB is a research-loop property, not 9B training |
| loops (depth via compute) | E4c | −0.017 bpb at 1.73× FLOPs (2 seeds) → **< 1× at iso-FLOP** | | negative at toy; R33 suggests positive on reasoning only |

Combined honest statement (toy): **≈ 2× parameter, ≈ 2.3× inference-FLOP and ≈ 2.3× training-FLOP efficiency vs a 2× dense baseline**, all three
from the Engram + MoE design, measured at 0.8M params and one data set. These are 2–3× numbers. The 100–10,000× range in §4 exists only as the
ratio against frontier models' size and remains [I]/unknown.
