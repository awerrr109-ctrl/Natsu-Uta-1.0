# Inference Specification (v0.1)

| feature | status | file |
|---|---|---|
| incremental decode, GDN recurrent state + conv state + KV cache (sliding/global) | [impl, tested exact vs full forward] | `model.py`, `tests/test_model.py` |
| variable loop count at inference (R=1..∞; loops > trained R reuse last loop params) | [impl] | `Natsu.forward(n_loops=)` |
| loop self-speculative decoding (draft = same weights at r<R) | [impl, tested: output == target greedy] | `generate.loop_speculative` |
| adaptive depth by KL convergence across loops | [impl] | `generate.adaptive_depth_logits` |
| hard per-token loop skipping via learned gate | [impl] | `forward(gate_threshold=)` |
| shared KV across loops for attention layers | [spec, H11.6] | — |
| 4-bit weights (GPTQ/AWQ-class), 8-bit KV | [spec] | — |
| verifier + best-of-n / tree search with loop budget | [spec] | — |

## Compute-accounting rule
Every inference-time method reports **Δquality / Δ(FLOPs per answer)** against the R=1 greedy
baseline, using the analytic `flops_per_token` (counts every linear, the attention score cost at the
actual context, GDN state ops, active MoE experts only, and PKM lookups).

## Serving footprint (analytic, C3)
- weights: 8.93B × 4.5 bits ≈ 5.0 GB.
- per-token state: GDN layers have O(1) state: 12 GDN-type layers × 24 heads × 128×128 × 2 bytes ≈ 9.4 MB per sequence, independent of length.
- KV: 4 attention layers (prelude/coda/core-shared) × 2 × 4 kv-heads × 128 × 2 bytes = 8 KB/token, so 256k ctx ≈ 2.1 GB.
  This holds **only if core attention KV is shared across loops**; without sharing, core attention KV grows by R×.
