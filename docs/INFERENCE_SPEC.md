# Inference Specification (v0.2)

| feature | status | file |
|---|---|---|
| incremental decode, GDN recurrent state + conv state + KV cache (sliding/global) | [impl, tested exact vs full forward] | `model.py`, `tests/test_model.py` |
| variable loop count at inference (R=1..∞; loops > trained R reuse last loop params) | [impl] | `Natsu.forward(n_loops=)` |
| loop self-speculative decoding (draft = same weights at r<R) | [impl, tested: output == target greedy] | `generate.loop_speculative` |
| adaptive depth by KL convergence across loops | [impl] | `generate.adaptive_depth_logits` |
| hard per-token loop skipping via learned gate | [impl] | `forward(gate_threshold=)` |
| shared KV across loops for attention layers | [impl `loop_kv=shared_first`; E4e2: no measurable loss, 3× less core KV] | `model.py` |
| 4-bit weights (GPTQ/AWQ-class), 8-bit KV | [spec] | — |
| verifier + best-of-n / majority vote with FLOP accounting | [impl `generate.sample_n/majority_vote/best_of_n/tts_accounting`; measured by `tts_eval.py` post-queue] | `generate.py` |
| exit-trained lookahead gate (per-token loop skipping) | [impl; E4p AUC 0.66; 33% skip → +0.011 bpb at 0.86× FLOPs] | `model.py` |
| Engram table offload + address prefetch | [spec; addresses depend only on token ids, so rows can be fetched one token ahead] | — |

## Compute-accounting rule
Every inference-time method reports **Δquality / Δ(FLOPs per answer)** against the R=1 greedy
baseline, using the analytic `flops_per_token` (counts every linear, the attention score cost at the
actual context, GDN state ops, active MoE experts only, and PKM lookups).

## Serving footprint (analytic, C3)
- weights: 8.93B × 4.5 bits ≈ 5.0 GB.
- per-token state: GDN layers have O(1) state: 12 GDN-type layers × 24 heads × 128×128 × 2 bytes ≈ 9.4 MB per sequence, independent of length.
- KV: 4 attention layers (prelude/coda/core-shared) × 2 × 4 kv-heads × 128 × 2 bytes = 8 KB/token, so 256k ctx ≈ 2.1 GB.
  This holds **only if core attention KV is shared across loops**; without sharing, core attention KV grows by R×.

## Session-2 measured trade-offs (toy; see REPORT_S2)
- Per-token loop skipping: th=0.3 → 33% tokens at R1, +0.011 bpb, 0.86× FLOPs; th=0.5 → 81%, +0.044, 0.66×. The gate is weak (AUC 0.66),
  so adaptive depth is not yet a strong inference-efficiency lever. Engram (−0.054 at 1.06× FLOPs) dominates.
- Test-time scaling (greedy vs maj@N vs oracle@N on addition, with FLOPs per problem) is pending (`scripts/post_s2.sh`).
- C4 (no loop) decoding: 1.29 GB moved per token at 4-bit with tables offloaded (analytic, efficiency_9b.md). That is the main latency lever at batch 1.

## Test-time compute allocation policy (v0.2, from R27, R51, R52)
- Budget per question is adaptive: sample in rounds of 4; stop when the majority answer's vote margin ≥ 2 or a verifier passes; cap N at 16 (R52: overthinking).
- Verifiers in order of preference: executable (unit tests, math checkers) > policy-matched PRM trained on own samples > none (R27: cross-policy PRMs fail).
- Training-side consequence (R51): because inference is sampled many times, pretrain C4 in the overtrained regime (≥ 450 tok/param stored; ≈ 2000 tok/param active).
