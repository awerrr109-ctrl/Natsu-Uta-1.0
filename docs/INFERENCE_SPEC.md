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

## 1 GB-RAM decoding: expert paging from flash (v0.3, from R63/R64 + own routing data)
**Evidence**
- [E R63, Routide arXiv:2609.29032] Qwen3.6-35B-A3B on an iPhone with experts paged from storage:
  - An expert payload is 1.69 MiB, read as 27 aligned 64 KiB units.
  - At a 512 MiB budget, LRU gets **0% hits**, a "capacity cliff" set by the minimum distinct-expert reuse distance of 312 experts. Random replacement gets 18.8%, and decode runs at 2.2 tok/s.
  - At 576 MiB, LRU gets 38.6% and 3.1 tok/s. At 1 GiB, a recency-frequency hybrid gets 49.8%, against an offline optimum of 68%.
  - A next-step expert predictor is 61% accurate.
- [E R64, llama.cpp discussion #27149 / tinygiant, a low-star prototype] Qwen3-30B-A3B on an M1 with 16 GB, using ~2 GB RAM:
  - Expert-contiguous re-layout is needed, because stock GGUF interleaving touches 111× the pages.
  - SSD streaming reaches 3 GB/s.
  - Decode rate by cache hit rate: 2.4 tok/s at 50%, 3.8 at 75%, 4.1 at 88%. At ≥88%, reads are fully hidden behind CPU compute.
  - Calibrated pinning from 10 tokens of text gives 42% hits; plus LRU, 56%.

**Natsu-specific consequences** [I]
1. **Loops make routing collapse useful for paging.** route_diag shows loop r re-uses 55–88% of loop 0's experts for the same token.
   In a looped core, the loop-1..R−1 expert reads are therefore mostly **cache hits by construction**. Expert I/O per token is ≈ 1 loop's worth, while compute is R loops.
   - This flips the earlier verdict: routing collapse is bad for *capacity* (R47) but good for *I/O under 1 GB*.
   - Design rule: decide per deployment. When RAM-bound, keep collapse (shared routers) and loops buy depth at almost no extra flash traffic. When compute-bound, use per-loop routers or LoRA (E4y/E4y3).
2. **Engram rows are prefetchable one token ahead** (addresses depend on ids only). Tables can live on flash with no stall if a row read (dm·nh·2 B ≈ 2 KB) finishes within one token's compute.
3. **The capacity cliff is the binding constraint**: the cache must exceed the reuse distance (#experts touched between reuses). For C4 at 9B (64 experts × 24 MoE layers, top-k=6), one token touches 144 experts.
   With ~2.3 MiB per expert at 4-bit, the hot set for one token is ≈ 330 MiB, which is inside 1 GB. A cache of 2–3 tokens' worth (~700–1000 MiB) is borderline, so the cache budget must leave room for the KV/GDN state (9.4 MB + 8 KB/token).
   → Analytic, not measured. **Measurement target for the 1B stage**: expert reuse-distance histogram and hit rate vs budget, on our own routing traces (route_diag can dump them).

## Tool-verifier policy (v0.3, from R68)
- Two-stage selection: (1) an executable check acts as a **soft** filter. It vetoes only on hard failures (exception, type error, wrong format); a mismatch on an ambiguous check downweights instead of rejecting.
  (2) A policy-matched scorer or majority vote ranks the survivors.
- Reason: R68 reports unrecoverable false negatives from hard tool rejection, and a symbolic-equivalence failure.
- Expected effect (R68, math): 1B + tool + N=64 > 8B. That is the strongest published size-class multiplier for verifiable domains. It does not transfer to non-verifiable tasks.
