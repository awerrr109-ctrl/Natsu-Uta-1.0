# Session 2 Report (consolidated; queue still running, pending items marked)

Scope: CPU sandbox (2 cores, 985 MB RAM). Toy models of ~0.8M parameters on TinyStories bytes, plus synthetic probes (addition, variable chains).
Evidence tags: [E] measured or cited, [I] inferred, [H] hypothesis. Read IDs refer to `research/notes/evidence_log.md`, failure IDs to `experiments/failures/`.

## 1. Search accounting (principle #28)
- Harvested (L0): **42,045 papers, 15,597 repos**, 1,339 query×source pairs. A Step-H expansion (P14) was derived from this session's anomalies.
- Read at L2 / L3-partial: **50 sources** (R1–R50), including GitHub *issues* for the first time (R40 Engram#20, R41 parcae#8/#10) and a 51★ repo (R39).
- Honest gap: harvesting breadth meets the 10k target; reading depth (50) does not. Four directly relevant 2026 papers (R43–R46) sat unread in the corpus until a targeted check. That check is now a rule before any novelty claim.

## 2. What is established (3 seeds or effect ≥ 3 sd)
| claim | evidence |
|---|---|
| Engram vs iso-param MoE: −0.053 bpb (t ≈ 12.8), 4.4× lower seed variance | E4c/E4j ×3 seeds |
| Seed spread at toy scale is 0.003–0.014 bpb; single-seed deltas < 0.02 are not claims | F008 |
| Unique depth ≫ looped depth on addition at iso-FLOP (0.484 vs 0.285 acc) | E8e vs E8b (1 seed, effect ≫ noise) |
| Abacus digit positions help non-looped addition (+0.13 acc) | E8g vs E8a |

## 3. What is suggestive (1 seed, effect 1–3 sd)
- KD: E5a −0.035 / E5b −0.022 (on top of Engram). A born-again same-size teacher (E5c) gives ~79% of the gain, so at toy scale KD acts as regularisation, not capacity transfer.
- Loop: 3-seed −0.013 (t ≈ 2) at 1.73× FLOPs. LTI reinjection E4o has the best single-seed R3 (1.4535) but fails its pre-registered R6 test.

## 4. Refuted or not supported (pre-registered tests)
| hypothesis | result | file |
|---|---|---|
| N1 memory-before-loop complementarity (bpb) | sub-additive; substitutes | F007 |
| N2 Engram signal as depth-router feature | no effect | F010 |
| N3 fixed→uniform loop curriculum | failed | F005 upd. 3 |
| LTI improves depth extrapolation (R28) | failed (R6 worse) | F005 upd. 5 |
| R36 "with positions fixed, loops help" | failed at our budget (E8h 0.066) | F012 |
| H13.7 block-cosine "loop bias without loops" | no LM effect, −0.066 acc on addition | evidence log |

## 5. Corrections I made to my own earlier claims
- Session-1 loop gain −0.030 → 3-seed −0.013 (F008).
- "C4 ≈ 4–6B dense" was unsupported. R42 (joint MoE laws) supports iso-total MoE ≥ dense given more tokens.
- E5a "beats teacher" was false. Only E5b does, and only because Engram alone already does.
- Expected Engram gain at 9B is ~0.004 bpb (R17 allocation law), not the 0.053 toy number.
- "Loops don't pay" is scoped to "toy scale, our recipe, fixed short budgets". R47/R48 report iso-FLOP gains at 0.7–1.7B with per-loop routers and residual scaling. F012 shows that looped models enter algorithmic phase transitions later at fixed steps.

## 6. Bugs found and fixed (all with regression tests)
- **F009**: the distributed sampler sampled with replacement (~63% coverage) and resume re-seeded the data order. Found by reading parcae#10.
- **F011**: the Engram v2 VIP key builder used the opposite n-gram order from the model, which invalidated E4s. The rerun E4s2 is queued.
- **F006 ×2**: my own foreground jobs OOM-killed queue runs. Now prevented by a mechanism (`scripts/fg_guard.sh`), not a written rule.

## 7. Design state (ARCHITECTURE §7, NEXT_GEN)
- **Main line C4**: 3:1 GDN:attention hybrid + fine-grained MoE + Engram (ρ = 77%, inside R17's optimum), MTP, no trained loop.
- **Engram stays conditionally.** R40 (an independent 1.3B dense report) found no iso-param gain. R17's win is defined against MoE. Decision rules are in `runbooks/LADDER_50M_GPU.md`.
- **Loops**: not in the main line. The decisive test is moved to the 50M ladder with the R47 recipe (implemented: `moe_loop_router`, `loop_res_scale`).
- **Test-time compute**: sampling plus executable verifiers (G2) preferred over loops at present. TTS curves pending.
- **New candidates from this session**:
  - H14.7: a factorised Engram table (R44/R45) to free table params for experts. Implemented; E4v/E4w/E4x queued.
  - N5: delayed memory schedule. E6L/E4u queued.
  - G1: an editable memory store (E9 queued).

## 8. Efficiency, honestly (TARGET_ANALYSIS §5–6)
- Measured at toy scale: ≈ 2× parameter efficiency and ≈ 2.3× inference- and training-FLOP efficiency vs a 2× dense baseline, all from Engram + MoE, on bpb.
- Analytic at 9B, C4 vs a Qwen3.5-9B-like layout: 3.1× fewer FLOP/token, 2.7× less KV, 3.2× fewer decode bytes. Quality parity is plausible (R42) but unproven.
- The "10,000×" framing holds only as a ratio against frontier-model size. No blended multiplier is claimed.

## 9. Pending (in queue order)
1. s2h: E4l, E5a seed 1, E6L ×4 + E6Lc ×2 (H14.1/N5), E4u.
2. post_s2: gate AUC (N2), TTS curves (E8a–d), downstream MC evals (ds_eval), route_diag (R47 expert collapse), L1 triage.
3. s2l: tests, E4v/E4w/E4x (H14.7), E4s2 (F011 rerun), E8i (Engram interference control), E4y/E4y2 (R47 recipe).
4. s2i: 10M 2×2 {MoE, dense} × {Engram, none} (R40 test at 10M).
5. s2j: tests incl. torchrun CPU smoke, E9 memory editing + collateral damage.
6. GPU required: the 50M ladder per runbook (~27 GPU-hours), then 100M → 1B → 9B.
