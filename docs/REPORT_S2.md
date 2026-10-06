# Session 2 Report (draft, updated as the queue finishes)

Scope: ~0.8M-param toy models on CPU (2 cores, 985 MB RAM). One seed unless stated; seed replicates are queued.
Evidence tags: [E] measured or cited, [I] inference, [H] hypothesis. Read IDs refer to `research/notes/evidence_log.md`.

## 1. Search accounting
- Harvested (L0): ≈40.0k papers, ≈15.4k repos (arXiv, OpenAlex, Crossref, GitHub; low-star buckets included).
- L1 scorer v2 (domain-aware): the on-topic paper estimate fell from 9,505 to 8,303. The v1 scorer gave rel=1.0 to vision/speech "transformer" papers.
- Read at L2 / L3-partial: **35 sources** (R1–R35). This is far below the 10k target for reading depth; harvest breadth meets the target, reading depth does not.

## 2. Main results (val bpb, TinyStories bytes, 1.64M training tokens)
| run | change | bpb | inf. FLOP/tok | note |
|---|---|---|---|---|
| E4c | hybrid GDN+attn + MoE | 1.4870 | 1.49e6 | baseline |
| E4j | + Engram | 1.4278 | 1.58e6 | |
| E4z | dense, 2× params | 1.4394 | 3.61e6 | KD teacher |
| **E5a** | E4c + DPT KD from E4z | **1.4522** | 1.49e6 | closes 73% of the gap to the 2× teacher |
| **E5b** | E4j + DPT KD from E4z | **1.4063** | 1.58e6 | **toy Pareto best**; below teacher, but E4j alone already was |
| E4g | loop R=3 fixed | 1.4574 | 2.57e6 | R6 1.552 |
| E4p | loop + lookahead gate + exit training | 1.4607 | 2.57e6 | anytime: R1 1.513 … R6 1.507 |
| E4m | fixed→uniform curriculum | 1.4701 | | fails pre-registered check |
| E4n | Poisson R | 1.5029 | | depth-flat, but worse |
| E4k | loop + Engram | 1.4327 | 2.51e6 | F007: substitutes |

BPE check (B1–B3): Engram −0.014 and loop −0.006 survive BPE. Both shrink about 4–5× relative to byte level, and the ranking is preserved.

## 3. What changed in the design
1. **Distillation and Engram are partly additive**, unlike loop and Engram. At toy scale, the C4 main line becomes C4 + KD.
   R35 (distillation scaling laws) predicts the KD gain fades at 9B token budgets (~500 tokens/param), so at 9B KD is used only with an existing teacher and mainly in post-training.
   Control E5c (born-again, same-size teacher) is queued.
2. **Loops**:
   - Exit training (E4p) is the only loop recipe that is both anytime and near-best.
   - Gate AUC 0.66 is weak; skipping 33% of tokens costs +0.011 bpb.
   - Loops are still dominated by Engram on gain per FLOP.
   - On the addition probe (E8), loops did **not** help at iso-param: answer accuracy 0.285 vs 0.297. This contradicts R33. Iso-FLOP control E8e and the H13.7 regulariser runs (E8f, E4t) are queued.
   - Status: loops stay an optional inference-time feature of C4, not a core component.
3. **Data (TRAINING_SPEC v0.2)**:
   - Keep ~25–30% of the web at 9B compute (R32: F_opt ∝ C^0.25).
   - Ensemble classifiers and rephrase the lower buckets (R30, R26).
   - Freeze a held-out evaluation suite before selection (R31/R32 Goodhart).

## 3b. Added this turn
- E6a (varchain, no loop): answer loss 0.542, full-sequence exact match 0.016 (long20: 0.047). Sequence EM is too strict to separate architectures. Per-step accuracy by dependency depth is needed (analysis pending, once E6b–d finish).
- R36 (Abacus) suggests E8's "no loop gain" comes from a positional bottleneck. Implemented `digit_pos` (cache-exact test). E8g/E8h are queued behind the test gate.

## 3c. Seed noise (F008) — changes the reading of section 2
Seed spread is 0.003–0.014 bpb. 2-seed means: E4c 1.4803, E4j 1.4262 (Engram −0.054, robust), E4g 1.4633 (loop −0.017, weak).
Loop-recipe rankings below 0.02 are withdrawn. E4e2 (shared-first KV) matches E4g within noise at 3× less core KV, so it is adopted.
E4i shows that concat reinjection explodes off the trained depth. E6 varchain (Engram and loop both worse than plain) is weak evidence (phase-transition timing); retest E6L is queued.

## 3d. Strongest counter-evidence so far (R40) and a self-found bug (F009)
- deepseek-ai/Engram issue #20: on a 1.3B **dense** model with 1T tokens, iso-param Engram gives no loss or eval gain. With extra params, loss gains but eval gains only on HellaSwag.
  Our toy wins are iso-param vs **MoE** (R17's setting) and measured in bpb only. So C4's Engram commitment is now conditional on:
  (a) the 10M 2×2 {MoE,dense}×{Engram,none} (s2i);
  (b) a downstream eval at ≥50M.
- Reading parcae issue #10 (sharding coverage bug) exposed the same bug class in our `train_dist.ShardedTokens` (F009). Fixed with an epoch-exact permutation and exact resume; the unit test gates queue s2j.
- New probe E9: memory editing (ENGRAFT, R39, a 51★ repo). Can facts be written into the Engram table alone, and what does it cost in collateral bpb and held-out phrasing?

## 4. Pending (queue s2c → s2d → s2e → s2f → post_s2)
- E6a–c (varchain); seed replicates (E4c/E4j/E4g ×2); E4i, E4e2, E4l.
- E4q/E4r (N2: Engram gate as depth-router feature); E4s (Engram v2 VIP).
- E4o (LTI, rerun); E6d, E8c, E8d (2×2 memory × loop on reasoning probes); E8e/E8f/E4t (H13.7).
- E5c (KD control).
- post_s2: gate_eval for E4q/E4r; tts_eval (greedy vs maj@N vs oracle@N) for E8a–d.

## 5. Failures this session
F005 updates 2–3 (loop recipe decomposition), F006 recurrence (my smoke test caused an OOM kill of E4o; binding rule added), F007 (N1 refuted on bpb).
