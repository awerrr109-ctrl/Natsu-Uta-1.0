# Runbook: 50M ladder stage on GPU (decisive tests that cannot run on the 1-GB CPU sandbox)

All code paths below exist and are CPU-tested (`train_dist.py`: DDP/FSDP2, epoch-exact sharded sampler, exact resume, bf16).
The 50M stage answers the three questions toy scale could not:
1. Does Engram beat iso-param MoE on *downstream* tasks (not just bpb)? (R40 vs R17)
2. Do loops with the R47 recipe pay at iso-FLOP? (F005 vs R47/R48)
3. Does delayed memory (N5) matter once induction heads form normally?

## Data
- Tokenizer: BPE-32k trained on the corpus mix (`natsu.tokenizer.train_bpe`). Corpus: FineWeb-Edu sample (10B tokens) or similar, encoded with `encode_file_bpe` to uint16 memmap.
- Held-out evals, frozen before any data selection (R31/R32): HellaSwag, PIQA, ARC-e/c, LAMBADA, plus `experiments/evalsets/` (TinyStories cloze / names) as a smoke test.

## Cells (iso-stored-params ≈ 54M, iso-tokens 2.5B ≈ 46 tok/param; 3 seeds for the A/B/C cells)
| cell | config | change vs C4_50M |
|---|---|---|
| A | `configs/ladder/C4_50M.json` | — (MoE + Engram, ρ≈77%) |
| B | C4_50M, `engram_slots=0`, experts resized iso-param | no Engram |
| C | C4_50M dense FFN, `moe_experts=0`, ffn resized iso-param | dense, no Engram (R40 control) |
| D | C4_50M + `n_loops=3, moe_loop_router=true, loop_res_scale=1.0, reinject=true, reinject_mode="lti", loop_kv="shared_first"`, `n_core` reduced so FLOPs ≈ A | R47-style loop at iso-FLOP |
| E | A with `--engram_lr_mult 1` | **table-lr control** (toy: ×5 = −0.027 bpb, 3 seeds; ×5 also speeds ICL emergence). Replaces the N5 cell (rejected at toy scale) |
| F | A with `--engram_lr_mult 20` | table-lr sweep point (R67 uses 50×; toy sweep ×10/20/50 pending in s2q). Replaces the H14.7 cell (rejected at toy scale) |

Use `scripts/iso_param_match.py` to size B/C/F, and the analytic `flops_per_token` for D.

## Command (8×GPU node; per cell)
```bash
torchrun --nproc_per_node 8 -m natsu.train_dist --config configs/ladder/C4_50M.json --override "$CELL_OVERRIDES" $CELL_FLAGS \
  --data data/fwedu_bpe32k.bin --seq 2048 --batch 16 --accum 2 --steps 9500 --lr 1.5e-3 --optim muon --bf16 \
  --out runs/50M_<cell>_s<seed> --save_every 1000
```
(LR for MoE cells: scale by E^-0.25 relative to the dense-tuned LR, R42.)

## Pre-registered decision rules
- Engram stays in C4 iff A beats B on the mean held-out downstream accuracy by ≥ 0.5 pt (3 seeds, paired by seed) **and** on val bpb.
- Loops are re-adopted iff D beats A at iso-FLOP on the reasoning subset (ARC-c, LAMBADA) by ≥ 0.5 pt without losing more than 0.3 pt elsewhere.
- The table lr ×5 default is kept iff A beats E on val bpb (3 seeds) **and** on the in-context subset. Otherwise ×1 is the default at scale (toy result does not transfer).
- F replaces ×5 iff F beats A on val bpb by ≥ 2 sd without an ICL loss.
- Domain gating: also evaluate A on an arithmetic probe (toy: Engram hurts addition, E8i/E8j). If the gap is > 2 pt, enable `engram_skip_digits` in C4 (pending the E8k/E8l result).

## Cost estimate
54M stored / ~19M active × 2.5B tokens ≈ 6 × 19e6 × 2.5e9 ≈ 2.9e17 FLOPs per run → ~1.5 GPU-hours (H100 at ~50% MFU). 6 cells × 3 seeds ≈ 27 GPU-hours.

Cell E passes `CELL_FLAGS="--engram_lr_mult 1"`, cell F `CELL_FLAGS="--engram_lr_mult 20"`. The default in train_dist is ×5. All cells set `"mtp":1` in the overrides (MTP auxiliary loss, `--mtp_w 0.3`).
CPU smoke test of exactly this path: `tests/test_dist_cpu.sh` (gloo ×2, MTP + engram_delay, save → resume).
