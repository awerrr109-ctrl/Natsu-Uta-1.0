# Natsu-Uta — parameter-efficient ~9B architecture research

Goal: a ~9B-stored-parameter system that maximises capability per unit of compute, memory, and data. It is
evaluated on a quality/cost Pareto frontier against public models. This repository holds the
research process (search → hypotheses → refutation → experiments → state) and a working implementation.

**Status (session 1):** research infrastructure, full model and training/inference code, toy-scale experiments in
progress on a 2-vCPU / 1 GB-RAM sandbox. **No 9B model has been trained.** The 9B configs are exact (parameter
counts come from the real code), but training them requires GPUs.

## Layout
| path | contents |
|---|---|
| `research/taxonomy.py` | problem → hypothesis → support/refute queries (all searches are traceable) |
| `research/harvest.py` | arXiv / OpenAlex / Crossref / GitHub (star-stratified) harvester → `research/db/research.sqlite` |
| `research/analyze.py` | L1 triage, corpus stats, reading queue (`research/notes/`) |
| `research/notes/evidence_log.md` | what was actually read (L2/L3), with evidence/inference/hypothesis tags |
| `src/natsu/deltanet.py` | Gated DeltaNet: chunkwise-parallel + recurrent (verified equal to 1e-8) |
| `src/natsu/model.py` | Natsu family: hybrid GDN/attention, looped core, looped MoE (per-loop router bias), shared-KV loops, Engram n-gram memory, PKM, depth gate, MTP |
| `src/natsu/train.py` | trainer: loop sampling, shallow→deep self-distillation, MTP, Muon/AdamW, WSD, RSS/FLOP logging |
| `src/natsu/generate.py` | cached decoding, loop self-speculative decoding (exact), KL adaptive depth |
| `src/natsu/bench.py`, `scripts/pareto.py` | Pareto benchmark (quality × params × train FLOPs × inference FLOPs × RAM × latency) |
| `scripts/size_candidates.py` | exact 9B candidate sizing on meta device → `configs/`, `docs/generated/candidates.md` |
| `docs/` | RESEARCH_PLAN, TARGET_ANALYSIS, ARCHITECTURE, TRAINING_SPEC, INFERENCE_SPEC, REPORT_S1 |
| `experiments/` | configs, results (JSONL logs + final.json), **failures/** (every failed experiment) |
| `state/PROJECT_STATE.json` | persistent research state for the next session |

## Quick start
```bash
pip install torch numpy
python tests/test_deltanet.py && (cd src && python ../tests/test_model.py && python ../tests/test_generate.py)
# data (TinyStories bytes): see docs/REPORT_S1.md §Reproduce
scripts/run_queue.sh experiments/configs/E3b_ts_hyb4.json
python -m natsu.bench --ckpt checkpoints/E3b_ts_hyb4.pt   # (from src/)
python scripts/size_candidates.py
```
