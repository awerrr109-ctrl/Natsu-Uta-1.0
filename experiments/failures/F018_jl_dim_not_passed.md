# F018 — post_s2b JL sweep measured only dim 32 (argument not passed)

- **observed**: `knnlm_s2b.jsonl` rows labelled `_jl64`, `_jl32` and `_jl16` all report `"dim": 32` and identical bpb to 16 digits.
- **cause**: `scripts/post_s2b.sh` looped `for r in 64 32 16` and used `$r` only in the output path, not in `--dim $r`. `knnlm compress` defaults to `--dim 32`.
- **impact**: only JL-32 is measured (valid). JL-64/16 are unknown, and an apparent "flat curve over dim" would have been a false finding if not caught. The identical-values check caught it.
- **fix**: pass `--dim $r` (applied to post_s2b.sh for reference) and re-queue JL-64/16 as s2v after s2u. Each needs to rebuild the 3.2M store (~780 MB on disk) because post_s2b deletes it.
- **lesson**: a sweep whose rows are bit-identical is a red flag. Analysis scripts should assert that the swept parameter appears in each output row (`dim` field == label).
