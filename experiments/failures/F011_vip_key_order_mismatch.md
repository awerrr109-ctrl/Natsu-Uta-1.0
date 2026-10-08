# F011 — Engram v2 VIP rows: the key builder and the model used opposite n-gram orders

- **hypothesis (R24 fix)**: dedicated collision-free rows for the 256 most frequent 2/3-grams improve on hashed-only Engram.
- **implementation**: `scripts/build_vip.py` built keys oldest-token-first; `Engram.vip_rows` builds them newest-token-first (`win[..., j]` = token at −j).
  The cache-exactness test (test_engram_vip_cache) passed because it built keys *with the model's own function*, so it never exercised the builder.
- **expected**: E4s ≤ E4j (1.4262 3-seed mean).
- **observed**: E4s 1.4337 (+0.0075, ≈ 5 sd of E4j's seed spread). Analysis: with the wrong order, the VIP hit rate on validation was **65% for 2-grams
  (instead of 93%) and 10.5% for 3-grams (instead of 55%)**. Hits were mostly on *reversed* n-grams, i.e. wrong rows. The hashed table was halved
  (slots 512 → 256) to keep iso-param, so the run was effectively "Engram with half the table plus mostly-misrouted VIP rows". The worse result says
  nothing about the VIP idea itself.
- **why**: two implementations of the same key function, with a test that only covered one of them.
- **fix**: builder now uses the model's order. Regression test `test_vip_key_order_matches_builder` compares builder keys against `Engram.ngram_key`.
  Corrected keys → `data_cache/vip_bytes_256_v2.json`; rerun **E4s2** queued.
- **lessons**: when an offline artefact (keys, vocab, tables) feeds a model, test the *offline builder* against the model's function, not the model against itself.
