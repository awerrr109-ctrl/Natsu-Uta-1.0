| metric | C0 Qwen3.5-9B-like dense 3:1 | C4 Natsu-9B (MoE+Engram) | C3' looped variant (shared KV) |
|---|---|---|---|
| stored_B | 9.223 | 9.116 | 8.334 |
| engram_table_B | 0.000 | 1.611 | 1.611 |
| experts_B | 0.000 | 5.379 | 4.926 |
| resident_B_if_table_offloaded | 9.223 | 7.505 | 6.724 |
| gflop_tok_prefill4k | 17.046 | 5.526 | 6.261 |
| gflop_tok_decode32k | 20.804 | 7.639 | 8.727 |
| weights_bf16_GB | 18.445 | 18.232 | 16.669 |
| weights_4bit_GB | 4.611 | 4.558 | 4.167 |
| resident_4bit_GB_table_offloaded | 4.611 | 3.753 | 3.362 |
| kv_MB_at_32k | 1073.742 | 402.653 | 268.435 |
| kv_GB_at_256k | 8.590 | 3.221 | 2.147 |
| recurrent_state_MB | 50.332 | 28.312 | 33.030 |
| decode_bytes_per_token_GB | 4.103 | 1.286 | 0.952 |
