| cand | stored params | non-embed | active params/token (1 pass) | fwd GFLOP/token @4k ctx (all loops) | eff. depth |
|---|---|---|---|---|---|
| C0_dense_hybrid_3to1 | 8.26B | 7.19B | 7.73B | 16.1 | 32 |
| C1_dense_looped | 8.30B | 7.22B | 7.76B | 27.3 | 56 |
| C2_looped_moe | 9.19B | 8.39B | 2.42B | 11.5 | 40 |
| C3_natsu_loopmoe_mem | 8.93B | 8.13B | 2.37B | 11.4 | 40 |
| C4_moe_noloop | 8.91B | 8.11B | 2.36B | 4.9 | 16 |
