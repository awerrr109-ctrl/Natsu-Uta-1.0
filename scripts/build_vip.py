"""Count n-gram frequencies over a token memmap (streaming, bounded RAM via numpy unique on chunks + Counter merge of top-K)
and save the top `k` keys per order -> data_cache/vip_<name>.json (keys compatible with Engram.ngram_key)."""
import json, sys, numpy as np
from collections import Counter
src, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
orders = [2, 3]
d = np.memmap(src, dtype=np.uint16, mode="r")
res = {}
for n in orders:
    cnt = Counter(); CH = 2_000_000
    for s in range(0, min(len(d), 40_000_000) - n, CH):
        a = d[s:s + CH + n - 1].astype(np.int64) + 1
        key = np.zeros(len(a) - n + 1, dtype=np.int64)
        # MUST match Engram.ngram_key on the model's window, which is ordered NEWEST token first:
        # win[..., j] = token at offset -j from the current position (F011: oldest-first here made 35-90% of VIP keys wrong).
        for j in range(n):
            key = key * (1 << 20) + a[n - 1 - j: len(a) - j]
        u, c = np.unique(key, return_counts=True)
        top = np.argsort(-c)[: 4 * k]
        cnt.update(dict(zip(u[top].tolist(), c[top].tolist())))
    res[n] = [kk for kk, _ in cnt.most_common(k)]
    print(n, "top count", cnt.most_common(1)[0][1], "kth", cnt.most_common(k)[-1][1])
json.dump(res, open(out, "w"))
