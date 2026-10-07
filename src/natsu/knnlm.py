"""Minimal kNN-LM retrieval channel (NEXT_GEN G1b; R55: ~1 datastore token per parameter captures most of the gain).

Datastore = (key = final hidden state before the LM head at position t, value = token at t+1), built from any token memmap.
Keys are stored as a float16 memmap on disk (constant RAM), and search is exact (chunked dot products), which is fine for
datastores of ~1M entries on CPU. At ~1B entries one would swap in an ANN index; the interface stays the same.

p(y|x) = (1-λ) p_LM(y|x) + λ p_kNN(y|x),  p_kNN ∝ Σ_{k in topK} exp(-d_k / T) [v_k = y]   (Khandelwal et al. 2020)

usage:
  python -m natsu.knnlm build --ckpt checkpoints/X.pt --data data_cache/ts_train.bin --tokens 800000 --out data_cache/knn_X
  python -m natsu.knnlm eval  --ckpt checkpoints/X.pt --store data_cache/knn_X --valid data_cache/ts_valid.bin --lams 0,0.1,0.25,0.5
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from natsu.model import Natsu, NatsuConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def load(ck):
    c = torch.load(os.path.join(ROOT, ck), map_location="cpu")
    m = Natsu(NatsuConfig(**c["config"])); m.load_state_dict(c["state_dict"]); m.eval()
    return m


@torch.no_grad()
def hidden(m, x):
    out = m(x, return_hidden=True)
    return m.norm(out["hidden"]), out["logits"]          # keys = normalised final hidden (what the LM head sees)


@torch.no_grad()
def build(a):
    torch.set_num_threads(2)
    m = load(a.ckpt); d = np.memmap(os.path.join(ROOT, a.data), dtype=np.uint16, mode="r")
    D = m.c.d_model; N = min(a.tokens, len(d) - 1); seq = a.seq
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    K = np.memmap(os.path.join(ROOT, a.out, "keys.f16"), dtype=np.float16, mode="w+", shape=(N, D))
    V = np.memmap(os.path.join(ROOT, a.out, "vals.u16"), dtype=np.uint16, mode="w+", shape=(N,))
    n = 0
    while n < N:
        L = min(seq, N - n)
        x = torch.from_numpy(d[n:n + L].astype(np.int64))[None]
        h, _ = hidden(m, x)
        K[n:n + L] = h[0].numpy().astype(np.float16)
        V[n:n + L] = d[n + 1:n + L + 1]
        n += L
    K.flush(); V.flush()
    json.dump({"N": int(N), "D": int(D), "ckpt": a.ckpt, "data": a.data}, open(os.path.join(ROOT, a.out, "meta.json"), "w"))
    print(json.dumps({"built": a.out, "N": int(N), "MB": round(N * D * 2 / 2**20, 1)}))


def compress(a):
    """Datastore cost reduction (G1b bottleneck: fp16 keys = 2*D bytes/token). Johnson-Lindenstrauss random orthogonal projection
    D -> r (scaled by sqrt(D/r) so distances, hence the softmax temperature, keep their scale) + symmetric int8 per-row quantisation.
    Bytes/token: 2D+2 (fp16) -> r+2+2 (int8 key + fp16 scale + u16 value). Training-free; built from an existing store."""
    meta = json.load(open(os.path.join(ROOT, a.store, "meta.json"))); N, D = meta["N"], meta["D"]
    K = np.memmap(os.path.join(ROOT, a.store, "keys.f16"), dtype=np.float16, mode="r", shape=(N, D))
    g = np.random.default_rng(a.seed).standard_normal((D, D)); Q, _ = np.linalg.qr(g)
    P = (Q[:, :a.dim] * math.sqrt(D / a.dim)).astype(np.float32)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    K8 = np.memmap(os.path.join(ROOT, a.out, "keys.i8"), dtype=np.int8, mode="w+", shape=(N, a.dim))
    S = np.memmap(os.path.join(ROOT, a.out, "scales.f16"), dtype=np.float16, mode="w+", shape=(N,))
    for s in range(0, N, 100_000):
        z = np.asarray(K[s:s + 100_000], dtype=np.float32) @ P
        sc = np.abs(z).max(1) / 127.0 + 1e-8
        K8[s:s + len(z)] = np.clip(np.round(z / sc[:, None]), -127, 127).astype(np.int8); S[s:s + len(z)] = sc.astype(np.float16)
    K8.flush(); S.flush(); np.save(os.path.join(ROOT, a.out, "proj.npy"), P)
    os.link(os.path.join(ROOT, a.store, "vals.u16"), os.path.join(ROOT, a.out, "vals.u16")) if not os.path.exists(os.path.join(ROOT, a.out, "vals.u16")) else None
    json.dump({**meta, "D": a.dim, "src_D": D, "quant": "int8", "proj": True}, open(os.path.join(ROOT, a.out, "meta.json"), "w"))
    print(json.dumps({"compressed": a.out, "N": N, "dim": a.dim, "bytes_per_token": a.dim + 4, "vs_fp16": 2 * D + 2}))


class Int8Keys:
    """array-like view: K[s:e] -> dequantised float32 rows (so knn() is unchanged)."""
    def __init__(self, path, N, D):
        self.k = np.memmap(os.path.join(path, "keys.i8"), dtype=np.int8, mode="r", shape=(N, D))
        self.s = np.memmap(os.path.join(path, "scales.f16"), dtype=np.float16, mode="r", shape=(N,)); self.shape = (N, D)
    def __getitem__(self, sl):
        return np.asarray(self.k[sl], dtype=np.float32) * np.asarray(self.s[sl], dtype=np.float32)[:, None]


def knn(q, K, V, k, chunk=50_000):
    """exact top-k by L2 distance over a float16 memmap, chunked. q: (T,D) float32 -> (dist (T,k), vals (T,k))."""
    best_d = torch.full((q.shape[0], k), float("inf")); best_v = torch.zeros((q.shape[0], k), dtype=torch.long)
    qq = (q * q).sum(-1, keepdim=True)
    for s in range(0, K.shape[0], chunk):
        kb = torch.from_numpy(np.asarray(K[s:s + chunk], dtype=np.float32))
        dist = qq - 2 * q @ kb.T + (kb * kb).sum(-1)[None]
        vb = torch.from_numpy(np.asarray(V[s:s + chunk], dtype=np.int64))
        dd, ii = torch.topk(dist, min(k, dist.shape[1]), largest=False)
        cat_d = torch.cat([best_d, dd], 1); cat_v = torch.cat([best_v, vb[ii]], 1)
        o = torch.topk(cat_d, k, largest=False)
        best_d, best_v = o.values, cat_v.gather(1, o.indices)
    return best_d, best_v


@torch.no_grad()
def evaluate(a):
    torch.set_num_threads(2)
    m = load(a.ckpt); meta = json.load(open(os.path.join(ROOT, a.store, "meta.json")))
    P = None
    if meta.get("proj"):
        K = Int8Keys(os.path.join(ROOT, a.store), meta["N"], meta["D"]); P = torch.from_numpy(np.load(os.path.join(ROOT, a.store, "proj.npy")))
    else:
        K = np.memmap(os.path.join(ROOT, a.store, "keys.f16"), dtype=np.float16, mode="r", shape=(meta["N"], meta["D"]))
    V = np.memmap(os.path.join(ROOT, a.store, "vals.u16"), dtype=np.uint16, mode="r", shape=(meta["N"],))
    d = np.memmap(os.path.join(ROOT, a.valid), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(4321)
    lams = [float(x) for x in a.lams.split(",")]
    nll = {l: 0.0 for l in lams}; n = 0
    for _ in range(a.batches):
        i = int(rng.integers(0, len(d) - a.seq - 1))
        x = torch.from_numpy(d[i:i + a.seq].astype(np.int64))[None]; y = torch.from_numpy(d[i + 1:i + a.seq + 1].astype(np.int64))
        h, lg = hidden(m, x)
        p_lm = F.softmax(lg[0].float(), -1)
        q = h[0].float() if P is None else h[0].float() @ P
        dist, vals = knn(q, K, V, a.k)
        w = F.softmax(-dist / a.temp, -1)
        p_knn = torch.zeros_like(p_lm).scatter_add_(1, vals, w)
        for l in lams:
            p = (1 - l) * p_lm + l * p_knn
            nll[l] += -torch.log(p.gather(1, y[:, None]).clamp_min(1e-12)).sum().item()
        n += a.seq
    bpt = a.bytes_per_token
    res = {"ckpt": a.ckpt, "store": a.store, "store_N": meta["N"], "key_dim": meta["D"], "quant": meta.get("quant", "fp16"), "k": a.k, "temp": a.temp,
           "bpb": {str(l): nll[l] / n / math.log(2) / bpt for l in lams}}
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--ckpt", required=True); b.add_argument("--data", required=True)
    b.add_argument("--tokens", type=int, default=800_000); b.add_argument("--seq", type=int, default=256); b.add_argument("--out", required=True)
    e = sp.add_parser("eval"); e.add_argument("--ckpt", required=True); e.add_argument("--store", required=True)
    e.add_argument("--valid", default="data_cache/ts_valid.bin"); e.add_argument("--seq", type=int, default=256)
    e.add_argument("--batches", type=int, default=24); e.add_argument("--k", type=int, default=16); e.add_argument("--temp", type=float, default=1.0)
    e.add_argument("--lams", default="0,0.1,0.25,0.5"); e.add_argument("--bytes_per_token", type=float, default=1.0)
    c = sp.add_parser("compress"); c.add_argument("--store", required=True); c.add_argument("--out", required=True)
    c.add_argument("--dim", type=int, default=32); c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"build": build, "eval": evaluate, "compress": compress}[a.cmd](a)


if __name__ == "__main__":
    main()
