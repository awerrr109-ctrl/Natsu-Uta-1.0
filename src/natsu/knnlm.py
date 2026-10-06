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
        dist, vals = knn(h[0].float(), K, V, a.k)
        w = F.softmax(-dist / a.temp, -1)
        p_knn = torch.zeros_like(p_lm).scatter_add_(1, vals, w)
        for l in lams:
            p = (1 - l) * p_lm + l * p_knn
            nll[l] += -torch.log(p.gather(1, y[:, None]).clamp_min(1e-12)).sum().item()
        n += a.seq
    bpt = a.bytes_per_token
    res = {"ckpt": a.ckpt, "store_N": meta["N"], "k": a.k, "temp": a.temp,
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
    a = ap.parse_args()
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()
