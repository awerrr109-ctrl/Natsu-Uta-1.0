"""
Data pipeline designed for a ~1GB-RAM research box.

* ByteTokenizer: 256 bytes + specials (PAD=256, BOS=257, EOS=258, SEP=259). Zero training cost,
  no OOV; used for toy/10M stages so tokenizer quality is not a confound. Final-scale
  tokenizer spec lives in docs/TRAINING_SPEC.md (BPE 128k, trained with scripts/train_bpe.py).
* encode_file_to_memmap: streams a text file in chunks -> uint16 .bin (never holds corpus in RAM).
* MemmapLoader: random windows from the memmap (OS page cache does the work).
* Synthetic task generators (principle #15: info-dense data + controllable difficulty):
    - mqar : multi-query associative recall (Arora et al. 2023, Zoology)  -> tests recall gap of linear RNNs
    - chain: k-hop variable dereferencing  (a=3;b=a;c=b;...;?c)              -> tests sequential depth / looping
    - add  : multi-digit addition, digits reversed                          -> tests algorithmic generalisation
  Each returns (tokens, loss_mask) so only answer tokens are scored.
"""
import os, random
import numpy as np
import torch

PAD, BOS, EOS, SEP = 256, 257, 258, 259


class ByteTokenizer:
    vocab_size = 260

    def encode(self, s, bos=False, eos=False):
        ids = list(s.encode("utf-8", errors="replace"))
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")


def encode_file_to_memmap(src, dst, tok=None, chunk_bytes=1 << 22, doc_sep="<|endoftext|>"):
    tok = tok or ByteTokenizer()
    n = 0
    with open(src, "r", encoding="utf-8", errors="replace") as f, open(dst, "wb") as out:
        buf = ""
        while True:
            s = f.read(chunk_bytes)
            if not s:
                break
            buf += s
            docs = buf.split(doc_sep)
            buf = docs.pop()
            arr = []
            for d in docs:
                d = d.strip()
                if d:
                    arr += tok.encode(d, bos=True)
            a = np.asarray(arr, dtype=np.uint16)
            a.tofile(out)
            n += a.size
        if buf.strip():
            a = np.asarray(tok.encode(buf.strip(), bos=True), dtype=np.uint16)
            a.tofile(out)
            n += a.size
    return n


class MemmapLoader:
    def __init__(self, path, seq, batch, seed=0):
        self.d = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq, self.batch = seq, batch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.d)

    def get(self):
        ix = self.rng.integers(0, len(self.d) - self.seq - 1, self.batch)
        x = np.stack([self.d[i:i + self.seq + 1].astype(np.int64) for i in ix])
        x = torch.from_numpy(x)
        return x[:, :-1], x[:, 1:], None

    def fixed_eval(self, n_batches, seed=1234):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_batches):
            ix = rng.integers(0, len(self.d) - self.seq - 1, self.batch)
            x = torch.from_numpy(np.stack([self.d[i:i + self.seq + 1].astype(np.int64) for i in ix]))
            out.append((x[:, :-1], x[:, 1:], None))
        return out


# --------------------------------------------------------------------------- synthetic tasks
def _pack(samples, seq):
    """samples: list of (ids, mask) -> padded tensors x,y,mask (mask on y positions)."""
    seq = min(seq, max(len(i) for i, _ in samples) - 1)   # trim padding: compute only what exists
    X = torch.full((len(samples), seq + 1), PAD, dtype=torch.long)
    M = torch.zeros((len(samples), seq + 1), dtype=torch.bool)
    for i, (ids, m) in enumerate(samples):
        ids, m = ids[: seq + 1], m[: seq + 1]
        X[i, : len(ids)] = torch.tensor(ids)
        M[i, : len(m)] = torch.tensor(m)
    return X[:, :-1], X[:, 1:], M[:, 1:]


def gen_chain(rng, hops, n_vars=None, distract=4, n_chains=2):
    """k-hop pointer chasing with `n_chains` disjoint chains of the SAME length and distinct root values.
    Query = tail of one chain. v1 (F001/F002) had a single chain, so "output the digit of the only
    variable that is referenced" solved every hop count (shortcut). With >=2 equal-length chains the
    only hop-independent strategy is guessing among roots (1/n_chains)."""
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    need = n_chains * (hops + 1)
    vs = rng.sample(letters, min(26, need))
    vals = rng.sample([str(d) for d in range(10)], n_chains)
    stmts, tails = [], []
    for c in range(n_chains):
        ch = vs[c * (hops + 1):(c + 1) * (hops + 1)]
        stmts.append(f"{ch[0]}={vals[c]}")
        stmts += [f"{ch[i]}={ch[i-1]}" for i in range(1, hops + 1)]
        tails.append(ch[-1])
    rng.shuffle(stmts)
    q = rng.randrange(n_chains)
    val = vals[q]
    prompt = ";".join(stmts) + f";?{tails[q]}="
    ids = [BOS] + list(prompt.encode()) + list(val.encode()) + [EOS]
    mask = [0] * (1 + len(prompt)) + [1, 1]
    return ids, mask


def gen_mqar(rng, n_pairs=16, n_q=8):
    keys = rng.sample([f"{a}{b}" for a in "abcdefghijklmnop" for b in "abcdefghijklmnop"], n_pairs)
    vals = [str(rng.randint(10, 99)) for _ in keys]
    kv = dict(zip(keys, vals))
    s = " ".join(f"{k}:{v}" for k, v in kv.items()) + " |"
    ids = [BOS] + list(s.encode())
    mask = [0] * len(ids)
    for k in rng.sample(keys, n_q):
        q = f" {k}:"
        ids += list(q.encode()); mask += [0] * len(q)
        ids += list(kv[k].encode()); mask += [1] * len(kv[k])
    return ids, mask


def gen_add(rng, nd):
    a, b = rng.randint(0, 10 ** nd - 1), rng.randint(0, 10 ** nd - 1)
    s = f"{str(a)[::-1]}+{str(b)[::-1]}="
    ans = str(a + b)[::-1]
    ids = [BOS] + list(s.encode()) + list(ans.encode()) + [EOS]
    return ids, [0] * (1 + len(s)) + [1] * (len(ans) + 1)


def gen_varchain_dense(rng, n_vars=8, n_steps=12, mod=10):
    """Densely-supervised LOOKUP+ARITHMETIC probe (F003 follow-up). HONEST SCOPE: intermediate values are written in
    context, so each step = retrieve the latest value of src (induction-like) + one mod-10 op. It tests in-context
    retrieval and arithmetic, NOT deep latent chaining (that needs hidden intermediates; see F003).
    Program: 'a=3;b=a+4;c=b*2;a=c+1;...' where each statement's RHS references an earlier variable; after each
    statement the model must emit the new value: 'b=a+4>7;'. EVERY step is supervised (the value after '>'),
    and step k depends on a chain of k earlier results -> depth of computation grows along the sequence.
    Values mod 10. Per-step accuracy vs 'dependency depth' gives a depth-scaling curve per architecture."""
    names = [chr(c) for c in range(ord("a"), ord("a") + n_vars)]
    vals, depth = {}, {}
    ids, mask, deps = [BOS], [0], []
    v0 = names[0]; x = rng.randrange(mod)
    s = f"{v0}={x}>"; ids += list(s.encode()); mask += [0] * len(s)
    ids += list(str(x).encode()); mask += [1]; ids += list(b";"); mask += [0]
    vals[v0], depth[v0] = x, 0
    for _ in range(n_steps):
        tgt = rng.choice(names); src = rng.choice(list(vals))
        op = rng.choice("+-*"); k = rng.randrange(1, mod)
        v = (vals[src] + k) % mod if op == "+" else (vals[src] - k) % mod if op == "-" else (vals[src] * k) % mod
        st = f"{tgt}={src}{op}{k}>"
        ids += list(st.encode()); mask += [0] * len(st)
        ids += list(str(v).encode()); mask += [1]; ids += list(b";"); mask += [0]
        vals[tgt], depth[tgt] = v, depth[src] + 1
        deps.append(depth[tgt])
    return ids, mask


def gen_facts(rng, n_facts=64, seed_world=1234, split="train"):
    """Invented-world facts (memory-editing probe, ENGRAFT-style). Fixed world (seed_world); each fact = 'The <name> of <place> is <value>.'
    train: canonical phrasing variants 0-2; test: held-out phrasing 3. Answer span = <value> (+ '.'); everything else unscored."""
    w = random.Random(seed_world)
    syl = ["ka", "lo", "mi", "ru", "te", "sa", "no", "vi", "pe", "zu", "ri", "ma"]
    word = lambda: "".join(w.choice(syl) for _ in range(3))
    attrs = ["king", "river", "color", "song", "bird"]
    facts = [(w.choice(attrs), word().capitalize(), word()) for _ in range(n_facts)]
    a, p, v = facts[rng.randrange(n_facts)]
    tmpl = [f"The {a} of {p} is ", f"In {p}, the {a} is ", f"Everyone in {p} knows the {a} is ", f"Ask about {p}: its {a} is "]
    t = tmpl[rng.randrange(3)] if split == "train" else tmpl[3]
    ids = [BOS] + list(t.encode()); ans = list((v + ".").encode())
    return ids + ans + [EOS], [0] * len(ids) + [1] * (len(ans) + 1)


class SyntheticLoader:
    def __init__(self, task, seq, batch, seed=0, **kw):
        self.task, self.seq, self.batch, self.kw = task, seq, batch, kw
        self.rng = random.Random(seed)

    def sample(self, rng, **over):
        kw = {**self.kw, **over}
        if self.task == "chain":
            return gen_chain(rng, rng.randint(kw.get("min_hops", 1), kw.get("hops", 4)), n_chains=kw.get("n_chains", 2))
        if self.task == "mqar":
            return gen_mqar(rng, kw.get("n_pairs", 16), kw.get("n_q", 8))
        if self.task == "varchain":
            return gen_varchain_dense(rng, kw.get("n_vars", 8), kw.get("n_steps", 12))
        if self.task == "facts":
            return gen_facts(rng, kw.get("n_facts", 64), kw.get("seed_world", 1234), kw.get("split", "train"))
        if self.task == "add":
            return gen_add(rng, rng.randint(kw.get("min_digits", 1), kw.get("digits", 5)))
        raise ValueError(self.task)

    def get(self):
        return _pack([self.sample(self.rng) for _ in range(self.batch)], self.seq)

    def fixed_eval(self, n_batches, seed=999, **over):
        rng = random.Random(seed)
        return [_pack([self.sample(rng, **over) for _ in range(self.batch)], self.seq) for _ in range(n_batches)]


class MixLoader:
    """Mixture of loaders with fixed weights (data-mixture experiments)."""
    def __init__(self, loaders, weights, seed=0):
        self.l, self.w = loaders, np.asarray(weights) / np.sum(weights)
        self.rng = np.random.default_rng(seed)

    def get(self):
        return self.l[self.rng.choice(len(self.l), p=self.w)].get()


class RegionLoader:
    """G1c: random seq-aligned windows from a contiguous region [start, start+tokens) of a memmap, returning per-position cached
    kNN targets (top-k values, weights) built by `knnlm targets` with the same start/seq. Without `targets` it is the control loader
    (same region, same window distribution)."""
    def __init__(self, path, start, tokens, seq, batch, seed=0, targets=None):
        self.d = np.memmap(path, dtype=np.uint16, mode="r"); self.start, self.seq, self.batch = start, seq, batch
        self.nw = tokens // seq; self.rng = np.random.default_rng(seed); self.tv = self.tw = None
        if targets:
            import json as _j
            mt = _j.load(open(os.path.join(targets, "meta.json")))
            assert mt["start"] == start and mt["T"] >= self.nw * seq, "targets do not cover the region"
            self.tv = np.memmap(os.path.join(targets, "vals.u16"), dtype=np.uint16, mode="r", shape=(mt["T"], mt["k"]))
            self.tw = np.memmap(os.path.join(targets, "w.f16"), dtype=np.float16, mode="r", shape=(mt["T"], mt["k"]))

    def get(self):
        w = self.rng.integers(0, self.nw, self.batch)
        x = torch.from_numpy(np.stack([self.d[self.start + i * self.seq:self.start + (i + 1) * self.seq + 1].astype(np.int64) for i in w]))
        self.aux = None
        if self.tv is not None:
            self.aux = (torch.from_numpy(np.stack([self.tv[i * self.seq:(i + 1) * self.seq].astype(np.int64) for i in w])),
                        torch.from_numpy(np.stack([self.tw[i * self.seq:(i + 1) * self.seq].astype(np.float32) for i in w])))
        return x[:, :-1], x[:, 1:], None
