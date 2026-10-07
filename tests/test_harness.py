import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig
from natsu.data import ByteTokenizer
from natsu.eval_harness import eval_mc, eval_gen, run_code, pass_at_k, extract_number

def test_harness_smoke():
    m = Natsu(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32)).eval(); t = ByteTokenizer()
    acc = eval_mc(m, t, [{"ctx": "The sky is", "choices": [" blue", " green"], "label": 0}] * 2)
    assert 0.0 <= acc <= 1.0
    assert eval_gen(m, t, [{"prompt": "1+1=", "answer": "2"}], max_new=4) in (0.0, 1.0)
    assert run_code("def f(x):\n    return x+1", "assert f(1)==2") and not run_code("def f(x):\n    return x", "assert f(1)==2")
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9 and extract_number("so the answer is 1,234.") == "1234"

if __name__ == "__main__":
    test_harness_smoke(); print("ok")

def test_sharded_tokens_epoch_exact():
    import numpy as np, tempfile, os
    from natsu.train_dist import ShardedTokens
    d = tempfile.mkdtemp(); p = os.path.join(d, "t.bin")
    np.arange(10 * 9 * 4 + 1, dtype=np.uint16).tofile(p)          # 40 windows of seq+1=9 (+1 token)
    W, B = 4, 2
    shards = [ShardedTokens(p, 8, B, r, W, seed=3) for r in range(W)]
    seen = []
    for _ in range(len(shards[0].mine) // B):
        for s in shards:
            x, _ = s.get(); seen += [int(v) // 9 for v in x[:, 0]]
    assert len(seen) == len(set(seen)) == 40, (len(seen), len(set(seen)))   # every window exactly once per epoch, no overlap
    s = shards[1]; st = s.state(); a = s.get()[0]
    t = ShardedTokens(p, 8, B, 1, W, seed=3); t.load_state(st); b = t.get()[0]
    assert torch.equal(a, b)                                           # exact resume

if __name__ == "__main__":
    test_sharded_tokens_epoch_exact(); print("ok sharded epoch-exact")

def test_knn_compress_preserves_neighbours():
    """G1b: JL projection + int8 keys. At dim=D the projection is a scaled rotation, so top-1 neighbours must match fp16 almost always."""
    import json, os, tempfile, types, numpy as np, torch
    from natsu import knnlm
    tmp = tempfile.mkdtemp(); N, D = 2000, 32
    rng = np.random.default_rng(0); keys = rng.standard_normal((N, D)).astype(np.float16)
    src = os.path.join(tmp, "s"); os.makedirs(src)
    np.memmap(os.path.join(src, "keys.f16"), dtype=np.float16, mode="w+", shape=(N, D))[:] = keys
    np.memmap(os.path.join(src, "vals.u16"), dtype=np.uint16, mode="w+", shape=(N,))[:] = np.arange(N) % 256
    json.dump({"N": N, "D": D}, open(os.path.join(src, "meta.json"), "w"))
    old = knnlm.ROOT; knnlm.ROOT = "/"
    try:
        for dim, need in ((D, 0.97), (D // 2, 0.5)):
            out = os.path.join(tmp, f"c{dim}")
            knnlm.compress(types.SimpleNamespace(store=src, out=out, dim=dim, seed=0))
            K8 = knnlm.Int8Keys(out, N, dim); P = torch.from_numpy(np.load(os.path.join(out, "proj.npy")))
            V = np.memmap(os.path.join(src, "vals.u16"), dtype=np.uint16, mode="r", shape=(N,))
            q = torch.from_numpy(keys[:200].astype(np.float32)) + 0.3 * torch.randn(200, D)
            _, v_ref = knnlm.knn(q, keys, V, 1); _, v_c = knnlm.knn(q @ P, K8, V, 1)
            agree = (v_ref == v_c).float().mean().item()
            assert agree >= need, (dim, agree)
    finally:
        knnlm.ROOT = old
