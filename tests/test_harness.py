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
