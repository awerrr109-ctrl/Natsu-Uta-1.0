import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig

def _check_cache(cfg):
    torch.manual_seed(0)
    m = Natsu(cfg).double().eval()
    x = torch.randint(0, cfg.vocab_size, (2, 21))
    full = m(x)["logits"]
    cache = {}
    out = m(x[:, :13], cache=cache); cache = out["cache"]; parts = [out["logits"]]
    for t in range(13, 21):
        out = m(x[:, t:t+1], cache=cache, pos0=t); cache = out["cache"]; parts.append(out["logits"])
    inc = torch.cat(parts, 1)
    assert torch.allclose(full, inc, atol=1e-6), (full - inc).abs().max()

def test_cache_hybrid():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="gas", window=8, chunk=8))

def test_cache_looped():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=3,
                             loop_lora_rank=4, reinject=True, depth_gate=True, pkm_keys=16, chunk=8, mtp=1))

def test_variable_loops_and_flops():
    c = NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, n_loops=4, depth_gate=True)
    m = Natsu(c)
    x = torch.randint(0, 260, (1, 9))
    for r in range(1, 5):
        assert m(x, n_loops=r)["logits"].shape == (1, 9, 260)
    assert m.flops_per_token(1) < m.flops_per_token(4)
    o = m(x, gate_threshold=0.99)
    assert "skip_frac" in o

if __name__ == "__main__":
    test_cache_hybrid(); test_cache_looped(); test_variable_loops_and_flops(); print("ok")

def test_moe_loop_cache():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=3,
                             moe_experts=8, moe_topk=2, chunk=8))
    c = NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, n_loops=3, moe_experts=8, moe_topk=2)
    m = Natsu(c)
    dense = Natsu(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, n_loops=3))
    assert m.flops_per_token() < m.flops_per_token() * 2
    print("moe params", m.param_count(), "dense", dense.param_count(), "flops", m.flops_per_token(), dense.flops_per_token())

if __name__ == "__main__":
    test_moe_loop_cache(); print("ok moe")

def test_shared_kv_cache():
    cfg = NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=3, chunk=8,
                      loop_kv="shared_first", reinject=True)
    _check_cache(cfg)
    m = Natsu(cfg).eval()
    x = torch.randint(0, 260, (1, 9))
    cache = m(x, cache={})["cache"]
    assert "c0_r1" not in cache and "c0_r2" not in cache and "c1_r2" in cache, list(cache)  # core attn (c0) caches loop 0 only; GDN (c1) keeps per-loop O(1) state

if __name__ == "__main__":
    test_shared_kv_cache(); print("ok shared kv")

def test_engram_cache():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=2, chunk=8,
                             engram_slots=257, engram_orders=(2, 3), engram_heads=2))
    from natsu.model import Engram
    e = Engram(NatsuConfig(d_model=64, engram_slots=101))
    ids = torch.randint(0, 256, (2, 10))
    a, _ = e.addresses(ids)
    b1, t = e.addresses(ids[:, :6]); b2, _ = e.addresses(ids[:, 6:], t)
    assert torch.equal(a, torch.cat([b1, b2], 1))   # addresses depend only on ids -> prefetchable

if __name__ == "__main__":
    test_engram_cache(); print("ok engram")

def test_lti_reinject_cache():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=3, chunk=8,
                             reinject=True, reinject_mode="lti"))

if __name__ == "__main__":
    test_lti_reinject_cache(); print("ok lti")

def test_engram_vip_cache():
    cfg = NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", chunk=8, engram_slots=257, engram_heads=2, engram_vip=16)
    from natsu.model import Engram
    torch.manual_seed(0)
    m = Natsu(cfg).double().eval()
    ids = torch.randint(0, 260, (2, 21))
    keys = {}
    for n in (2, 3):
        win = torch.stack([ids[0, j: j + 19] for j in range(n)], -1)[None] if n == 3 else torch.stack([ids[0, j: j + 20] for j in range(n)], -1)[None]
        keys[n] = Engram.ngram_key(win).flatten().tolist()[:16]
    m.engram.load_vip(keys)
    _check_cache_model(m, ids)

def _check_cache_model(m, x):
    full = m(x)["logits"]; cache = {}
    out = m(x[:, :13], cache=cache); cache = out["cache"]; parts = [out["logits"]]
    for t in range(13, x.shape[1]):
        out = m(x[:, t:t+1], cache=cache, pos0=t); cache = out["cache"]; parts.append(out["logits"])
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-6)

if __name__ == "__main__":
    test_engram_vip_cache(); print("ok vip")

def test_digit_pos_cache():
    torch.manual_seed(0)
    m = Natsu(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", chunk=8, digit_pos=12)).double().eval()
    x = torch.tensor([list(b"x12345+678=9") + list(b"0123456789")[:9]])   # digit runs crossing the prefill/decode split
    di, _ = Natsu.digit_index(x)
    assert di[0, :7].tolist() == [0, 1, 2, 3, 4, 5, 0]
    _check_cache_model(m, x)

if __name__ == "__main__":
    test_digit_pos_cache(); print("ok digit_pos")

def test_vip_key_order_matches_builder():
    """F011 regression: keys produced by scripts/build_vip.py must equal Engram.ngram_key on the model's (newest-first) window."""
    import numpy as np
    from natsu.model import Engram
    toks = np.array([5, 9, 2, 7, 7, 3], dtype=np.int64)
    a = toks + 1
    for n in (2, 3):
        key = np.zeros(len(a) - n + 1, dtype=np.int64)
        for j in range(n):
            key = key * (1 << 20) + a[n - 1 - j: len(a) - j]
        full = torch.tensor(toks)[None]
        L = len(toks) - n + 1
        win = torch.stack([full[:, n - 1 - j: n - 1 - j + L] for j in range(n)], -1)   # same slicing as Engram.vip_rows with maxo=n
        assert Engram.ngram_key(win)[0].tolist() == key.tolist(), n

if __name__ == "__main__":
    test_vip_key_order_matches_builder(); print("ok vip key order")

def test_engram_factorised_cache():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", chunk=8,
                             engram_slots=257, engram_heads=2, engram_factor_rank=8))

if __name__ == "__main__":
    test_engram_factorised_cache(); print("ok engram factorised")

def test_loop_router_resscale_cache():
    _check_cache(NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", chunk=8, n_loops=3,
                             moe_experts=4, moe_topk=2, moe_shared=1, moe_loop_router=True, loop_res_scale=1.0))

if __name__ == "__main__":
    test_loop_router_resscale_cache(); print("ok loop router + res scale")
