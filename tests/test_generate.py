import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig
from natsu.generate import generate, loop_speculative

def test_spec_equals_target_greedy():
    torch.manual_seed(0)
    c = NatsuConfig(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32, pattern="ga", n_loops=3, chunk=8, reinject=True)
    m = Natsu(c).double()
    x = torch.randint(0, 256, (1, 11))
    ref = generate(m, x, 20, n_loops=3)
    for k in (1, 3, 5):
        out, st = loop_speculative(m, x, 20, r_draft=1, r_target=3, k=k)
        assert torch.equal(out, ref), (k, out, ref)
    # draft == target -> everything accepted
    out, st = loop_speculative(m, x, 20, r_draft=3, r_target=3, k=4)
    assert torch.equal(out, ref) and st["accept_rate"] == 1.0, st

if __name__ == "__main__":
    test_spec_equals_target_greedy(); print("ok")
