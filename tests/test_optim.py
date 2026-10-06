import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.optim import Adam8bit, Muon

def run(opt_fn, steps=300, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(64, 64) / 8; X = torch.randn(512, 64); Y = X @ W
    lin = torch.nn.Linear(64, 64, bias=False); opt = opt_fn(lin.parameters())
    for _ in range(steps):
        l = ((lin(X) - Y) ** 2).mean(); l.backward(); opt.step(); opt.zero_grad()
    return l.item(), opt

def test_adam8bit_close_to_adamw():
    l32, _ = run(lambda p: torch.optim.AdamW(p, lr=3e-3, betas=(0.9, 0.95), weight_decay=0))
    l8, o = run(lambda p: Adam8bit(p, lr=3e-3, min_8bit=1))
    print("adamw", l32, "adam8bit", l8, "state bytes", o.state_bytes(), "fp32 would be", 64 * 64 * 8)
    assert l8 < 5 * l32 + 1e-4 and o.state_bytes() < 64 * 64 * 8 / 3

if __name__ == "__main__":
    test_adam8bit_close_to_adamw(); print("ok")
