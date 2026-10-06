"""
Optimizers & schedules.

* Muon (Jordan et al. 2024; Liu et al. 2025 "Muon is Scalable for LLM Training", arXiv:2502.16982):
  momentum + Newton-Schulz orthogonalisation for 2-D hidden matrices; AdamW for
  embeddings / norms / gains / 1-D params. Moonshot's update-RMS matching (0.2*sqrt(max(A,B)))
  is used so the AdamW lr/wd transfer.
  Memory note: Muon keeps ONE state buffer (momentum) vs Adam's two -> 33% less optimizer
  RAM for matrix params; relevant to the 1GB constraint.
* WSD schedule (warmup-stable-decay; Hu et al. MiniCPM 2024; Hägele et al. 2024): allows
  checkpoint-reuse "continue from stable phase" experiments -> research efficiency.
"""
import math
import torch


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if tr else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                m.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(m, alpha=g["momentum"]) if g["nesterov"] else m
                shp = u.shape
                u2 = u.reshape(-1, shp[-1]) if u.ndim > 2 else u
                o = newton_schulz(u2).reshape(shp)
                scale = 0.2 * math.sqrt(max(u2.shape))
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(o, alpha=-g["lr"] * scale)


def build_optim(model, kind="adamw", lr=3e-3, wd=0.1, betas=(0.9, 0.95)):
    matrix, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in n and "values" not in n and "engram.table" not in n and "engram.basis" not in n and "vip_table" not in n and "loop_emb" not in n and "loop_bias" not in n and n != "inj.weight" and "lti_" not in n:
            matrix.append(p)
        else:
            other.append(p)
    adam_other = torch.optim.AdamW([{"params": other, "weight_decay": 0.0}], lr=lr, betas=betas)
    if kind == "adamw":
        return [torch.optim.AdamW([{"params": matrix, "weight_decay": wd}], lr=lr, betas=betas), adam_other]
    if kind == "muon":
        return [Muon(matrix, lr=lr, wd=wd), adam_other]
    if kind == "adam8bit":
        return [Adam8bit([{"params": matrix, "weight_decay": wd}], lr=lr, betas=betas), adam_other]
    raise ValueError(kind)


def wsd(step, total, warmup, decay_frac=0.2, min_ratio=0.0):
    if step < warmup:
        return (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return 1.0
    t = (step - d0) / max(1, total - d0)
    return min_ratio + (1 - min_ratio) * (1 - math.sqrt(t))   # 1-sqrt cooldown (Hägele et al.)


def cosine(step, total, warmup, min_ratio=0.1):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


class Adam8bit(torch.optim.Optimizer):
    """AdamW with block-wise int8 optimizer state (cf. Dettmers et al. 2022, 8-bit optimizers via block-wise quantization).
    m: signed linear int8 per block of `block` values with an fp32 absmax scale.
    v: stored as sqrt(v) in uint8 with a per-block max scale (sqrt compresses dynamic range; simpler than the dynamic
       tree quantization of bitsandbytes -> expect slightly worse fidelity; measured in tests/test_optim.py).
    State memory: 2 bytes/param + 8 bytes/block, vs 8 bytes/param for fp32 Adam (3.9x smaller at block=256).
    Small tensors (< min_8bit numel) keep fp32 state."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, block=256, min_8bit=4096):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.block, self.min8 = block, min_8bit

    def _q(self, x, signed):
        B = self.block
        flat = x.reshape(-1)
        pad = (-flat.numel()) % B
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blk = flat.view(-1, B)
        sc = blk.abs().amax(1, keepdim=True).clamp(min=1e-12)
        if signed:
            q = torch.round(blk / sc * 127).clamp(-127, 127).to(torch.int8)
        else:
            q = torch.round(blk / sc * 255).clamp(0, 255).to(torch.uint8)
        return q, sc

    def _dq(self, q, sc, shape, signed):
        x = q.float() * (sc / (127 if signed else 255))
        return x.reshape(-1)[: torch.Size(shape).numel()].view(shape)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                small = p.numel() < self.min8
                if not st:
                    st["t"] = 0
                    if small:
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    else:
                        st["m"] = self._q(torch.zeros_like(p), True)
                        st["v"] = self._q(torch.zeros_like(p), False)
                st["t"] += 1
                t = st["t"]
                gr = p.grad.float()
                if small:
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(gr, alpha=1 - b1); v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                else:
                    m = self._dq(*st["m"], p.shape, True)
                    v = self._dq(*st["v"], p.shape, False) ** 2
                    m.mul_(b1).add_(gr, alpha=1 - b1); v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    st["m"] = self._q(m, True); st["v"] = self._q(v.sqrt(), False)
                mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_((mh / (vh.sqrt() + g["eps"])).to(p.dtype), alpha=-g["lr"])

    def state_bytes(self):
        n = 0
        for st in self.state.values():
            for k in ("m", "v"):
                x = st.get(k)
                if isinstance(x, tuple):
                    n += x[0].numel() * x[0].element_size() + x[1].numel() * 4
                elif x is not None:
                    n += x.numel() * x.element_size()
        return n
