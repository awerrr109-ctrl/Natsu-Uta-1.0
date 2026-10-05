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
        if p.ndim >= 2 and "embed" not in n and "values" not in n and "engram.table" not in n and "loop_emb" not in n and "loop_bias" not in n and n != "inj.weight":
            matrix.append(p)
        else:
            other.append(p)
    adam_other = torch.optim.AdamW([{"params": other, "weight_decay": 0.0}], lr=lr, betas=betas)
    if kind == "adamw":
        return [torch.optim.AdamW([{"params": matrix, "weight_decay": wd}], lr=lr, betas=betas), adam_other]
    if kind == "muon":
        return [Muon(matrix, lr=lr, wd=wd), adam_other]
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
