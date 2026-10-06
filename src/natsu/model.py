"""
Natsu model family. A single config covers all candidate architectures (C0..C4,
see docs/ARCHITECTURE.md) so that comparisons are apples-to-apples.

Layout:  embed -> prelude[n_prelude] -> (core[n_core] looped n_loops times) -> coda[n_coda] -> head
  * every block = token mixer ('gdn' | 'attn' | 'swa') + channel mixer (SwiGLU | ReLU^2) ;
    mixer pattern is given by a string such as "ggga" repeated over all layers.
  * core blocks share weights across loops; per-loop specialisation via
      - loop embedding added to the residual stream,
      - optional per-loop LoRA on the FFN (Relaxed Recursive Transformers, arXiv:2410.20672),
      - optional input re-injection of the prelude output (Huginn, arXiv:2502.05171).
  * optional per-token soft depth gate on loops r>=1 (MoR-like, arXiv:2507.10524),
    trained with a compute penalty; used at inference for loop skipping / early exit.
  * optional product-key memory FFN (Lample et al. 2019) in the coda (knowledge storage).
  * every loop count 1..n_loops is a valid model -> the same weights give a cheap
    draft model for self-speculative decoding (see natsu/generate.py).
"""
import math
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .deltanet import gated_delta_chunk, gated_delta_recurrent


@dataclass
class NatsuConfig:
    vocab_size: int = 260
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    gdn_expand_v: int = 1
    ffn_mult: float = 2.667
    ffn_act: str = "swiglu"          # swiglu | relu2
    n_prelude: int = 1
    n_core: int = 2
    n_coda: int = 1
    n_loops: int = 1
    pattern: str = "ga"              # per-layer mixer cycle: g=gdn a=attn s=sliding-window attn
    window: int = 256
    loop_lora_rank: int = 0
    reinject: bool = False
    reinject_mode: str = "concat"    # concat (Huginn, identity-init) | lti (Parcae-style stable: x <- a*x + b*norm(e), a=exp(-dt*exp(logA)) in (0,1))
    depth_gate: bool = False
    gate_mem_feat: bool = False      # N2: depth decider also sees Engram retrieval confidence (gate value) — docs/NOVELTY.md
    gate_mode: str = "soft"          # soft (FLOP-penalty, E4d) | lookahead (TaH2-style: gate is a classifier of
                                     # "does loop r improve this token?", trained on measured CE deltas; forward = no gating in training)
    pkm_keys: int = 0                # 0 disables; n_keys per sub-key set (memory = n_keys^2 slots)
    pkm_topk: int = 8
    max_seq: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    chunk: int = 32
    mtp: int = 0                     # extra multi-token-prediction heads
    init_std: float = 0.02
    grad_ckpt: bool = False          # activation recomputation per block (1GB-RAM mode)
    moe_experts: int = 0             # >0: fine-grained MoE FFN in CORE (looped) blocks
    moe_topk: int = 2
    moe_shared: int = 1              # always-on shared experts (DeepSeekMoE)
    moe_expert_mult: float = 0.0     # expert hidden = moe_expert_mult*d_model (0 -> ffn_mult/topk+shared)
    moe_loop_bias: bool = True       # per-loop router bias -> encourages routing divergence across loops
    engram_slots: int = 0            # >0: Engram-style hashed n-gram memory (arXiv:2601.07372) after prelude
    engram_orders: tuple = (2, 3)
    engram_heads: int = 4
    engram_vip: int = 0              # Engram v2 (R24 fix for Zipf cold tail / collisions): number of dedicated rows per order
                                     # reserved for the most frequent n-grams (collision-free); tail keeps hashed buckets.
                                     # VIP ids come from a frequency table built on training data (scripts/build_vip.py).
    engram_dim: int = 0              # 0 -> d_model // (len(orders)*heads) per head
    loop_kv: str = "per_loop"        # per_loop | shared_first: core attention in loops r>0 reuses loop-0 K/V
                                     # (KV cache size independent of R; train/infer consistent)

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, pos, theta):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class ShortConv(nn.Module):
    """Causal depthwise conv (k=4) with decode-time state."""
    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(d, k) / math.sqrt(k))

    def forward(self, x, state=None):  # x: (B,L,D)
        B, L, D = x.shape
        prev = state if state is not None else x.new_zeros(B, self.k - 1, D)
        xx = torch.cat([prev, x], 1)
        y = sum(xx[:, i:i + L] * self.w[:, i] for i in range(self.k))
        return F.silu(y), xx[:, -(self.k - 1):]


class GDNMixer(nn.Module):
    def __init__(self, c: NatsuConfig):
        super().__init__()
        self.h, self.dk = c.n_heads, c.head_dim
        self.dv = c.head_dim * c.gdn_expand_v
        D = c.d_model
        self.q = nn.Linear(D, self.h * self.dk, bias=False)
        self.k = nn.Linear(D, self.h * self.dk, bias=False)
        self.v = nn.Linear(D, self.h * self.dv, bias=False)
        self.ab = nn.Linear(D, 2 * self.h, bias=False)          # alpha (decay) and beta logits
        self.gate = nn.Linear(D, self.h * self.dv, bias=False)
        self.o = nn.Linear(self.h * self.dv, D, bias=False)
        self.cq, self.ck, self.cv = ShortConv(self.h * self.dk), ShortConv(self.h * self.dk), ShortConv(self.h * self.dv)
        A = torch.empty(self.h).uniform_(1, 16)
        self.A_log = nn.Parameter(A.log())
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.onorm = RMSNorm(self.dv)
        self.chunk = c.chunk

    def forward(self, x, pos, cache=None, skip=None):
        B, L, _ = x.shape
        st = cache or {}
        q, sq = self.cq(self.q(x), st.get("cq"))
        k, sk = self.ck(self.k(x), st.get("ck"))
        v, sv = self.cv(self.v(x), st.get("cv"))
        sh = lambda t, d: t.view(B, L, self.h, d).transpose(1, 2)
        q, k, v = sh(q, self.dk), sh(k, self.dk), sh(v, self.dv)
        q = F.normalize(q, dim=-1) * self.dk ** -0.5
        k = F.normalize(k, dim=-1)
        cd = torch.promote_types(x.dtype, torch.float32)   # recurrence always in >= fp32
        a, b = self.ab(x).to(cd).chunk(2, -1)
        g = (-self.A_log.exp() * F.softplus(a + self.dt_bias)).transpose(1, 2)
        beta = torch.sigmoid(b).transpose(1, 2)
        if skip is not None:  # skipped tokens leave state untouched: a=1, beta=0
            m = skip[:, None, :].float()
            g, beta = g * (1 - m), beta * (1 - m)
        S0 = st.get("S")
        if L == 1 and S0 is not None:
            o, S = gated_delta_recurrent(q.to(cd), k.to(cd), v.to(cd), g, beta, S0)
        else:
            o, S = gated_delta_chunk(q.to(cd), k.to(cd), v.to(cd), g, beta, S0, self.chunk)
        o = o.to(x.dtype).transpose(1, 2)
        o = self.onorm(o) * F.silu(self.gate(x).view(B, L, self.h, self.dv))
        new = {"cq": sq, "ck": sk, "cv": sv, "S": S} if cache is not None else None
        return self.o(o.reshape(B, L, -1)), new


class AttnMixer(nn.Module):
    def __init__(self, c: NatsuConfig, window=None):
        super().__init__()
        self.h, self.kvh, self.dh = c.n_heads, c.n_kv_heads, c.head_dim
        D = c.d_model
        self.q = nn.Linear(D, self.h * self.dh, bias=False)
        self.kv = nn.Linear(D, 2 * self.kvh * self.dh, bias=False)
        self.o = nn.Linear(self.h * self.dh, D, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)   # QK-norm for stability
        self.theta, self.window = c.rope_theta, window

    def forward(self, x, pos, cache=None, skip=None, shared_kv=None):
        B, L, _ = x.shape
        q = self.qn(self.q(x).view(B, L, self.h, self.dh)).transpose(1, 2)
        q = rope(q, pos, self.theta)
        new = None
        if shared_kv is not None:          # reuse K/V (incl. past) produced by loop 0 of this layer
            k, v = shared_kv
        else:
            k, v = self.kv(x).view(B, L, 2, self.kvh, self.dh).unbind(2)
            k, v = self.kn(k).transpose(1, 2), v.transpose(1, 2)
            k = rope(k, pos, self.theta)
            if cache is not None and "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            if cache is not None:
                kk, vv = (k, v) if self.window is None else (k[:, :, -self.window:], v[:, :, -self.window:])
                new = {"k": kk, "v": vv}
            self.last_kv = (k, v)
        T = k.shape[2]
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        qpos = torch.arange(T - L, T, device=x.device)[:, None]
        kpos = torch.arange(T, device=x.device)[None]
        mask = kpos <= qpos
        if self.window is not None:
            mask = mask & (kpos > qpos - self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(o.transpose(1, 2).reshape(B, L, -1)), new


class FFN(nn.Module):
    def __init__(self, c: NatsuConfig, n_lora=0):
        super().__init__()
        D, Hd = c.d_model, int(c.ffn_mult * c.d_model)
        self.act = c.ffn_act
        self.up = nn.Linear(D, (2 if self.act == "swiglu" else 1) * Hd, bias=False)
        self.down = nn.Linear(Hd, D, bias=False)
        r = c.loop_lora_rank
        self.lora = None
        if n_lora > 1 and r > 0:  # per-loop low-rank delta on the down projection
            self.lA = nn.Parameter(torch.randn(n_lora, Hd, r) * (1 / math.sqrt(Hd)))
            self.lB = nn.Parameter(torch.zeros(n_lora, r, D))
            self.lora = True

    def forward(self, x, loop=0):
        u = self.up(x)
        if self.act == "swiglu":
            a, b = u.chunk(2, -1)
            h = F.silu(a) * b
        else:
            h = F.relu(u).square()
        y = self.down(h)
        if self.lora:
            y = y + (h @ self.lA[loop]) @ self.lB[loop]
        return y


class MoE(nn.Module):
    """Fine-grained MoE with shared experts and aux-loss-free load balancing
    (DeepSeek-V3 style bias update, arXiv:2412.19437). In looped cores, the router also gets a
    learned per-loop bias so the same physical layer can select different experts on each pass
    ("routing divergence", arXiv:2605.09165). Dense compute via batched einsum over all experts
    (correct but not FLOP-efficient: fine for toy scale; GPU path would use grouped GEMM)."""
    def __init__(self, c: NatsuConfig, n_loops=1):
        super().__init__()
        D, E = c.d_model, c.moe_experts
        self.E, self.k, self.ns = E, c.moe_topk, c.moe_shared
        m = c.moe_expert_mult or c.ffn_mult / (c.moe_topk + c.moe_shared)
        H = max(8, int(m * D))
        self.H = H
        self.router = nn.Linear(D, E, bias=False)
        self.w_up = nn.Parameter(torch.randn(E + self.ns, D, 2 * H) * c.init_std)
        self.w_down = nn.Parameter(torch.randn(E + self.ns, H, D) * c.init_std)
        self.loop_bias = nn.Parameter(torch.zeros(n_loops, E)) if (c.moe_loop_bias and n_loops > 1) else None
        self.register_buffer("bal_bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))
        self.last_routes = None

    def forward(self, x, loop=0):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.router(xf).float()
        if self.loop_bias is not None:
            logits = logits + self.loop_bias[loop]
        scores = torch.sigmoid(logits)
        _, idx = (scores + self.bal_bias).topk(self.k, -1)      # balancing bias affects selection only
        w = scores.gather(1, idx)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        gate = torch.zeros(xf.shape[0], self.E, dtype=x.dtype, device=x.device).scatter(1, idx, w)
        if self.ns:
            gate = torch.cat([gate, torch.ones(xf.shape[0], self.ns, dtype=x.dtype, device=x.device)], 1)
        if self.training:
            with torch.no_grad():
                self.load.mul_(0.0).add_(torch.bincount(idx.flatten(), minlength=self.E).float())
        self.last_routes = idx.detach()
        h = torch.einsum("td,edh->teh", xf, self.w_up)
        a, b = h.chunk(2, -1)
        h = F.silu(a) * b * gate[..., None]
        y = torch.einsum("teh,ehd->td", h, self.w_down)
        return y.view(B, L, D)

    @torch.no_grad()
    def update_balance(self, rate=1e-3):
        if self.load.sum() > 0:
            self.bal_bias.add_(rate * torch.sign(self.load.mean() - self.load))


class Engram(nn.Module):
    """Conditional n-gram memory (Cheng et al. 2026, arXiv:2601.07372), simplified:
    - suffix n-grams (orders e.g. 2,3) of token ids, K hash heads per order, multiplicative-XOR hash into a
      table of `slots` rows (prime-sized per head in the paper; one shared table with per-head offsets here);
    - fusion: hidden state is the query, retrieved memory is key & value: g = sigmoid(<q(h), k(m)>/sqrt d), out = g * v(m);
      followed by a depthwise causal conv (k=4).
    Addresses depend ONLY on token ids -> rows can be prefetched from host RAM / SSD before the layer runs
    (the property that makes this cheaper than product-key memory for offloading, cf. R15).
    Decode-time state: the last (max_order-1) token ids + conv state."""
    PRIMES = (1000003, 1000033, 1000037, 1000039, 1000081, 1000099, 1000117, 1000121)

    def __init__(self, c: NatsuConfig):
        super().__init__()
        self.orders, self.K, self.S = tuple(c.engram_orders), c.engram_heads, c.engram_slots
        nh = len(self.orders) * self.K
        self.dm = c.engram_dim or max(8, c.d_model // nh)
        self.table = nn.Embedding(self.S * nh, self.dm)
        nn.init.normal_(self.table.weight, std=0.02)
        M = self.dm * nh
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(M, c.d_model, bias=False)
        self.v = nn.Linear(M, c.d_model, bias=False)
        self.norm = RMSNorm(c.d_model)
        self.conv = ShortConv(c.d_model)
        self.maxo = max(self.orders)
        mult = torch.tensor([1000003 + 2 * i for i in range(self.maxo)], dtype=torch.long)
        self.register_buffer("mult", mult, persistent=False)
        self.vip = c.engram_vip
        if self.vip:
            # per order: sorted 64-bit n-gram keys of VIP n-grams (filled by load_vip); dedicated table
            for n in self.orders:
                self.register_buffer(f"vip_keys_{n}", torch.full((self.vip,), -1, dtype=torch.long))
            self.vip_table = nn.Embedding(len(self.orders) * self.vip + 1, self.dm * self.K)   # +1 = "not VIP" row (zero)
            nn.init.normal_(self.vip_table.weight, std=0.02)
            with torch.no_grad():
                self.vip_table.weight[-1].zero_()

    @staticmethod
    def ngram_key(win):
        """exact (collision-free for ids < 2^20, n<=3) integer key of an n-gram window (B,L,n)."""
        k = torch.zeros(win.shape[:-1], dtype=torch.long, device=win.device)
        for j in range(win.shape[-1]):
            k = k * (1 << 20) + (win[..., j] + 1)
        return k

    def load_vip(self, keys_by_order):
        for n, keys in keys_by_order.items():
            t = torch.as_tensor(sorted(keys)[: self.vip], dtype=torch.long)
            buf = getattr(self, f"vip_keys_{n}"); buf.fill_(-1); buf[: len(t)] = t
            buf.copy_(torch.sort(buf).values)

    def vip_rows(self, full, L):
        rows = []
        for oi, n in enumerate(self.orders):
            win = torch.stack([full[:, self.maxo - 1 - j: self.maxo - 1 - j + L] for j in range(n)], -1)
            key = self.ngram_key(win)
            keys = getattr(self, f"vip_keys_{n}")
            pos = torch.searchsorted(keys, key).clamp(max=self.vip - 1)
            hit = keys[pos] == key
            rows.append(torch.where(hit, pos + oi * self.vip, torch.full_like(pos, len(self.orders) * self.vip)))
        return torch.stack(rows, -1), torch.stack([r != len(self.orders) * self.vip for r in rows], -1)

    def addresses(self, ids, prev=None):
        """ids (B,L) -> row indices (B,L,nh). prev: (B, maxo-1) previous ids for decode."""
        B, L = ids.shape
        pad = prev if prev is not None else ids.new_full((B, self.maxo - 1), -1)
        full = torch.cat([pad, ids], 1)                       # (B, L+maxo-1)
        out, h = [], 0
        for n in self.orders:
            win = torch.stack([full[:, self.maxo - 1 - j: self.maxo - 1 - j + L] for j in range(n)], -1)  # (B,L,n) suffix
            for k in range(self.K):
                hv = torch.zeros(B, L, dtype=torch.long, device=ids.device)
                for j in range(n):
                    hv = (hv * (self.mult[j] + 7919 * k)) ^ (win[..., j] + 1)
                out.append((hv.abs() % self.S) + h * self.S)
                h += 1
        return torch.stack(out, -1), full[:, -(self.maxo - 1):]

    def forward(self, x, ids, state=None):
        st = state or {}
        idx, tail = self.addresses(ids, st.get("tail"))
        m = self.table(idx)                                   # (B,L,nh,dm)
        if self.vip:
            prev = st.get("tail")
            full = torch.cat([prev if prev is not None else ids.new_full((ids.shape[0], self.maxo - 1), -1), ids], 1)
            vr, hit = self.vip_rows(full, ids.shape[1])       # (B,L,n_orders)
            vm = self.vip_table(vr).view(*vr.shape, self.K, self.dm)   # (B,L,n_orders,K,dm)
            hm = m.view(*m.shape[:2], len(self.orders), self.K, self.dm)
            m = torch.where(hit[..., None, None], vm, hm).flatten(2, 3)  # VIP n-grams bypass the hashed table
        m = m.flatten(-2)                                     # (B,L,M)
        k, v = self.k(m), self.v(m)
        g = torch.sigmoid((self.norm(self.q(x)) * self.norm(k)).sum(-1, keepdim=True) / x.shape[-1] ** 0.5)
        y, cs = self.conv(g * v, st.get("conv"))
        self.last_gate = g.detach()                           # (B,L,1) retrieval confidence, used by N2 depth decider
        return y, {"tail": tail, "conv": cs}


class PKM(nn.Module):
    """Product-key memory (Lample et al. 2019): n_keys^2 value slots, top-k sparse read."""
    def __init__(self, c: NatsuConfig):
        super().__init__()
        D, n, self.k = c.d_model, c.pkm_keys, c.pkm_topk
        self.dq = D // 2
        self.qp = nn.Linear(D, D, bias=False)
        self.keys = nn.Parameter(torch.randn(2, n, self.dq) / math.sqrt(self.dq))
        self.values = nn.EmbeddingBag(n * n, D, mode="sum")
        nn.init.normal_(self.values.weight, std=D ** -0.5)
        self.n = n

    def forward(self, x, loop=0):
        B, L, D = x.shape
        q = self.qp(x).view(B * L, 2, self.dq)
        s1 = q[:, 0] @ self.keys[0].T
        s2 = q[:, 1] @ self.keys[1].T
        v1, i1 = s1.topk(self.k, -1)
        v2, i2 = s2.topk(self.k, -1)
        sc = (v1[:, :, None] + v2[:, None, :]).view(B * L, -1)
        idx = (i1[:, :, None] * self.n + i2[:, None, :]).view(B * L, -1)
        sc, j = sc.topk(self.k, -1)
        idx = idx.gather(1, j)
        w = F.softmax(sc, -1)
        return self.values(idx, per_sample_weights=w).view(B, L, D)


class Block(nn.Module):
    def __init__(self, c: NatsuConfig, kind: str, n_lora=0, pkm=False, moe=False):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.mix = GDNMixer(c) if kind == "g" else AttnMixer(c, c.window if kind == "s" else None)
        self.ffn = PKM(c) if pkm else (MoE(c, max(1, n_lora)) if moe else FFN(c, n_lora))

    def forward(self, x, pos, cache=None, loop=0, skip=None, shared_kv=None):
        if shared_kv is not None:
            y, new = self.mix(self.n1(x), pos, cache, skip, shared_kv=shared_kv)
        else:
            y, new = self.mix(self.n1(x), pos, cache, skip)
        x = x + y
        x = x + self.ffn(self.n2(x), loop)
        return x, new


class Natsu(nn.Module):
    def __init__(self, c: NatsuConfig):
        super().__init__()
        self.c = c
        kinds = [c.pattern[i % len(c.pattern)] for i in range(c.n_prelude + c.n_core + c.n_coda)]
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.prelude = nn.ModuleList([Block(c, kinds[i]) for i in range(c.n_prelude)])
        self.engram = Engram(c) if c.engram_slots > 0 else None
        self.core = nn.ModuleList([Block(c, kinds[c.n_prelude + i], n_lora=c.n_loops, moe=c.moe_experts > 0)
                                   for i in range(c.n_core)])
        self.coda = nn.ModuleList([Block(c, kinds[c.n_prelude + c.n_core + i],
                                         pkm=(c.pkm_keys > 0 and i == c.n_coda - 1)) for i in range(c.n_coda)])
        if c.n_loops > 1:
            self.loop_emb = nn.Parameter(torch.zeros(c.n_loops, c.d_model))
            if c.reinject and c.reinject_mode == "concat":
                self.inj = nn.Linear(2 * c.d_model, c.d_model, bias=False)
            if c.reinject and c.reinject_mode == "lti":
                # Parcae (arXiv:2604.12946): A = Diag(-exp(logA)), ZOH: Abar = exp(dt*A) in (0,1) -> spectral radius < 1
                self.lti_logA = nn.Parameter(torch.zeros(c.d_model))
                self.lti_dt = nn.Parameter(torch.full((c.d_model,), -2.0))   # softplus(-2)≈0.13 -> Abar≈0.88 at init
                self.lti_B = nn.Parameter(torch.zeros(c.d_model))           # injection starts at 0 -> pure residual
                self.lti_norm = RMSNorm(c.d_model)
            if c.depth_gate:
                self.gate = nn.Linear(c.d_model + (1 if c.gate_mem_feat else 0), 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 2.0)
        self.norm = RMSNorm(c.d_model)
        self.head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.mtp = nn.ModuleList([nn.Sequential(RMSNorm(c.d_model), nn.Linear(c.d_model, c.d_model, bias=False))
                                  for _ in range(c.mtp)])
        self.apply(self._init)
        if c.n_loops > 1 and c.reinject and c.reinject_mode == "concat":
            # identity-preserving init: inj([x,e]) = x at step 0 (F005: random init destroyed the residual stream, cos≈0)
            with torch.no_grad():
                self.inj.weight.zero_()
                self.inj.weight[:, : c.d_model].copy_(torch.eye(c.d_model))
        # depth-scaled output projections
        n_eff = c.n_prelude + c.n_core * c.n_loops + c.n_coda
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=c.init_std / math.sqrt(2 * n_eff))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.c.init_std)
            if m.bias is not None and m is not getattr(self, "gate", None):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.c.init_std)

    def logits(self, h):
        h = self.norm(h)
        return F.linear(h, self.embed.weight) if self.head is None else self.head(h)

    def forward(self, idx, n_loops=None, cache=None, pos0=0, gate_threshold=None, return_hidden=False, return_inter=False):
        """cache: dict layer_name -> state (mutated/returned). n_loops overrides the loop count
        (any 1..c.n_loops is valid: smaller = cheaper draft)."""
        c = self.c
        R = c.n_loops if n_loops is None else n_loops
        B, L = idx.shape
        pos = torch.arange(pos0, pos0 + L, device=idx.device)
        x = self.embed(idx)
        new_cache = {} if cache is not None else None
        gates, skipped, gate_logits, inter = [], [], [], []
        want_inter = return_inter

        shared = {}

        def run(blk, name, x, loop=0, skip=None, core_i=None):
            skv = None
            if core_i is not None and c.loop_kv == "shared_first" and blk.kind != "g" and loop_idx[0] > 0:
                skv = shared[core_i]
            st = None if cache is None or skv is not None else cache.get(name, {})
            if c.grad_ckpt and self.training and cache is None and skv is None:
                x = torch.utils.checkpoint.checkpoint(lambda t: blk(t, pos, None, loop, skip)[0], x, use_reentrant=False)
                return x
            x, ns = blk(x, pos, st, loop, skip, shared_kv=skv)
            if core_i is not None and c.loop_kv == "shared_first" and blk.kind != "g" and loop_idx[0] == 0:
                shared[core_i] = blk.mix.last_kv
            if new_cache is not None and skv is None:
                new_cache[name] = ns
            return x
        loop_idx = [0]

        for i, b in enumerate(self.prelude):
            x = run(b, f"p{i}", x)
        if self.engram is not None:
            st = None if cache is None else cache.get("engram")
            y, ns = self.engram(x, idx, st)
            x = x + y
            if new_cache is not None:
                new_cache["engram"] = ns
        e = x
        mem_g = self.engram.last_gate if (self.engram is not None) else x.new_zeros(*x.shape[:2], 1)
        for r in range(R):
            loop_idx[0] = r
            if r > 0 and c.reinject:
                if c.reinject_mode == "lti":
                    dt = F.softplus(self.lti_dt)
                    x = torch.exp(-dt * torch.exp(self.lti_logA)) * x + dt * self.lti_B * self.lti_norm(e)
                else:
                    x = self.inj(torch.cat([x, e], -1))
            rr = min(r, c.n_loops - 1)                         # extrapolation: reuse last loop's params
            if c.n_loops > 1:
                x = x + self.loop_emb[rr]
            skip, gval = None, None
            if r > 0 and c.depth_gate and c.n_loops > 1:
                gin = x if not c.gate_mem_feat else torch.cat([x, mem_g], -1)
                gval = torch.sigmoid(self.gate(gin))        # (B,L,1) prob. of executing loop r
                gates.append(gval)
                if c.gate_mode == "lookahead":
                    gate_logits.append(self.gate(gin.detach())[..., 0])   # classifier only; no gradient into the trunk
                    if self.training and want_inter:
                        inter.append(x)                                  # state BEFORE loop r -> CE if we stopped here
                    if gate_threshold is None:
                        gval = None                                       # no soft gating in lookahead mode
                if gate_threshold is not None:
                    skip = gval[..., 0] < gate_threshold
                    skipped.append(skip.float().mean())
            x_in = x
            for i, b in enumerate(self.core):
                x = run(b, f"c{i}_r{r}", x, loop=rr, skip=skip, core_i=i)
            if gval is not None:
                if skip is not None:
                    x = torch.where(skip[..., None], x_in, x)
                else:
                    x = x_in + gval * (x - x_in)             # soft gate (training)
        for i, b in enumerate(self.coda):
            x = run(b, f"d{i}", x)
        out = {"logits": self.logits(x), "cache": new_cache}
        if gates:
            out["gate_mean"] = torch.stack([g.mean() for g in gates]).mean()
        if skipped:
            out["skip_frac"] = torch.stack(skipped).mean()
        if gate_logits:
            out["gate_logits"] = gate_logits
        if inter:
            # logits from stopping before loop r (r=1..R-1): run coda on the intermediate state (no cache)
            outs = []
            for h in inter:
                for i, b in enumerate(self.coda):
                    h = b(h, pos, None, 0, None)[0]
                outs.append(self.logits(h))
            out["inter_logits"] = outs
        if self.mtp:
            out["mtp_logits"] = [self.logits(m(x)) for m in self.mtp]
        if return_hidden:
            out["hidden"] = x
        return out

    # ---------------------------------------------------------------- accounting
    def param_count(self, exclude_embed=False):
        n = sum(p.numel() for p in self.parameters())
        return n - (self.embed.weight.numel() if exclude_embed else 0)

    def flops_per_token(self, n_loops=None, seq=1024, gate_exec=1.0):
        """Analytical forward FLOPs/token (2*MACs) incl. attention score cost at avg context seq/2."""
        c = self.c
        R = c.n_loops if n_loops is None else n_loops

        def blk(b):
            f = 0
            for m in b.modules():
                if isinstance(m, nn.Linear):
                    f += 2 * m.in_features * m.out_features
            if isinstance(b.ffn, MoE):
                mo = b.ffn
                f += 2 * c.d_model * mo.E + (mo.k + mo.ns) * (2 * c.d_model * 2 * mo.H + 2 * mo.H * c.d_model)
            if isinstance(b.ffn, PKM):
                f += 2 * 2 * c.pkm_keys * (c.d_model // 2) + 2 * c.pkm_topk * c.d_model
            if isinstance(b.ffn, FFN) and b.ffn.lora:
                f += 2 * b.ffn.lA.shape[1] * b.ffn.lA.shape[2] + 2 * b.ffn.lB.shape[1] * b.ffn.lB.shape[2]
            if b.kind == "a":
                f += 2 * 2 * c.n_heads * c.head_dim * (seq / 2)
            elif b.kind == "s":
                f += 2 * 2 * c.n_heads * c.head_dim * min(c.window, seq / 2)
            else:  # gdn state update + read ~ 4*dk*dv per head
                f += 4 * 2 * c.n_heads * c.head_dim * c.head_dim * c.gdn_expand_v
            return f
        total = sum(blk(b) for b in self.prelude) + sum(blk(b) for b in self.coda)
        core = sum(blk(b) for b in self.core)
        total += core * (1 + (R - 1) * (gate_exec if c.depth_gate else 1.0))
        if self.engram is not None:
            en = self.engram
            total += 2 * c.d_model * c.d_model * 3 + 2 * 2 * en.table.embedding_dim * len(en.orders) * en.K * c.d_model
        total += 2 * c.d_model * c.vocab_size
        return total
