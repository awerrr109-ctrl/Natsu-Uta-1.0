"""
Gated Delta Rule (Yang, Kautz, Hatamizadeh 2024, "Gated Delta Networks", arXiv:2412.06464)
pure-PyTorch implementation: chunkwise-parallel (training/prefill) + recurrent step (decode).

Recurrence (state S in R^{d_k x d_v}):
    S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T ,   o_t = S_t^T q_t
with a_t = exp(g_t) in (0,1], b_t in (0,1), ||k_t||=1.

The chunkwise form follows the WY/UT-transform derivation used in the
flash-linear-attention reference (`naive.py`), but the triangular inverse is
done with `solve_triangular` instead of an O(C^2) python loop. Correctness is
asserted against the step-by-step recurrence in tests/test_deltanet.py.
"""
import torch
import torch.nn.functional as F


def gated_delta_recurrent(q, k, v, g, beta, S0=None):
    """Reference / decode path. q,k: (B,H,L,dk) v: (B,H,L,dv) g,beta: (B,H,L). Returns o, S."""
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv) if S0 is None else S0
    outs = []
    for t in range(L):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        a = g[:, :, t].exp()[..., None, None]
        b = beta[:, :, t][..., None]
        S = a * S
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)          # k^T S  (old memory read)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, b * (vt - pred))  # delta update
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, 2), S


def gated_delta_chunk(q, k, v, g, beta, S0=None, chunk=32):
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    pad = (chunk - L % chunk) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = q.shape[2] // chunk
    rs = lambda x: x.reshape(B, H, n, chunk, *x.shape[3:])
    q, k, v, g, beta = map(rs, (q, k, v, g, beta))
    kb = k * beta[..., None]
    vb = v * beta[..., None]
    gc = g.cumsum(-1)                                            # (B,H,n,C)
    diff = gc[..., :, None] - gc[..., None, :]
    tril = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    Ldec = torch.where(tril, diff, torch.full_like(diff, -float("inf"))).exp()  # decay i<-j, j<=i
    strict = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    A = (kb @ k.transpose(-1, -2)) * Ldec * strict               # strictly lower
    I = torch.eye(chunk, device=q.device, dtype=q.dtype)
    # T = (I + A)^{-1}
    T = torch.linalg.solve_triangular(I + A, I.expand_as(A), upper=False, unitriangular=True)
    U = T @ vb                                                    # pseudo values
    W = T @ (kb * gc[..., None].exp())                            # decay-weighted keys
    S = q.new_zeros(B, H, dk, dv) if S0 is None else S0
    o = torch.empty(B, H, n, chunk, dv, device=q.device, dtype=q.dtype)
    for i in range(n):
        qi, ki = q[:, :, i], k[:, :, i]
        att = (qi @ ki.transpose(-1, -2)) * Ldec[:, :, i]         # inclusive lower (Ldec is 0 above diag)
        vnew = U[:, :, i] - W[:, :, i] @ S
        o[:, :, i] = (qi * gc[:, :, i, :, None].exp()) @ S + att @ vnew
        glast = gc[:, :, i, -1]
        S = S * glast[..., None, None].exp() + \
            (ki * (glast[..., None] - gc[:, :, i]).exp()[..., None]).transpose(-1, -2) @ vnew
    o = o.reshape(B, H, n * chunk, dv)
    return (o[:, :, :L] if pad else o), S
