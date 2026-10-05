"""
Inference for Natsu models.

1. generate()            : standard incremental decoding with recurrent/KV caches.
2. loop_speculative()    : SELF-speculative decoding where the *draft* is the same weights
                           run with fewer loops (r_draft < R) and the *target* is full depth.
                           No extra parameters; draft/target share prelude & embeddings.
                           Exactness: greedy acceptance -> output identical to target greedy.
                           Implementation note (honest): to keep the code simple and exactly
                           correct for both attention and recurrent (GDN) layers, draft and
                           target keep SEPARATE caches and the target cache is rebuilt by
                           re-running the accepted prefix chunk in one parallel call. This
                           is the standard draft/verify loop; the cost model is reported by
                           acceptance rate, not wall-clock (CPU wall-clock is not meaningful
                           for GPU deployment).
3. adaptive_depth()      : per-sequence test-time compute: increase loops until the
                           next-token distribution stops changing (KL < tol) — a training-free
                           halting rule (cf. Huginn's KL-based exit, arXiv:2502.05171).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new, n_loops=None, temperature=0.0, top_k=None, eos=None):
    model.eval()
    out = model(idx, n_loops=n_loops, cache={})
    cache, pos = out["cache"], idx.shape[1]
    toks = []
    logits = out["logits"][:, -1]
    for _ in range(max_new):
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            lg = logits / temperature
            if top_k:
                v, _ = lg.topk(top_k)
                lg[lg < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
        toks.append(nxt)
        if eos is not None and (nxt == eos).all():
            break
        out = model(nxt, n_loops=n_loops, cache=cache, pos0=pos)
        cache, pos = out["cache"], pos + 1
        logits = out["logits"][:, -1]
    return torch.cat([idx] + toks, 1)


@torch.no_grad()
def loop_speculative(model, idx, max_new, r_draft=1, r_target=None, k=4, eos=None):
    """Greedy self-speculative decoding (batch=1). Returns (tokens, stats)."""
    model.eval()
    R = r_target or model.c.n_loops
    assert idx.shape[0] == 1
    seq = idx
    t_out = model(seq, n_loops=R, cache={})
    t_cache = t_out["cache"]
    t_next = t_out["logits"][:, -1].argmax(-1, keepdim=True)    # target's next token (always valid)
    d_out = model(seq, n_loops=r_draft, cache={})
    d_cache = d_out["cache"]
    proposed = accepted = target_calls = draft_calls = 0
    produced = 0
    while produced < max_new:
        # the target's next token is known; append it
        seq = torch.cat([seq, t_next], 1); produced += 1
        if eos is not None and t_next.item() == eos:
            break
        # draft k tokens from the draft model (catch draft cache up by feeding t_next)
        d_pos = seq.shape[1] - 1
        o = model(t_next, n_loops=r_draft, cache=d_cache, pos0=d_pos); draft_calls += 1
        d_cache = o["cache"]
        drafts = []
        nxt = o["logits"][:, -1].argmax(-1, keepdim=True)
        snap = d_cache
        for j in range(k):
            drafts.append(nxt)
            o = model(nxt, n_loops=r_draft, cache=d_cache, pos0=d_pos + 1 + j); draft_calls += 1
            d_cache = o["cache"]
            nxt = o["logits"][:, -1].argmax(-1, keepdim=True)
        dr = torch.cat(drafts, 1)
        proposed += k
        # verify: one parallel target call over [t_next] + drafts
        inp = torch.cat([t_next, dr], 1)
        o = model(inp, n_loops=R, cache=t_cache, pos0=seq.shape[1] - 1); target_calls += 1
        preds = o["logits"].argmax(-1)                           # preds[:, i] = target token after inp[:, :i+1]
        n_ok = 0
        while n_ok < k and preds[0, n_ok].item() == dr[0, n_ok].item():
            n_ok += 1
        accepted += n_ok
        acc_toks = dr[:, :n_ok]
        if n_ok:
            seq = torch.cat([seq, acc_toks], 1); produced += n_ok
        t_next = preds[:, n_ok:n_ok + 1]
        # rebuild caches to the accepted prefix (exact for recurrent layers): re-run from the
        # cache *before* this verify step on the accepted tokens only.
        keep = torch.cat([inp[:, :1], acc_toks], 1)
        o = model(keep, n_loops=R, cache=t_cache, pos0=seq.shape[1] - keep.shape[1])
        t_cache = o["cache"]
        o = model(acc_toks, n_loops=r_draft, cache=snap, pos0=seq.shape[1] - n_ok) if n_ok else {"cache": snap}
        d_cache = o["cache"]
        if eos is not None and eos in acc_toks[0].tolist():
            break
    stats = {"proposed": proposed, "accepted": accepted, "accept_rate": accepted / max(1, proposed),
             "target_calls": target_calls, "draft_calls": draft_calls, "tokens": produced,
             "tokens_per_target_call": produced / max(1, target_calls)}
    return seq[:, : idx.shape[1] + max_new], stats


@torch.no_grad()
def adaptive_depth_logits(model, idx, r_max=None, tol=1e-3):
    """Return (logits at last position, loops used) using KL-convergence halting over loops."""
    model.eval()
    r_max = r_max or model.c.n_loops * 2
    prev = None
    for r in range(1, r_max + 1):
        lg = model(idx, n_loops=r)["logits"][:, -1].float()
        lp = F.log_softmax(lg, -1)
        if prev is not None:
            kl = F.kl_div(lp, prev, log_target=True, reduction="batchmean").item()
            if kl < tol:
                return lg, r
        prev = lp
    return lg, r_max
