"""
Benchmark harness for 50M+ stages (deliverable F). Implements log-likelihood multiple-choice scoring (HellaSwag/ARC/PIQA
style), exact-match generation (GSM8K-style with answer extraction) and code pass@k with sandboxed execution.
Datasets are read from local JSONL so the harness runs offline. Prepare once with `scripts/fetch_evals.py` (HF hub).
Format per line:
  mc:   {"ctx": str, "choices": [str,...], "label": int}
  gen:  {"prompt": str, "answer": str}
  code: {"prompt": str, "tests": str}   # tests = python asserts executed in a subprocess with timeout
Status: mc + gen paths unit-tested on synthetic items (tests/test_harness.py); code path tested on a trivial item.
At toy scale these numbers are near chance and are NOT reported as capability claims.
"""
import json, math, os, re, subprocess, sys, tempfile
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.generate import generate, sample_n


@torch.no_grad()
def loglik(model, tok, ctx, cont, n_loops=None):
    ci, xi = tok.encode(ctx, bos=True), tok.encode(cont)
    ids = torch.tensor([ci + xi])
    lp = F.log_softmax(model(ids[:, :-1], n_loops=n_loops)["logits"].float(), -1)
    tgt = ids[0, 1:]
    s = lp[0, len(ci) - 1:, :].gather(1, tgt[len(ci) - 1:, None]).sum().item()
    return s, len(xi)


def eval_mc(model, tok, items, norm=True, n_loops=None):
    ok = 0
    for it in items:
        sc = []
        for ch in it["choices"]:
            s, n = loglik(model, tok, it["ctx"], ch, n_loops)
            sc.append(s / max(1, n) if norm else s)
        ok += int(max(range(len(sc)), key=sc.__getitem__) == it["label"])
    return ok / max(1, len(items))


def extract_number(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else None


def eval_gen(model, tok, items, max_new=64, n_samples=1, n_loops=None):
    from natsu.generate import majority_vote
    ok = 0
    for it in items:
        x = torch.tensor([tok.encode(it["prompt"], bos=True)])
        if n_samples == 1:
            outs = [generate(model, x, max_new, n_loops=n_loops)[0, x.shape[1]:].tolist()]
        else:
            outs = sample_n(model, x, n_samples, max_new, n_loops=n_loops)
        ans = [extract_number(tok.decode(o)) for o in outs]
        ok += int(majority_vote(ans) == extract_number(it["answer"]))
    return ok / max(1, len(items))


def run_code(program, tests, timeout=5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program + "\n\n" + tests + "\n")
        p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(p)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def eval_code(model, tok, items, n=8, k=1, max_new=128):
    tot = 0.0
    for it in items:
        x = torch.tensor([tok.encode(it["prompt"], bos=True)])
        outs = sample_n(model, x, n, max_new)
        c = sum(run_code(it["prompt"] + tok.decode(o), it["tests"]) for o in outs)
        tot += pass_at_k(n, c, k)
    return tot / max(1, len(items))
