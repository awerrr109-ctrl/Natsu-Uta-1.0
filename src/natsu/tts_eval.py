"""
Test-time-scaling evaluation on a verifiable task (principle #13: report gain / extra compute).

Task: multi-digit addition with reversed digits (data.gen_add) -> exact answers, programmatic verifier.
Strategies, all measured in analytic FLOPs:
  greedy@R           : 1 sample, n_loops=R (latent depth scaling, for looped models)
  maj@N              : N samples (T=0.7) + majority vote (self-consistency)
  oracle@N           : pass@N (upper bound with a perfect verifier: tool/executor case)
Output: experiments/results/<run>/tts.json with accuracy vs FLOPs curve per strategy.
"""
import argparse, json, os, random, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.bench import load
from natsu.data import gen_add, EOS
from natsu.generate import generate, sample_n, majority_vote
from natsu.train import ROOT


def parse(tokens):
    s = bytes([t for t in tokens if t < 256]).decode("utf-8", "replace")
    out = ""
    for ch in s:
        if ch.isdigit():
            out += ch
        else:
            break
    return out or None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_problems", type=int, default=64)
    ap.add_argument("--digits", type=int, default=5)
    ap.add_argument("--Ns", default="1,4,16")
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, name = load(a.ckpt)
    rng = random.Random(4321)
    probs = []
    for _ in range(a.n_problems):
        ids, mask = gen_add(rng, a.digits)
        k = mask.index(1)
        probs.append((torch.tensor([ids[:k]]), parse(ids[k:])))
    Rs = sorted(set([1, m.c.n_loops, m.c.n_loops + 1])) if m.c.n_loops > 1 else [1]
    res = {"name": name, "digits": a.digits, "n": a.n_problems, "curves": {}}
    for R in Rs:
        fpt = m.flops_per_token(n_loops=R, seq=32)
        acc = sum(parse(generate(m, x, a.digits + 2, n_loops=R)[0, x.shape[1]:].tolist()) == y for x, y in probs) / len(probs)
        res["curves"][f"greedy@R{R}"] = {"acc": acc, "flops_per_problem": fpt * (probs[0][0].shape[1] + a.digits + 2)}
        for N in [int(v) for v in a.Ns.split(",") if int(v) > 1]:
            maj = orc = 0
            for x, y in probs:
                ans = [parse(s) for s in sample_n(m, x, N, a.digits + 2, temperature=0.7, top_k=10, n_loops=R)]
                maj += majority_vote(ans) == y
                orc += y in ans
            fl = N * fpt * (probs[0][0].shape[1] + a.digits + 2)
            res["curves"][f"maj@{N}_R{R}"] = {"acc": maj / len(probs), "flops_per_problem": fl}
            res["curves"][f"oracle@{N}_R{R}"] = {"acc": orc / len(probs), "flops_per_problem": fl}
        # adaptive-N (INFERENCE_SPEC v0.2 / R52): rounds of 4, stop at vote margin 2, cap 16 -> accuracy vs mean samples used
        from natsu.generate import adaptive_vote
        ok = used = 0
        for x, y in probs:
            a_, n_, _ = adaptive_vote(lambda k: [parse(s_) for s_ in sample_n(m, x, k, a.digits + 2, temperature=0.7, top_k=10, n_loops=R)],
                                      round_size=4, max_n=16, margin=2)
            ok += a_ == y; used += n_
        res["curves"][f"adaptive_R{R}"] = {"acc": ok / len(probs), "mean_samples": used / len(probs),
                                           "flops_per_problem": used / len(probs) * fpt * (probs[0][0].shape[1] + a.digits + 2)}
        print(json.dumps({k: v for k, v in res["curves"].items() if k.endswith(f"R{R}")}), flush=True)
    os.makedirs(os.path.join(ROOT, "experiments", "results", name), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "experiments", "results", name, "tts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
