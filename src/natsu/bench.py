"""
Benchmark a trained checkpoint along the Pareto axes (principle #26):
  quality (bpb at each loop count), params (stored / active), train FLOPs + tokens (from final.json),
  inference FLOPs/token (analytic), decode latency (CPU wall-clock, batch 1), peak RSS,
  loop-speculative acceptance rate & tokens per target call, adaptive-depth average loops.

Usage: python -m natsu.bench --ckpt checkpoints/NAME.pt [--prompts 4] [--new 48]
Writes experiments/results/NAME/bench.json
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.model import Natsu, NatsuConfig
from natsu.generate import generate, loop_speculative, adaptive_depth_logits
from natsu.data import MemmapLoader, ByteTokenizer
from natsu.train import evaluate, peak_rss_mb, ROOT


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = Natsu(NatsuConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck.get("run", os.path.basename(path)[:-3])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid", default="data_cache/ts_valid.bin")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, name = load(a.ckpt)
    R = m.c.n_loops
    out = {"name": name, "params": m.param_count(), "config": m.c.to_dict()}
    ev = MemmapLoader(os.path.join(ROOT, a.valid), a.seq, 16).fixed_eval(12, seed=2024)
    loops = sorted(set([1, R, R + 1, 2 * R]) if R > 1 else [1])
    out["quality_by_loops"] = {r: {**evaluate(m, ev, n_loops=r), "fwd_flops_per_token": m.flops_per_token(n_loops=r, seq=a.seq)}
                               for r in loops}
    # adaptive depth on next-token prediction at random positions
    tok = ByteTokenizer()
    data = MemmapLoader(os.path.join(ROOT, a.valid), 96, 1, seed=7)
    used, agree = [], 0
    for _ in range(16 if R > 1 else 0):
        x, y, _ = data.get()
        lg, r = adaptive_depth_logits(m, x, r_max=2 * R, tol=2e-3)
        full = m(x, n_loops=R)["logits"][:, -1]
        agree += int(lg.argmax() == full.argmax()); used.append(r)
    if used:
        out["adaptive_depth"] = {"mean_loops": sum(used) / len(used), "argmax_agree_with_R": agree / len(used)}
    # decode latency + speculation
    lat, spec = [], []
    for i in range(a.prompts):
        x, _, _ = MemmapLoader(os.path.join(ROOT, a.valid), 32, 1, seed=100 + i).get()
        t = time.time(); ref = generate(m, x, a.new, n_loops=R); lat.append((time.time() - t) / a.new)
        if R > 1:
            t = time.time(); o, st = loop_speculative(m, x, a.new, r_draft=1, r_target=R, k=4)
            st["wall_s_per_tok"] = (time.time() - t) / a.new
            st["exact_match_with_target"] = bool(torch.equal(o, ref))
            spec.append(st)
        if i == 0:
            out["sample"] = tok.decode(ref[0].tolist())
    out["decode_s_per_token_cpu"] = sum(lat) / len(lat)
    if spec:
        out["loop_speculation"] = {k: sum(s[k] for s in spec) / len(spec) for k in
                                   ("accept_rate", "tokens_per_target_call", "wall_s_per_tok")}
        out["loop_speculation"]["all_exact"] = all(s["exact_match_with_target"] for s in spec)
    out["peak_rss_mb"] = peak_rss_mb()
    fj = os.path.join(ROOT, "experiments", "results", name, "final.json")
    if os.path.exists(fj):
        f = json.load(open(fj)); out["train_flops"] = f["train_flops"]; out["train_tokens"] = f["tokens"]
    json.dump(out, open(os.path.join(ROOT, "experiments", "results", name, "bench.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("config",)}, indent=1)[:3000])


if __name__ == "__main__":
    main()
