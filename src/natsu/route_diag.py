"""Expert-selection-collapse diagnostic for looped MoE (LOOM, arXiv:2610.01153).
For each core MoE block: Jaccard overlap of top-k expert sets for the same token across loops r and r', and the cosine of per-loop expert-load vectors.
Also (INFERENCE_SPEC v0.3, 1 GB paging): replays the decode-order expert access trace (token-major; per token: loop r, block b, its top-k experts)
through an LRU expert cache of C entries and reports the hit rate vs C, plus the I/O ratio = misses per token / distinct experts per token at R=1.
usage: python -m natsu.route_diag --ckpt checkpoints/X.pt [--valid data_cache/ts_valid.bin] [--cache 2,4,8,16]"""
import argparse, json, os, torch
from natsu.model import Natsu, NatsuConfig, MoE
from natsu import data as D

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def lru_sim(rec, n_blocks, sizes):
    """rec[i] = list over calls (loop-major: loop0 block i, loop1 block i, ...) of (T,k) routes. Expert id = (block, e).
    Decode order per token t: for r in loops: for block b: access its k experts. Returns hit rate and misses/token per cache size."""
    from collections import OrderedDict
    R = len(rec[0]); T = rec[0][0].shape[0]
    trace = []
    for t in range(T):
        for r in range(R):
            for b in range(n_blocks):
                trace.append([(b, int(e)) for e in rec[b][r][t]])
    res = {}
    for C in sizes:
        lru, hit, tot = OrderedDict(), 0, 0
        for acc in trace:
            for key in acc:
                tot += 1
                if key in lru:
                    hit += 1; lru.move_to_end(key)
                else:
                    lru[key] = 1
                    if len(lru) > C:
                        lru.popitem(last=False)
        res[str(C)] = {"hit": hit / tot, "misses_per_token": (tot - hit) / T, "accesses_per_token": tot / T}
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--valid", default="data_cache/ts_valid.bin"); ap.add_argument("--cache", default="2,4,8,12,16")
    a = ap.parse_args(); torch.set_num_threads(2)
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu")
    m = Natsu(NatsuConfig(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.eval()
    R = m.c.n_loops
    if m.c.moe_experts == 0:
        print(json.dumps({"ckpt": a.ckpt, "skip": "not MoE"})); return
    x, _, _ = D.MemmapLoader(os.path.join(ROOT, a.valid), 128, 8, 777).fixed_eval(1)[0]
    moes = [mod for mod in m.core.modules() if isinstance(mod, MoE)]
    rec = {i: [] for i in range(len(moes))}
    hooks = []
    for i, mo in enumerate(moes):
        hooks.append(mo.register_forward_hook(lambda mod, inp, out, i=i: rec[i].append(mod.last_routes.clone())))
    m(x, n_loops=R)
    for h in hooks:
        h.remove()
    out = {"ckpt": a.ckpt, "loops": R, "blocks": [], "cache_sim": lru_sim(rec, len(moes), [int(c) for c in a.cache.split(",")])}
    for i, routes in rec.items():
        if len(routes) < 2:
            continue
        E = moes[i].E; jac, cos = [], []
        for r in range(len(routes)):
            for s in range(r + 1, len(routes)):
                A = torch.zeros(routes[r].shape[0], E).scatter(1, routes[r], 1.0); B = torch.zeros_like(A).scatter(1, routes[s], 1.0)
                inter = (A * B).sum(1); uni = ((A + B) > 0).float().sum(1)
                jac.append((inter / uni).mean().item())
                la, lb = A.sum(0), B.sum(0); cos.append(torch.nn.functional.cosine_similarity(la, lb, dim=0).item())
        # chance Jaccard for random top-k of E: k/(2E-k)
        k = routes[0].shape[1]
        out["blocks"].append({"block": i, "mean_jaccard_same_token": sum(jac) / len(jac), "chance_jaccard": k / (2 * E - k),
                              "mean_load_cos": sum(cos) / len(cos)})
    print(json.dumps(out))

if __name__ == "__main__":
    main()
