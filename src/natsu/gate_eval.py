"""Evaluate a looped checkpoint's depth decider (N2 / R21): does the gate predict when an extra loop helps?
For each eval token and loop r>=1: label = CE(stop before r) - CE(full depth) > margin. Report AUC of gate prob,
and the compute/quality trade-off of hard skipping at thresholds. Also the oracle: skip exactly where label==0."""
import argparse, json, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.bench import load
from natsu.data import MemmapLoader
from natsu.train import ROOT, evaluate


def auc(scores, labels):
    s = torch.cat(scores); l = torch.cat(labels).bool()
    if l.all() or (~l).all():
        return float("nan")
    order = s.argsort(); ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    npos = l.sum().item(); nneg = (~l).sum().item()
    return ((ranks[l].sum().item() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--valid", default="data_cache/ts_valid.bin")
    ap.add_argument("--margin", type=float, default=0.02); a = ap.parse_args()
    torch.set_num_threads(2)
    m, name = load(a.ckpt); m.train(False)
    ev = MemmapLoader(os.path.join(ROOT, a.valid), 128, 16).fixed_eval(6, seed=2025)
    S, L = [], []
    for x, y, _ in ev:
        m.training = True                       # enable intermediate exits collection (no dropout in model)
        o = m(x, return_inter=True); m.training = False
        ce = lambda lg: F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none")
        full = ce(o["logits"])
        for j, il in enumerate(o.get("inter_logits", [])):
            S.append(torch.sigmoid(o["gate_logits"][j].reshape(-1).float())); L.append((ce(il) - full > a.margin).float())
    res = {"name": name, "auc": auc(S, L), "frac_tokens_helped": torch.cat(L).mean().item() if L else None}
    res["skip"] = {th: evaluate(m, ev, gate_threshold=th) for th in (0.3, 0.5, 0.7)}
    res["full"] = evaluate(m, ev)
    json.dump(res, open(os.path.join(ROOT, "experiments", "results", name, "gate_eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
