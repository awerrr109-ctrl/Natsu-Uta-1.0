"""Pre-run sizing step (runs inside the queue, never in the foreground — F006).
Adjusts moe_expert_mult of a target config so its total stored params match a reference config (±1%)."""
import json, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig

def count(m):
    with torch.device("meta"):
        net = Natsu(NatsuConfig(**m))
    return sum(p.numel() for p in net.parameters())

ref, tgt = sys.argv[1], sys.argv[2]
R = json.load(open(ref)); T = json.load(open(tgt))
goal = count(R["model"])
lo, hi = 0.05, 2.0
for _ in range(30):
    mid = (lo + hi) / 2; T["model"]["moe_expert_mult"] = round(mid, 4)
    if count(T["model"]) < goal: lo = mid
    else: hi = mid
n = count(T["model"])
T["model"]["moe_expert_mult"] = round((lo + hi) / 2, 4)
T.setdefault("sizing", {})["matched_to"] = {"ref": os.path.basename(ref), "ref_params": goal, "params": n}
json.dump(T, open(tgt, "w"), indent=1)
print(f"matched {os.path.basename(tgt)}: mult={T['model']['moe_expert_mult']} params={n:,} vs ref {goal:,}")
