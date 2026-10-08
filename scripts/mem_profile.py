#!/usr/bin/env python3
"""F017: peak-RSS profile of one train step for a config, toggling suspects (MoE dense vs sparse, batch, seq).
usage: python scripts/mem_profile.py <config.json> [key=val ...]"""
import json, os, resource, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from natsu.model import Natsu, NatsuConfig
cfg = json.load(open(sys.argv[1])); over = dict(a.split("=") for a in sys.argv[2:])
mc = cfg["model"]; tc = cfg["train"]
for k, v in over.items():
    (mc if k in NatsuConfig.__dataclass_fields__ else tc)[k] = json.loads(v)
torch.set_num_threads(1)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
m = Natsu(NatsuConfig(**mc)); r0 = rss()
x = torch.randint(0, mc.get("vocab_size", 260), (tc["batch"], tc["seq"]))
out = m(x); r1 = rss()
F.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]).float(), x.reshape(-1)).backward(); r2 = rss()
print(json.dumps({"over": over, "batch": tc["batch"], "seq": tc["seq"], "build_mb": round(r0), "fwd_peak_mb": round(r1), "bwd_peak_mb": round(r2)}))
