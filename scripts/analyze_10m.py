#!/usr/bin/env python3
"""Apply the pre-registered 10M rules (evidence_log, 2026-10-08 09:25) to L10a-e results."""
import json, os
R = "experiments/results"; T = 0.010
def b(n):
    p = f"{R}/{n}/final.json"
    if not os.path.exists(p): return None
    f = json.load(open(p)); f = f.get("by_loops", f); f = f.get("1", f); return f["main"]["bpb"]
a, bb, c, d, e = (b(n) for n in ["L10a_C4_10M", "L10b_C0like_10M_noEngram", "L10c_dense_10M_noEngram_noMoE", "L10d_dense_10M_engram", "L10e_C4_10M_engramlr5"])
out = {"bpb": dict(L10a=a, L10b=bb, L10c=c, L10d=d, L10e=e)}
if a and bb: out["R1_engram_moe"] = {"delta": a - bb, "holds": a < bb - T}
if c and d: out["R2_engram_dense"] = {"delta": d - c, "holds": d < c - T}
if bb and c: out["R3_moe_vs_dense"] = {"delta": bb - c, "holds": bb < c - T}
if a and e: out["R4_lr20_vs_lr5"] = {"delta": a - e, "keep_lr20": a <= e - 0.005}
if a and bb: out["transfer_ratio_vs_toy"] = (bb - a) / 0.093
print(json.dumps(out, indent=1))
