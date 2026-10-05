"""Build the Pareto table (quality vs stored params vs train FLOPs vs inference FLOPs) for LM runs.
Reads experiments/results/*/final.json (+bench.json if present). Writes docs/generated/pareto_toy.md.
A run is Pareto-optimal if no other run is <= on all cost axes and strictly better bpb."""
import glob, json, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = []
for f in sorted(glob.glob(f"{ROOT}/experiments/results/E[34]*/final.json")):
    d = json.load(open(f)); n = f.split("/")[-2]
    for r, e in d["by_loops"].items():
        m = e["main"]
        if "acc" in m:
            continue
        rows.append(dict(run=n, loops=int(r), bpb=m["bpb"], params=d["info"]["params"], train_flops=d["train_flops"],
                         inf_flops=e["fwd_flops_per_token"], tokens=d["tokens"], rss=d["peak_rss_mb"], wall=d["wall_s"]))
def dominated(a, b):  # b dominates a
    return (b["params"] <= a["params"] and b["train_flops"] <= a["train_flops"] and b["inf_flops"] <= a["inf_flops"]
            and b["bpb"] <= a["bpb"] and (b["bpb"] < a["bpb"] or b["inf_flops"] < a["inf_flops"]))
for a in rows:
    a["pareto"] = not any(dominated(a, b) for b in rows if b is not a)
rows.sort(key=lambda r: r["bpb"])
L = ["| run | loops@eval | val bpb | stored params | train FLOPs | inf MFLOP/tok | train tokens | peak RSS MB | wall s | Pareto |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['run']} | {r['loops']} | {r['bpb']:.4f} | {r['params']:,} | {r['train_flops']:.3g} | {r['inf_flops']/1e6:.2f} | "
             f"{r['tokens']:,} | {r['rss']:.0f} | {r['wall']:.0f} | {'**yes**' if r['pareto'] else ''} |")
os.makedirs(f"{ROOT}/docs/generated", exist_ok=True)
open(f"{ROOT}/docs/generated/pareto_toy.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
