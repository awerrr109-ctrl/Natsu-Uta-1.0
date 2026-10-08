"""Build the Pareto table (quality vs stored params vs train FLOPs vs inference FLOPs) for LM runs.
Reads experiments/results/*/final.json (+bench.json if present). Writes docs/generated/pareto_toy.md.
A run is Pareto-optimal if no other run is <= on all cost axes and strictly better bpb."""
import glob, json, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = []
for f in sorted(glob.glob(f"{ROOT}/experiments/results/E[345]*/final.json")):
    d = json.load(open(f)); n = f.split("/")[-2]
    if any(n.startswith(p) for p in ("E5c",)) or "_s1" in n or "_s2" in n:
        continue                                   # seeds are folded in below; controls are not candidates
    cfgp = f"{ROOT}/experiments/configs/{n}.json"
    teacher_flops = 0.0
    if os.path.exists(cfgp):
        kd = json.load(open(cfgp))["train"].get("distill")
        if kd:   # charge teacher forward on every student token (+ its own training, reported separately)
            tn = os.path.basename(kd["teacher"]).replace(".pt", "")
            tf = json.load(open(f"{ROOT}/experiments/results/{tn}/final.json"))
            teacher_flops = d["tokens"] * tf["info"]["flops_per_token_fwd"] + tf["train_flops"]
    seeds = []
    for k in (1, 2):
        sp = f"{ROOT}/experiments/results/{n}_s{k}/final.json"
        if os.path.exists(sp):
            seeds.append(json.load(open(sp)))
    for r, e in d["by_loops"].items():
        m = e["main"]
        if "acc" in m:
            continue
        vals = [m["bpb"]] + [sd["by_loops"][r]["main"]["bpb"] for sd in seeds if r in sd["by_loops"]]
        rows.append(dict(run=n + (f" (n={len(vals)})" if len(vals) > 1 else ""), loops=int(r), bpb=sum(vals) / len(vals),
                         spread=(max(vals) - min(vals)) if len(vals) > 1 else None, params=d["info"]["params"],
                         train_flops=d["train_flops"] + teacher_flops,
                         inf_flops=e["fwd_flops_per_token"], tokens=d["tokens"], rss=d["peak_rss_mb"], wall=d["wall_s"]))
def dominated(a, b):  # b dominates a
    return (b["params"] <= a["params"] and b["train_flops"] <= a["train_flops"] and b["inf_flops"] <= a["inf_flops"]
            and b["bpb"] <= a["bpb"] and (b["bpb"] < a["bpb"] or b["inf_flops"] < a["inf_flops"]))
for a in rows:
    a["pareto"] = not any(dominated(a, b) for b in rows if b is not a)
rows.sort(key=lambda r: r["bpb"])
L = ["Train FLOPs include teacher forward + teacher training for KD runs. bpb = seed mean where replicates exist (spread in brackets).", "",
     "| run | loops@eval | val bpb | stored params | train FLOPs | inf MFLOP/tok | train tokens | peak RSS MB | wall s | Pareto |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    r['sp'] = '' if r['spread'] is None else ' [%.3f]' % r['spread']
    L.append(f"| {r['run']} | {r['loops']} | {r['bpb']:.4f}{r['sp']} | {r['params']:,} | {r['train_flops']:.3g} | {r['inf_flops']/1e6:.2f} | "
             f"{r['tokens']:,} | {r['rss']:.0f} | {r['wall']:.0f} | {'**yes**' if r['pareto'] else ''} |")
os.makedirs(f"{ROOT}/docs/generated", exist_ok=True)
open(f"{ROOT}/docs/generated/pareto_toy.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
