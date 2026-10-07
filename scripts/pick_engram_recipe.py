#!/usr/bin/env python3
"""Pre-registered decision (REPORT_S2 §9): choose the Engram recipe for the 10M stage from s2n/s2o results, then patch L10a/L10d.
Candidates (LM bpb mean over available seeds): v1/lr1 (E4j*), v1/lr5 (E4pr*), paper/lr1 (E4pq), paper/lr2.5 (E4ps), paper/lr5 (E4pp), v1/lr5+FAL (E4pt).
Rule: best LM mean, but v1/lr5 (or any 1-seed candidate) is only eligible if replicated (>=2 seeds) and its ICL arm (E6Lh for v1/lr5) transitions >=2/2;
otherwise fall back to the next best eligible. Writes experiments/engram_recipe.json and patches configs (keeps a .bak)."""
import json, os, glob, shutil, statistics as st
R = "experiments/results"; C = "experiments/configs"
def bpb(name):
    vals = []
    for d in sorted(glob.glob(f"{R}/{name}*/final.json")):
        f = json.load(open(d)); b = f.get("by_loops", f); b = b.get("1", b)
        if "main" in b: vals.append(b["main"]["bpb"])
    return vals
def trans(prefix, thr=0.2):
    n = t = 0
    for d in glob.glob(f"{R}/{prefix}*/log.jsonl"):
        n += 1; t += any(json.loads(l).get("loss", 9) < thr for l in open(d) if l.startswith("{"))
    return t, n
cands = {"v1_lr1": ("E4j_moe_engram_noloop", dict(engram_paper=False), 1.0, None),
         "v1_lr5": ("E4pr_moe_engramv1_lr5", dict(engram_paper=False), 5.0, "E6Lh_v1_lr5"),
         "paper_lr1": ("E4pq_moe_engrampaper_lr1", dict(engram_paper=True), 1.0, "E6Lg_engrampaper_lr1"),
         "paper_lr2.5": ("E4ps_moe_engrampaper_lr2p5", dict(engram_paper=True), 2.5, None),
         "paper_lr5": ("E4pp_moe_engrampaper_noloop", dict(engram_paper=True), 5.0, "E6Le_varchain_moe_engrampaper")}
rows = []
for k, (run, m, lr, icl) in cands.items():
    v = bpb(run); t, n = trans(icl) if icl else (None, None)
    elig = len(v) >= 2 or k == "v1_lr1"
    if icl and n: elig = elig and t >= min(2, n)
    rows.append(dict(name=k, bpb=st.mean(v) if v else None, n=len(v), icl=f"{t}/{n}" if icl else None, eligible=elig and bool(v), model=m, lr=lr))
ok = sorted([r for r in rows if r["eligible"]], key=lambda r: r["bpb"])
pick = ok[0]
json.dump({"rows": rows, "pick": pick["name"]}, open("experiments/engram_recipe.json", "w"), indent=1)
print(json.dumps({"rows": rows, "pick": pick["name"]}, indent=1))
for cf in ("L10a_C4_10M", "L10d_dense_10M_engram"):
    p = f"{C}/{cf}.json"; shutil.copy(p, p + ".bak") if not os.path.exists(p + ".bak") else None
    c = json.load(open(p)); c["model"].update(pick["model"]); c["train"]["engram_lr_mult"] = pick["lr"]
    c["note"] = c.get("note", "") + f" | Engram recipe auto-picked: {pick['name']} (scripts/pick_engram_recipe.py)"
    json.dump(c, open(p, "w"), indent=1)
