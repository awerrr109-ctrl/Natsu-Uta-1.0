#!/usr/bin/env python3
"""F014 method: on bimodal (phase-transition) tasks, report per-seed transition and steps-to-transition, then count per arm.
usage: transition_table.py <thr_train_loss> <run_prefix>...   (arm = run name with the trailing _sN removed)"""
import json, os, re, sys, collections
R = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
thr = float(sys.argv[1]); arms = collections.defaultdict(list)
for pre in sys.argv[2:]:
    for d in sorted(os.listdir(R)):
        if not d.startswith(pre) or not os.path.isfile(os.path.join(R, d, "log.jsonl")): continue
        recs = [json.loads(l) for l in open(os.path.join(R, d, "log.jsonl")) if l.startswith("{")]
        st = next((r["step"] for r in recs if "loss" in r and r["loss"] < thr), None)
        fin = os.path.join(R, d, "final.json"); em = None
        if os.path.exists(fin):
            f = json.load(open(fin)); f = f.get("by_loops", f); f = f.get("1", f); em = f.get("main", {}).get("acc")
        arms[re.sub(r"_s\d+$", "", d)].append((d, st, em))
for a, rs in arms.items():
    n_t = sum(1 for _, st, _ in rs if st is not None)
    print(f"{a}: transitioned {n_t}/{len(rs)}  " + "  ".join(f"[{d[-2:]} step={st} EM={em}]" for d, st, em in rs))
