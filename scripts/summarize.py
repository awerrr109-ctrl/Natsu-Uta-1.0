"""Print a compact results table from experiments/results/*/final.json."""
import json, glob, os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for f in sorted(glob.glob(f"{ROOT}/experiments/results/{pat}/final.json")):
    d = json.load(open(f)); n = f.split("/")[-2]
    print(f"\n## {n}  params={d['info']['params']:,}  trainFLOPs={d['train_flops']:.3g}  tokens={d['tokens']:,}  peakRSS={d['peak_rss_mb']:.0f}MB  wall={d['wall_s']:.0f}s")
    for r, e in d["by_loops"].items():
        cells = []
        for k, v in e.items():
            if isinstance(v, dict):
                cells.append(f"{k}: " + (f"acc={v['acc']:.3f} " if 'acc' in v else "") + f"loss={v['loss']:.4f}" + (f" bpb={v['bpb']:.4f}" if 'acc' not in v else ""))
        print(f"  loops={r:>2} fwdFLOP/tok={e['fwd_flops_per_token']:.3g} | " + " | ".join(cells))
    if "gated" in d:
        for th, v in d["gated"].items():
            print(f"  gate th={th}: loss={v['loss']:.4f} skip={v.get('skip_frac',0):.3f}")
