"""Downstream MC eval on held-out TinyStories cloze + name-consistency (in-context retrieval) sets. Byte-level checkpoints only.
usage: python -m natsu.ds_eval --ckpt checkpoints/X.pt [--n 200]  -> one JSON line"""
import argparse, json, os, torch
from natsu.model import Natsu, NatsuConfig
from natsu.data import ByteTokenizer
from natsu.eval_harness import eval_mc

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args(); torch.set_num_threads(2)
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu")
    m = Natsu(NatsuConfig(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.eval()
    if m.c.vocab_size != 260 and m.c.vocab_size < 256:
        print(json.dumps({"ckpt": a.ckpt, "skip": "non-byte vocab"})); return
    tok = ByteTokenizer()
    out = {"ckpt": a.ckpt}
    for name in ("ts_cloze", "ts_names"):
        items = json.load(open(os.path.join(ROOT, f"data_cache/evals/{name}.json")))[: a.n]
        out[name] = eval_mc(m, tok, items, norm=True)
    out["chance"] = 0.25
    print(json.dumps(out))

if __name__ == "__main__":
    main()
