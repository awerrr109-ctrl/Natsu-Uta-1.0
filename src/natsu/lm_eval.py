"""Evaluate a checkpoint's bpb on a memmap validation set (collateral-damage check after edits/fine-tunes).
usage: python -m natsu.lm_eval --ckpt checkpoints/X.pt [--valid data_cache/ts_valid.bin] [--batches 12]"""
import argparse, json, os, torch
from natsu.model import Natsu, NatsuConfig
from natsu import data as D
from natsu import train as T

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid", default="data_cache/ts_valid.bin"); ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(2)
    ck = torch.load(os.path.join(ROOT, a.ckpt) if not os.path.isabs(a.ckpt) else a.ckpt, map_location="cpu")
    m = Natsu(NatsuConfig(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.eval()
    ev = D.MemmapLoader(os.path.join(ROOT, a.valid), a.seq, a.batch, 777).fixed_eval(a.batches)
    r = T.evaluate(m, ev)
    out = {"ckpt": a.ckpt, "valid": a.valid, **{k: v for k, v in r.items() if isinstance(v, (int, float))}}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
