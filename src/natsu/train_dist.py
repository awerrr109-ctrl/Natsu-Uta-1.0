"""
Distributed / large-scale training entry point (FSDP2-style sharding via torch.distributed.fsdp.fully_shard when
available, else DDP). Designed for the 50M→9B ladder on GPUs; verified HERE only on CPU with the gloo backend
(world_size 1 and 2) for correctness of: init, sharded forward/backward, grad accumulation, checkpoint save/resume,
deterministic data sharding. GPU throughput/stability is UNVERIFIED (no GPU in this sandbox).

Launch (GPU):  torchrun --nproc_per_node=8 -m natsu.train_dist --config configs/ladder/C4_1B.json --data data/tokens.bin \
               --steps 20000 --batch 16 --seq 4096 --lr 3e-4 --out runs/c4_1b --bf16
Launch (CPU check): torchrun --nproc_per_node=2 -m natsu.train_dist --config ... --cpu --steps 3
Checkpoint format: <out>/step_<N>/{model.pt (full state dict, rank0), optim_rank<r>.pt, meta.json}
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.model import Natsu, NatsuConfig, Block
from natsu.optim import build_optim, wsd


def setup(cpu):
    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("gloo" if cpu else "nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cpu") if cpu else torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if not cpu:
        torch.cuda.set_device(dev)
    return r, w, dev


class ShardedTokens:
    """Each rank reads disjoint deterministic windows from a uint16/uint32 memmap (no RAM copy of the corpus)."""
    def __init__(self, path, seq, batch, rank, world, seed=0, dtype=np.uint16):
        self.d = np.memmap(path, dtype=dtype, mode="r")
        self.seq, self.batch, self.rank, self.world = seq, batch, rank, world
        self.rng = np.random.default_rng(seed * 1000 + rank)

    def get(self):
        ix = self.rng.integers(0, len(self.d) - self.seq - 1, self.batch)
        x = torch.from_numpy(np.stack([self.d[i:i + self.seq + 1].astype(np.int64) for i in ix]))
        return x[:, :-1], x[:, 1:]


def wrap(model, world, dev, cpu):
    try:
        from torch.distributed.fsdp import fully_shard   # PyTorch >= 2.6 (FSDP2)
        if world > 1 and not cpu:
            for m in model.modules():
                if isinstance(m, Block):
                    fully_shard(m)
            fully_shard(model)
            return model, "fsdp2"
    except ImportError:
        pass
    if world > 1:
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=None if cpu else [dev]), "ddp"
    return model, "single"


def save(out, step, model, opts, rank, meta):
    d = os.path.join(out, f"step_{step}")
    os.makedirs(d, exist_ok=True)
    sd = (model.module if hasattr(model, "module") else model).state_dict()
    if rank == 0:
        torch.save({"config": meta["model"], "state_dict": sd}, os.path.join(d, "model.pt"))
        json.dump(meta | {"step": step}, open(os.path.join(d, "meta.json"), "w"))
    torch.save([o.state_dict() for o in opts], os.path.join(d, f"optim_rank{rank}.pt"))
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--optim", default="adamw")
    ap.add_argument("--out", default="runs/tmp"); ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--resume", default=""); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--override", default="{}", help="json dict of model overrides (e.g. small test sizes)")
    a = ap.parse_args()
    rank, world, dev = setup(a.cpu)
    torch.manual_seed(0)
    mcfg = {**json.load(open(a.config))["model"], **json.loads(a.override)}
    model = Natsu(NatsuConfig(**mcfg)).to(dev)
    start = 0
    if a.resume:
        ck = torch.load(os.path.join(a.resume, "model.pt"), map_location=dev)
        model.load_state_dict(ck["state_dict"]); start = json.load(open(os.path.join(a.resume, "meta.json")))["step"]
    model, mode = wrap(model, world, dev, a.cpu)
    core = model.module if hasattr(model, "module") else model
    opts = build_optim(core, a.optim, a.lr)
    if a.resume:
        for o, s in zip(opts, torch.load(os.path.join(a.resume, f"optim_rank{rank}.pt"), map_location=dev)):
            o.load_state_dict(s)
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    data = ShardedTokens(a.data, a.seq, a.batch, rank, world, seed=start)
    meta = {"model": mcfg, "world": world, "mode": mode, "args": vars(a)}
    if rank == 0:
        print(json.dumps({"params": core.param_count(), "mode": mode, "world": world}), flush=True)
    t0, toks = time.time(), 0
    for step in range(start, a.steps):
        f = wsd(step, a.steps, max(1, a.steps // 50))
        for o, b in zip(opts, base):
            for g, lr in zip(o.param_groups, b):
                g["lr"] = lr * f
        for _ in range(a.accum):
            x, y = data.get(); x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16 and not a.cpu):
                lg = model(x)["logits"]
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1))
            (loss / a.accum).backward()
            toks += x.numel() * world
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        for m in core.modules():
            if hasattr(m, "update_balance"):
                m.update_balance()
        lt = loss.detach().clone(); dist.all_reduce(lt); lt /= world
        if rank == 0 and (step % 10 == 0 or step == a.steps - 1):
            print(json.dumps({"step": step, "loss": lt.item(), "tok_s": toks / (time.time() - t0), "lr_f": f}), flush=True)
        if (step + 1) % a.save_every == 0 or step == a.steps - 1:
            save(a.out, step + 1, model, opts, rank, meta)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
