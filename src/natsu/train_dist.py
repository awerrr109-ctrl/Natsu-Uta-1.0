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
    """Epoch-exact sharded sampler over a uint16/uint32 token memmap (no RAM copy of the corpus).

    The corpus is cut into non-overlapping windows of seq+1 tokens. Each epoch, ONE permutation (seeded by seed+epoch, identical on
    every rank) is drawn, and rank r takes positions r, r+world, ... of it. So within an epoch every window is read exactly once
    across all ranks, and ranks never overlap. (The previous version sampled random offsets with replacement per rank: ~63% unique
    coverage per epoch, the same failure class as sandyresearch/parcae issue #10. Fixed after reading it; see F009.)
    State (epoch, cursor) is saved/restored for exact resume."""
    def __init__(self, path, seq, batch, rank, world, seed=0, dtype=np.uint16):
        self.d = np.memmap(path, dtype=dtype, mode="r")
        self.seq, self.batch, self.rank, self.world, self.seed = seq, batch, rank, world, seed
        self.n_win = (len(self.d) - 1) // (seq + 1)
        self.epoch, self.cursor = 0, 0
        self._perm()

    def _perm(self):
        p = np.random.default_rng(self.seed + 1_000_003 * self.epoch).permutation(self.n_win)
        self.mine = p[self.rank::self.world]
        usable = (len(self.mine) // self.batch) * self.batch     # drop the ragged tail so all ranks take the same number of steps
        self.mine = self.mine[:usable]

    def get(self):
        if self.cursor + self.batch > len(self.mine):
            self.epoch += 1; self.cursor = 0; self._perm()
        ix = self.mine[self.cursor:self.cursor + self.batch] * (self.seq + 1)
        self.cursor += self.batch
        x = torch.from_numpy(np.stack([self.d[i:i + self.seq + 1].astype(np.int64) for i in ix]))
        return x[:, :-1], x[:, 1:]

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state(self, st):
        self.epoch, self.cursor = st["epoch"], st["cursor"]; self._perm()


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
    ap.add_argument("--engram_delay", default="", help="'start,end' fractions: linear ramp of the Engram branch (N5/H14.1)")
    ap.add_argument("--mtp_w", type=float, default=0.3, help="weight of the MTP auxiliary loss when the model has MTP heads")
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
    data = ShardedTokens(a.data, a.seq, a.batch, rank, world, seed=0)
    if a.resume:   # exact data position (not a re-seed: the old seed=start re-drew the order and repeated/skipped data)
        dp = os.path.join(a.resume, "data_state.json")
        if os.path.exists(dp):
            data.load_state(json.load(open(dp))[str(rank)])
        else:   # legacy checkpoint: fast-forward by the number of batches already consumed
            for _ in range(start * a.accum):
                data.cursor += a.batch
                if data.cursor + a.batch > len(data.mine):
                    data.epoch += 1; data.cursor = 0; data._perm()
    meta = {"model": mcfg, "world": world, "mode": mode, "args": vars(a)}
    if rank == 0:
        print(json.dumps({"params": core.param_count(), "mode": mode, "world": world}), flush=True)
    t0, toks = time.time(), 0
    ed = [float(v) for v in a.engram_delay.split(",")] if a.engram_delay else None
    for step in range(start, a.steps):
        f = wsd(step, a.steps, max(1, a.steps // 50))
        if ed is not None and getattr(core, "engram", None) is not None:
            fr = step / max(1, a.steps)
            core.engram_scale = float(min(1.0, max(0.0, (fr - ed[0]) / max(1e-9, ed[1] - ed[0]))))
        for o, b in zip(opts, base):
            for g, lr in zip(o.param_groups, b):
                g["lr"] = lr * f
        for _ in range(a.accum):
            x, y = data.get(); x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16 and not a.cpu):
                out = model(x)
            lg = out["logits"]
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1))
            total = loss
            for j, ml in enumerate(out.get("mtp_logits", []) or [], start=1):   # MTP: predict y shifted by j more tokens
                if y.shape[1] > j:
                    total = total + a.mtp_w * F.cross_entropy(ml[:, :-j].float().reshape(-1, ml.shape[-1]), y[:, j:].reshape(-1))
            (total / a.accum).backward()
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
            st = [None] * world
            dist.all_gather_object(st, data.state())
            if rank == 0:
                json.dump({str(r): v for r, v in enumerate(st)}, open(os.path.join(a.out, f"step_{step + 1}", "data_state.json"), "w"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
