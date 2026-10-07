"""
Memory-aware trainer. Usage:
    python -m natsu.train --config experiments/configs/X.json
Config JSON: {"name":..., "model":{NatsuConfig fields}, "train":{...}, "data":{...}}

Train-time features specific to this project:
  * loop_sampling: "fixed" | "uniform" | "poisson"  -> random loop count per step
    (Huginn-style, arXiv:2502.05171) so ANY loop count is a valid model at inference.
  * self_distill: weight of KL(p_r || stopgrad p_R) between a sampled shallow loop count r
    and full depth R (our addition: makes shallow loops good *drafts* of deep loops
    -> self-speculative decoding with zero extra parameters).
  * gate_penalty: compute penalty on the soft depth gate (learned budget).
  * mtp: multi-token-prediction auxiliary loss (weight mtp_w).
Logging: JSONL with loss, lr, tokens, tok/s, RSS MB (principle #18).
"""
import argparse, json, math, os, resource, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natsu.model import Natsu, NatsuConfig
from natsu.optim import build_optim, wsd, cosine
from natsu import data as D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def masked_ce(logits, y, mask):
    l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), reduction="none",
                        ignore_index=D.PAD).view_as(y)
    valid = (y != D.PAD)
    if mask is not None:
        valid = valid & mask
    return (l * valid).sum() / valid.sum().clamp(min=1), l, valid


def make_loader(dc, seq, batch, seed, split="train"):
    if dc["kind"] == "memmap":
        return D.MemmapLoader(os.path.join(ROOT, dc[split]), seq, batch, seed)
    if dc["kind"] == "synthetic":
        kw = dict(dc.get("kw", {}))
        return D.SyntheticLoader(dc["task"], seq, batch, seed, **kw)
    if dc["kind"] == "region":   # G1c: contiguous train region (+ optional cached kNN targets); eval stays on the normal valid set
        if split != "train":
            return D.MemmapLoader(os.path.join(ROOT, dc["valid"]), seq, batch, seed)
        return D.RegionLoader(os.path.join(ROOT, dc["train"]), dc["start"], dc["tokens"], seq, batch, seed,
                              os.path.join(ROOT, dc["knn_targets"]) if dc.get("knn_targets") else None)
    if dc["kind"] == "mix":
        ls = [make_loader(x, seq, batch, seed + i, split) for i, x in enumerate(dc["parts"])]
        return D.MixLoader(ls, dc["weights"], seed)
    raise ValueError(dc)


BYTES_PER_TOKEN = [1.0]   # set from data config; bpb = loss/ln2/bytes_per_token (tokenizer-independent)


@torch.no_grad()
def evaluate(model, batches, n_loops=None, gate_threshold=None):
    model.eval()
    tot, n, correct_seq, n_seq = 0.0, 0, 0, 0
    skip = []
    for x, y, m in batches:
        out = model(x, n_loops=n_loops, gate_threshold=gate_threshold)
        loss, l, valid = masked_ce(out["logits"], y, m)
        tot += (l * valid).sum().item(); n += valid.sum().item()
        if m is not None:  # exact-match accuracy over answer spans
            pred = out["logits"].argmax(-1)
            ok = ((pred == y) | ~valid).all(-1)
            correct_seq += ok.sum().item(); n_seq += len(ok)
        if "skip_frac" in out:
            skip.append(out["skip_frac"].item())
    model.train()
    r = {"loss": tot / max(n, 1)}
    r["ppl"] = math.exp(min(r["loss"], 20))
    r["bpb"] = r["loss"] / math.log(2) / BYTES_PER_TOKEN[0]
    if n_seq:
        r["acc"] = correct_seq / n_seq
    if skip:
        r["skip_frac"] = sum(skip) / len(skip)
    return r


def block_cos_reg(model, k=2):
    """R33 regulariser: mean (1 - cos) between matching >=2-D tensors of core blocks i and i+k."""
    core = list(model.core)
    terms = []
    for i in range(len(core) - k):
        pa = dict(core[i].named_parameters()); pb = dict(core[i + k].named_parameters())
        for n, a in pa.items():
            b = pb.get(n)
            if b is None or a.dim() < 2 or a.shape != b.shape:
                continue
            terms.append(1 - F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return torch.stack(terms).mean() if terms else torch.zeros(())


def sample_loops(R, mode, step_rng, frac=0.0):
    """frac = training progress in [0,1]. Modes:
       fixed | uniform | poisson (Huginn, mean R/2+.5) | poisson_R (Parcae-like, mean R, clip [1, 2R]) |
       fixed_then_uniform (H13.1 curriculum: fixed R for 80% of steps, then U{1..R} to buy the anytime property)"""
    if mode == "fixed_then_uniform":
        mode = "fixed" if frac < 0.8 else "uniform"
    if mode == "poisson_R":
        return int(min(2 * R, max(1, step_rng.poisson(R))))
    if R == 1 or mode == "fixed":
        return R
    if mode == "uniform":
        return int(step_rng.integers(1, R + 1))
    if mode == "poisson":  # Huginn: heavy-ish tail around mean R/2+1, clipped
        return int(min(R, max(1, step_rng.poisson(R / 2 + 0.5))))
    raise ValueError(mode)


def train(cfg):
    import numpy as np
    tc, dc = cfg["train"], cfg["data"]
    torch.manual_seed(tc.get("seed", 0))
    torch.set_num_threads(tc.get("threads", os.cpu_count()))
    BYTES_PER_TOKEN[0] = dc.get("bytes_per_token", 1.0)
    mc = NatsuConfig(**cfg["model"])
    model = Natsu(mc)
    if tc.get("init_from"):   # continue from a checkpoint (e.g. memory-editing probe)
        ck0 = torch.load(os.path.join(ROOT, tc["init_from"]), map_location="cpu")
        model.load_state_dict(ck0["state_dict"])
    tr_only = tc.get("train_only")   # list of name substrings; everything else frozen (e.g. ["engram.table"] = memory-only edit)
    if tr_only:
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(any(k in n_ for k in tr_only))
    if mc.engram_vip and dc.get("vip"):
        v = json.load(open(os.path.join(ROOT, dc["vip"])))
        model.engram.load_vip({int(k): x for k, x in v.items()})
    seq, batch = tc["seq"], tc["batch"]
    opts = build_optim(model, tc.get("optim", "adamw"), tc["lr"], tc.get("wd", 0.1))
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    if tc.get("optim") == "muon":  # AdamW part (embeddings/norms) uses adam_lr
        for g in opts[1].param_groups:
            g["lr"] = tc.get("adam_lr", 3e-3)
        base_lrs[1] = [tc.get("adam_lr", 3e-3)]
    if tc.get("engram_lr_mult") and model.engram is not None:   # paper: Engram table lr x5, no wd (split it into its own Adam group)
        tid = {id(p) for n, p in model.named_parameters() if n.startswith("engram.") and ("table" in n or "basis" in n)}
        g0 = opts[1].param_groups[0]
        keep = [p for p in g0["params"] if id(p) not in tid]
        tab = [p for p in g0["params"] if id(p) in tid]
        g0["params"] = keep
        opts[1].add_param_group({"params": tab, "lr": g0["lr"] * tc["engram_lr_mult"], "weight_decay": 0.0})
        base_lrs[1] = [g0["lr"], g0["lr"] * tc["engram_lr_mult"]]
    teacher, kd = None, tc.get("distill")
    if kd:   # distilled pretraining (DPT, arXiv:2509.01649): KL to teacher on all but the lowest-entropy tokens
        ck = torch.load(os.path.join(ROOT, kd["teacher"]), map_location="cpu")
        teacher = Natsu(NatsuConfig(**ck["config"])); teacher.load_state_dict(ck["state_dict"]); teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
    tr = make_loader(dc, seq, batch, tc.get("seed", 0), "train")
    ev = make_loader(dc, seq, tc.get("eval_batch", batch), 777, "valid" if dc["kind"] in ("memmap", "region") else "train")
    ev_batches = ev.fixed_eval(tc.get("eval_batches", 8))
    extra_eval = {}
    for name, over in cfg.get("extra_eval", {}).items():  # e.g. OOD hop counts
        extra_eval[name] = ev.fixed_eval(tc.get("eval_batches", 8), seed=4242, **over)
    out_dir = os.path.join(ROOT, "experiments", "results", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "log.jsonl"), "w")
    info = {"name": cfg["name"], "params": model.param_count(), "params_nonembed": model.param_count(True),
            "flops_per_token_fwd": model.flops_per_token(seq=seq), "config": cfg}
    print(json.dumps({k: v for k, v in info.items() if k != "config"}), flush=True)
    steps, warm = tc["steps"], tc.get("warmup", 50)
    sched = wsd if tc.get("sched", "wsd") == "wsd" else cosine
    rng = np.random.default_rng(tc.get("seed", 0))
    R = mc.n_loops
    tok_seen, ans_seen, t0, flops = 0, 0, time.time(), 0.0
    accum = tc.get("accum", 1)
    steps_to = {}
    ed = tc.get("engram_delay", None)   # H14.1: [start_frac, end_frac] linear ramp of the Engram branch 0->1
    for step in range(steps):
        if ed is not None and getattr(model, "engram", None) is not None:
            fr = step / max(1, steps)
            model.engram_scale = float(min(1.0, max(0.0, (fr - ed[0]) / max(1e-9, ed[1] - ed[0]))))
        f = sched(step, steps, warm)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * f
        for _ in range(accum):
            x, y, m = tr.get()
            r = sample_loops(R, tc.get("loop_sampling", "fixed"), rng, step / max(1, steps))
            la = mc.depth_gate and mc.gate_mode == "lookahead" and r > 1
            out = model(x, n_loops=r, return_inter=la)
            loss, _, valid = masked_ce(out["logits"], y, m)
            total = loss
            if mc.mtp and "mtp_logits" in out:
                for j, lg in enumerate(out["mtp_logits"], start=1):
                    yy = torch.cat([y[:, j:], torch.full_like(y[:, :j], D.PAD)], 1)
                    mm = None if m is None else torch.cat([m[:, j:], torch.zeros_like(m[:, :j])], 1)
                    total = total + tc.get("mtp_w", 0.3) * masked_ce(lg, yy, mm)[0]
            if mc.depth_gate and mc.gate_mode == "soft" and "gate_mean" in out:
                total = total + tc.get("gate_penalty", 0.0) * out["gate_mean"]
            if la and "inter_logits" in out:
                # lookahead depth supervision: label_r = 1[CE(after loops>=r) < CE(stop before r) - margin]
                with torch.no_grad():
                    ces = [masked_ce(lg, y, m)[1] for lg in out["inter_logits"]] + [masked_ce(out["logits"], y, m)[1]]
                gl = 0.0
                for j, g in enumerate(out["gate_logits"]):
                    lab = (ces[j + 1] < ces[j] - tc.get("gate_margin", 0.02)).float()
                    gl = gl + (F.binary_cross_entropy_with_logits(g.float(), lab, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
                    # deep supervision of the intermediate exits keeps early stops usable (anytime) — cheap: coda only
                total = total + tc.get("gate_w", 0.1) * gl
                if tc.get("exit_w", 0.0) > 0:
                    for lg in out["inter_logits"]:
                        total = total + tc["exit_w"] * masked_ce(lg, y, m)[0] / len(out["inter_logits"])
            kw_ = tc.get("knn_w", 0.0)
            if kw_ > 0 and getattr(tr, "aux", None) is not None:
                # G1c retrieval distillation: soft CE to cached p_kNN (MemDec-style hybrid objective, but into the model itself)
                kv, kwt = tr.aux
                lsm = F.log_softmax(out["logits"].float(), -1)
                kl_knn = -(lsm.gather(-1, kv) * kwt).sum(-1)
                total = total + kw_ * (kl_knn * valid).sum() / valid.sum().clamp(min=1)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(x)["logits"].float() / kd.get("T", 1.0)
                    tlp = F.log_softmax(tl, -1)
                    ent = -(tlp.exp() * tlp).sum(-1)
                    keep = valid.clone()
                    q = kd.get("skip_low_entropy", 0.0)      # token routing: hard labels only on lowest-entropy tokens
                    if q > 0:
                        thr = torch.quantile(ent[valid].flatten()[:65536], q)
                        keep = keep & (ent > thr)
                slp = F.log_softmax(out["logits"].float() / kd.get("T", 1.0), -1)
                klt = F.kl_div(slp, tlp, log_target=True, reduction="none").sum(-1)
                kdl = (klt * keep).sum() / keep.sum().clamp(min=1)
                w = kd.get("w", 0.5)
                total = total + w * kdl                    # CE kept at full weight; KD added (DPT-style)
            sd = tc.get("self_distill", 0.0)
            if sd > 0 and R > 1:
                rs = int(rng.integers(1, R))
                with torch.no_grad():
                    tgt = out["logits"].detach() if r == R else model(x, n_loops=R)["logits"]
                lo = model(x, n_loops=rs)["logits"]
                kl = F.kl_div(F.log_softmax(lo.float(), -1), F.log_softmax(tgt.float(), -1),
                              log_target=True, reduction="none").sum(-1)
                total = total + sd * (kl * valid).sum() / valid.sum().clamp(min=1)
            bc = tc.get("block_cos", 0.0)
            if bc > 0:
                # H13.7 (R33): looping-inspired regulariser — pull core block i towards block i+k (cosine) to get the loop
                # inductive bias without loop FLOPs or weight sharing. Only same-shape >=2-D tensors are tied.
                total = total + bc * block_cos_reg(model, tc.get("block_cos_k", 2))
            (total / accum).backward()
            tok_seen += (x != D.PAD).sum().item(); ans_seen += valid.sum().item()
            flops += 3 * model.flops_per_token(n_loops=r, seq=seq) * x.numel()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip", 1.0)).item()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        for mod in model.modules():
            if hasattr(mod, "update_balance"):
                mod.update_balance(tc.get("moe_bias_rate", 1e-3))
        if step % tc.get("log_every", 25) == 0 or step == steps - 1:
            el = time.time() - t0
            rec = {"step": step, "loss": loss.item(), "lr_f": f, "gnorm": gn, "tokens": tok_seen, "scored": ans_seen, "scored_tokens": ans_seen,
                   "tok_s": tok_seen / el, "rss_mb": rss_mb(), "peak_rss_mb": peak_rss_mb(), "train_flops": flops,
                   "loops": r}
            if step % tc.get("eval_every", 200) == 0 or step == steps - 1:
                rec["eval"] = evaluate(model, ev_batches)
                # F012: iso-loss / steps-to-threshold bookkeeping (first eval step reaching each acc or loss threshold)
                for th in tc.get("acc_thresholds", [0.1, 0.2, 0.3, 0.4]):
                    if "acc" in rec["eval"] and rec["eval"]["acc"] >= th and f"acc>={th}" not in steps_to:
                        steps_to[f"acc>={th}"] = {"step": step, "train_flops": flops}
                for th in tc.get("loss_thresholds", []):
                    if rec["eval"]["loss"] <= th and f"loss<={th}" not in steps_to:
                        steps_to[f"loss<={th}"] = {"step": step, "train_flops": flops}
                for name, b in extra_eval.items():
                    rec[f"eval_{name}"] = evaluate(model, b)
            if not math.isfinite(rec["loss"]):
                rec["diverged"] = True
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(json.dumps(rec), flush=True)
            if rec.get("diverged"):
                break
    # final multi-depth evaluation (test-time compute scaling curve)
    final = {"info": {k: v for k, v in info.items() if k != "config"}, "train_flops": flops, "tokens": tok_seen, "scored_tokens": ans_seen,
             "wall_s": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "steps_to": steps_to}
    loops_eval = sorted(set([1, R] + list(tc.get("eval_loops", []))))
    final["by_loops"] = {}
    for rr in loops_eval:
        e = {"main": evaluate(model, ev_batches, n_loops=rr)}
        for name, b in extra_eval.items():
            e[name] = evaluate(model, b, n_loops=rr)
        e["fwd_flops_per_token"] = model.flops_per_token(n_loops=rr, seq=seq)
        final["by_loops"][rr] = e
    if mc.depth_gate:
        final["gated"] = {th: evaluate(model, ev_batches, gate_threshold=th) for th in (0.3, 0.5, 0.7)}
    json.dump(final, open(os.path.join(out_dir, "final.json"), "w"), indent=1)
    if tc.get("save", True):
        ck = os.path.join(ROOT, "checkpoints"); os.makedirs(ck, exist_ok=True)
        torch.save({"config": mc.to_dict(), "state_dict": model.state_dict(), "run": cfg["name"]},
                   os.path.join(ck, cfg["name"] + ".pt"))
    print("FINAL", json.dumps(final["by_loops"]), flush=True)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override e.g. train.steps=100 model.n_loops=2")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k == "name":
            cfg["name"] = v
            continue
        sec, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[sec][key] = v
    train(cfg)
