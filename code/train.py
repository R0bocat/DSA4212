"""
train.py - small GPT training under a fixed token budget, with domain/general data mixing.

Needs train_utils.py next to it.  Data folders come from prepare_data.py (train.bin / val.bin / test.bin / meta.json).

Pilot examples (PubMed only):
  python train.py --domain-dir data/pubmed_v2 --size small --tokens 50000000 --name pilot_abundant
  python train.py --domain-dir data/pubmed_v2 --domain-unique-tokens 10000000 --tokens 100000000 --name pilot_U10M

Mixture example (later, once the general corpus is prepared with the SAME tokenizer):
  python train.py --domain-dir data/pubmed_mix --general-dir data/fineweb_mix --domain-frac 0.3 \
                  --domain-unique-tokens 40000000 --tokens 300000000 --name S_U40M_d0.3_s1

What is logged (runs/<name>/log.jsonl, summary.json):
  validation loss and bits-per-byte on the domain (and general) held-out sets, loss on a fixed sample of the
  domain TRAIN data (so the train/validation gap is visible), learning rate, tokens seen, elapsed time.
  Test metrics are computed once at the end, for the best-validation checkpoint AND for the final weights.
"""
import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils import TokenSource, bpb, get_batch, load_meta, lr_at


# ----------------------------------------------------------------- model
@dataclass
class Config:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int = 8192
    block_size: int = 256
    dropout: float = 0.0


PRESETS = {
    "tiny":   dict(n_layer=4,  n_head=4,  n_embd=128),
    "small":  dict(n_layer=6,  n_head=6,  n_embd=384),
    "medium": dict(n_layer=8,  n_head=8,  n_embd=512),
}


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.n_head = c.n_head
        self.p = c.dropout
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style scaled init for residual projections
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, src, a, device, use_amp, max_blocks=0):
    """Mean cross-entropy (nats/token) over deterministic windows of `src`."""
    model.eval()
    total, count = 0.0, 0
    for x, y in src.eval_batches(a.eval_batch_size, max_blocks):
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model(x, y)
        total += loss.item() * x.numel()
        count += x.numel()
    model.train()
    return total / max(count, 1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--general-dir", default=None)
    ap.add_argument("--domain-frac", type=float, default=1.0, help="share of training sequences from the domain corpus")
    ap.add_argument("--domain-unique-tokens", type=int, default=0, help="U: use only the first U domain train tokens (0 = all)")
    ap.add_argument("--general-unique-tokens", type=int, default=0)
    ap.add_argument("--size", choices=list(PRESETS), default="small")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=300_000_000, help="TOTAL training tokens = the compute budget")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="bf16 is very slow on RDNA2")
    ap.add_argument("--eval-interval", type=int, default=1000, help="in optimizer steps")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--eval-blocks", type=int, default=0, help="cap on val blocks per evaluation (0 = all)")
    ap.add_argument("--train-eval-blocks", type=int, default=256, help="blocks of domain TRAIN data used for the gap")
    ap.add_argument("--time-limit-min", type=float, default=0, help="safety cutoff in minutes (0 = off)")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evals without a new best (0 = off)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if a.domain_frac < 1.0 and not a.general_dir:
        raise SystemExit("--domain-frac < 1 needs --general-dir")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = a.dtype == "fp16" and device.type == "cuda"
    name = a.name or f"{a.size}_d{a.domain_frac}_U{a.domain_unique_tokens}_s{a.seed}"
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ---- data
    dmeta = load_meta(a.domain_dir)
    gmeta = load_meta(a.general_dir) if a.general_dir else None
    if gmeta and gmeta["vocab_size"] != dmeta["vocab_size"]:
        raise SystemExit("domain and general data were tokenized with different vocab sizes")
    dom_train = TokenSource(os.path.join(a.domain_dir, "train.bin"), a.block_size, a.domain_unique_tokens)
    dom_val = TokenSource(os.path.join(a.domain_dir, "val.bin"), a.block_size)
    dom_test = TokenSource(os.path.join(a.domain_dir, "test.bin"), a.block_size)
    gen_train = gen_val = gen_test = None
    if a.general_dir:
        gen_train = TokenSource(os.path.join(a.general_dir, "train.bin"), a.block_size, a.general_unique_tokens)
        gen_val = TokenSource(os.path.join(a.general_dir, "val.bin"), a.block_size)
        gen_test = TokenSource(os.path.join(a.general_dir, "test.bin"), a.block_size)

    # ---- model / optimiser
    cfg = Config(vocab_size=dmeta["vocab_size"], block_size=a.block_size, dropout=a.dropout, **PRESETS[a.size])
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, a.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tokens_per_step = a.batch_size * a.block_size
    max_steps = a.tokens // tokens_per_step
    warmup_steps = max(10, int(a.warmup_frac * max_steps))
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"run {name} | {dev_name} | torch {torch.__version__} (hip={getattr(torch.version, 'hip', None)})")
    print(f"model {a.size}: {n_params / 1e6:.1f}M params | steps {max_steps:,} x {tokens_per_step:,} tokens "
          f"= {max_steps * tokens_per_step / 1e6:,.0f}M tokens | dtype {a.dtype}")
    print(f"domain train tokens available: {dom_train.n / 1e6:,.1f}M | domain frac {a.domain_frac}"
          + (f" | general train tokens: {gen_train.n / 1e6:,.1f}M" if gen_train else ""))

    log_f = open(os.path.join(out, "log.jsonl"), "w")
    best = {"val": float("inf"), "step": -1, "state": None}
    evals_since_best = 0
    run_loss = torch.zeros((), device=device)
    run_steps = 0
    dom_seqs = 0
    eval_time = 0.0
    diverged = truncated = stopped_early = False
    t0 = time.time()
    model.train()

    def do_eval(step):
        nonlocal eval_time, evals_since_best, run_loss, run_steps
        te = time.time()
        tokens_seen = step * tokens_per_step
        rec = {"step": step, "tokens": tokens_seen, "lr": opt.param_groups[0]["lr"],
               "train_loss_running": (run_loss / max(run_steps, 1)).item()}
        rec["val_domain_loss"] = evaluate(model, dom_val, a, device, use_amp, a.eval_blocks)
        rec["val_domain_bpb"] = bpb(rec["val_domain_loss"], dmeta["val"]["tokens_per_byte"])
        rec["train_domain_loss"] = evaluate(model, dom_train, a, device, use_amp, a.train_eval_blocks)
        rec["gap"] = rec["val_domain_loss"] - rec["train_domain_loss"]
        if gen_val is not None:
            rec["val_general_loss"] = evaluate(model, gen_val, a, device, use_amp, a.eval_blocks)
            rec["val_general_bpb"] = bpb(rec["val_general_loss"], gmeta["val"]["tokens_per_byte"])
        rec["domain_epochs"] = dom_seqs * a.block_size / dom_train.n
        run_loss = torch.zeros((), device=device)
        run_steps = 0
        if rec["val_domain_loss"] < best["val"]:
            best.update(val=rec["val_domain_loss"], step=step,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            evals_since_best = 0
            rec["new_best"] = True
        else:
            evals_since_best += 1
        eval_time += time.time() - te
        rec["elapsed_min"] = (time.time() - t0) / 60
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        gen_txt = f" | gen bpb {rec['val_general_bpb']:.4f}" if "val_general_bpb" in rec else ""
        print(f"step {step:>6} | {tokens_seen / 1e6:>6.1f}M tok | lr {rec['lr']:.2e} | train {rec['train_loss_running']:.3f} "
              f"| val dom bpb {rec['val_domain_bpb']:.4f} | gap {rec['gap']:+.3f}{gen_txt} "
              f"| ep {rec['domain_epochs']:.1f} | {rec['elapsed_min']:.1f} min" + (" *" if rec.get("new_best") else ""))
        return rec

    # ---- training loop
    step = 0
    for step in range(1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, max_steps, a.lr, warmup_steps, a.min_lr_ratio)
        x, y, n_dom = get_batch(dom_train, gen_train, a.domain_frac, a.batch_size, rng)
        dom_seqs += n_dom
        x = torch.from_numpy(x).to(device, non_blocking=True)
        y = torch.from_numpy(y).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model(x, y)
        scaler.scale(loss).backward()
        if a.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.detach()
        run_steps += 1

        out_of_time = a.time_limit_min and (time.time() - t0) / 60 > a.time_limit_min
        if step % a.eval_interval == 0 or step == max_steps or out_of_time:
            rec = do_eval(step)
            if not math.isfinite(rec["val_domain_loss"]):
                diverged = True
                print("non-finite validation loss: run diverged")
                break
            if out_of_time:
                truncated = True
                print("time limit reached: stopping")
                break
            if a.patience and evals_since_best >= a.patience:
                stopped_early = True
                print(f"no improvement for {a.patience} evaluations: stopping early")
                break

    # ---- final test evaluation (final weights, then best-validation weights)
    total_min = (time.time() - t0) / 60
    summary = {"name": name, "args": vars(a), "params": n_params, "config": asdict(cfg), "device": dev_name,
               "torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "python": platform.python_version(),
               "steps_done": step, "tokens_done": step * tokens_per_step, "max_steps": max_steps,
               "diverged": diverged, "truncated_by_time": truncated, "stopped_early": stopped_early,
               "total_minutes": total_min, "eval_minutes": eval_time / 60,
               "train_tokens_per_sec": step * tokens_per_step / max(time.time() - t0 - eval_time, 1e-9),
               "domain_tokens_available": dom_train.n, "domain_epochs": dom_seqs * a.block_size / dom_train.n,
               "best_step": best["step"], "best_val_domain_loss": best["val"],
               "best_val_domain_bpb": bpb(best["val"], dmeta["val"]["tokens_per_byte"])}
    if not diverged:
        summary["final_test_domain_bpb"] = bpb(evaluate(model, dom_test, a, device, use_amp), dmeta["test"]["tokens_per_byte"])
        if gen_test is not None:
            summary["final_test_general_bpb"] = bpb(evaluate(model, gen_test, a, device, use_amp), gmeta["test"]["tokens_per_byte"])
        if best["state"] is not None:
            model.load_state_dict(best["state"])
            torch.save(best["state"], os.path.join(out, "best.pt"))
            summary["best_test_domain_bpb"] = bpb(evaluate(model, dom_test, a, device, use_amp), dmeta["test"]["tokens_per_byte"])
            if gen_test is not None:
                summary["best_test_general_bpb"] = bpb(evaluate(model, gen_test, a, device, use_amp), gmeta["test"]["tokens_per_byte"])
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== summary ===")
    for k in ["steps_done", "tokens_done", "total_minutes", "eval_minutes", "train_tokens_per_sec", "domain_epochs",
              "best_step", "best_val_domain_bpb", "best_test_domain_bpb", "final_test_domain_bpb",
              "best_test_general_bpb", "final_test_general_bpb", "diverged", "truncated_by_time"]:
        if k in summary:
            v = summary[k]
            print(f"  {k:<26}{v:,.4f}" if isinstance(v, float) else f"  {k:<26}{v}")
    print(f"  saved to {out}/")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
