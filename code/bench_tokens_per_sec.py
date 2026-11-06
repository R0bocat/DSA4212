"""
bench_tokens_per_sec.py - Training throughput (tokens/sec) on your GPU.

Uses a minimal nanoGPT-style model with random token data, so no dataset or
repo is needed. Measures real training steps (forward + backward + AdamW).

Examples:
  python bench_tokens_per_sec.py
  python bench_tokens_per_sec.py --dtype fp16
  python bench_tokens_per_sec.py --sizes tiny small --batch-size 64 --block-size 512
  python bench_tokens_per_sec.py --compile

Tip: run once with --dtype bf16 and once with --dtype fp16 and keep whichever
is faster and numerically stable for your actual experiments.
"""
import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- model
@dataclass
class Config:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int = 8192
    block_size: int = 256


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

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights counted once


PRESETS = {
    "tiny":   dict(n_layer=4,  n_head=4,  n_embd=128),
    "small":  dict(n_layer=6,  n_head=6,  n_embd=384),
    "medium": dict(n_layer=8,  n_head=8,  n_embd=512),
    "large":  dict(n_layer=12, n_head=12, n_embd=768),
}


# ----------------------------------------------------------------- benchmark
def bench(name, args, dev):
    c = Config(vocab_size=args.vocab_size, block_size=args.block_size, **PRESETS[name])
    torch.manual_seed(0)
    model = GPT(c).to(dev)
    n_params = model.n_params()
    n_nonemb = n_params - c.vocab_size * c.n_embd - c.block_size * c.n_embd

    run_model = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    use_amp = args.dtype in ("fp16", "bf16")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16"))

    x = torch.randint(0, c.vocab_size, (args.batch_size, c.block_size), device=dev)
    y = torch.randint(0, c.vocab_size, (args.batch_size, c.block_size), device=dev)
    tokens_per_step = args.batch_size * c.block_size

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss = run_model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):  # warmup (also triggers compile / kernel selection)
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    tok_s = tokens_per_step * args.steps / elapsed
    # Approximate training FLOPs: 6 * params(non-embedding) * tokens, plus attention term
    flops_per_tok = 6 * n_nonemb + 12 * c.n_layer * c.n_embd * c.block_size
    tflops = flops_per_tok * tok_s / 1e12
    mem = torch.cuda.max_memory_allocated() / 1024**3

    return dict(name=name, params=n_params, tok_s=tok_s, tflops=tflops, mem=mem,
                loss=loss.item(), step_ms=1000 * elapsed / args.steps,
                tokens_30min=tok_s * 1800, tokens_60min=tok_s * 3600)


def fmt_tokens(n):
    return f"{n / 1e6:.0f}M" if n < 1e9 else f"{n / 1e9:.2f}B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"],
                    choices=list(PRESETS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compile", action="store_true", help="use torch.compile (may fail on Windows)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to PyTorch. Run check_gpu.py first.")
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"torch   : {torch.__version__} (hip={getattr(torch.version, 'hip', None)})")
    print(f"settings: dtype={args.dtype} batch={args.batch_size} block={args.block_size} "
          f"vocab={args.vocab_size} compile={args.compile}\n")

    rows = []
    for name in args.sizes:
        try:
            r = bench(name, args, dev)
            rows.append(r)
            if not math.isfinite(r["loss"]):
                print(f"  [WARN] {name}: non-finite loss with {args.dtype}; try another --dtype")
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {name}: reduce --batch-size or --block-size")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    print(f"\n{'size':<8}{'params':>9}{'tok/s':>11}{'ms/step':>9}{'TFLOPS':>8}"
          f"{'peakGiB':>9}{'tok@30m':>9}{'tok@60m':>9}")
    for r in rows:
        print(f"{r['name']:<8}{r['params'] / 1e6:>8.1f}M{r['tok_s']:>11,.0f}{r['step_ms']:>9.1f}"
              f"{r['tflops']:>8.1f}{r['mem']:>9.2f}{fmt_tokens(r['tokens_30min']):>9}"
              f"{fmt_tokens(r['tokens_60min']):>9}")
    print("\nNote: random data, so loss values are meaningless here. Only throughput matters.")


if __name__ == "__main__":
    main()
