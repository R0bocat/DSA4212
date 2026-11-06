"""
check_gpu.py - Is my GPU actually working with PyTorch?

Run:  python check_gpu.py

Checks, in order:
  1. Which PyTorch build is installed (ROCm/HIP vs CUDA vs CPU-only)
  2. Whether PyTorch can see a GPU
  3. Whether real compute on the GPU runs and matches the CPU result
  4. Whether fp32 / fp16 / bf16 matmuls work
  5. Whether a backward pass + optimizer step works
  6. Whether scaled_dot_product_attention (used by nanoGPT) works
"""
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def main():
    failures = 0

    # ------------------------------------------------------------------ 1
    header("1. PyTorch build")
    print(f"  python        : {sys.version.split()[0]}")
    print(f"  torch         : {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = torch.version.cuda
    print(f"  torch.version.hip  : {hip}")
    print(f"  torch.version.cuda : {cuda}")
    if hip:
        ok("This is a ROCm (HIP) build of PyTorch")
    elif cuda:
        print("  [WARN] This is an NVIDIA CUDA build. It will not drive an AMD GPU.")
    else:
        print("  [WARN] This is a CPU-only build. Install a ROCm build of PyTorch.")

    # ------------------------------------------------------------------ 2
    header("2. GPU visibility")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False. PyTorch cannot see a GPU.")
        print("  Likely causes: CPU-only/CUDA wheel installed, unsupported GPU for")
        print("  this ROCm build, or missing/old driver. Stopping here.")
        sys.exit(1)
    ok("torch.cuda.is_available() is True")
    n = torch.cuda.device_count()
    print(f"  device count  : {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        arch = getattr(p, "gcnArchName", "n/a")
        print(f"  [{i}] {p.name} | arch: {arch} | VRAM: {p.total_memory / 1024**3:.1f} GiB")
    dev = torch.device("cuda:0")
    print(f"  using         : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------ 3
    header("3. Compute correctness (GPU vs CPU)")
    try:
        torch.manual_seed(0)
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        ref = a @ b
        out = (a.to(dev) @ b.to(dev)).cpu()
        err = (ref - out).abs().max().item()
        print(f"  max abs diff vs CPU: {err:.2e}")
        if err < 1e-2:
            ok("fp32 matmul on GPU matches CPU")
        else:
            fail("GPU result differs from CPU by more than expected")
            failures += 1
    except Exception:
        fail("matmul raised an exception")
        traceback.print_exc()
        failures += 1

    # ------------------------------------------------------------------ 4
    header("4. Precision support (4096x4096 matmul)")
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            x = torch.randn(4096, 4096, device=dev, dtype=dt)
            y = torch.randn(4096, 4096, device=dev, dtype=dt)
            for _ in range(3):  # warmup
                _ = x @ y
            torch.cuda.synchronize()
            t0 = time.time()
            iters = 10
            for _ in range(iters):
                _ = x @ y
            torch.cuda.synchronize()
            dt_s = (time.time() - t0) / iters
            tflops = 2 * 4096**3 / dt_s / 1e12
            ok(f"{name}: {tflops:.1f} TFLOPS")
        except Exception as e:
            fail(f"{name}: {type(e).__name__}: {e}")
            failures += 1
    try:
        print(f"  torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")
    except Exception:
        pass

    # ------------------------------------------------------------------ 5
    header("5. Backward pass + optimizer step")
    try:
        model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 10)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(64, 256, device=dev)
        y = torch.randint(0, 10, (64,), device=dev)
        losses = []
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
        if losses[-1] < losses[0]:
            ok("Loss decreased: autograd and optimizer work on the GPU")
        else:
            fail("Loss did not decrease")
            failures += 1
    except Exception:
        fail("training step raised an exception")
        traceback.print_exc()
        failures += 1

    # ------------------------------------------------------------------ 6
    header("6. scaled_dot_product_attention (causal)")
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            q = torch.randn(4, 8, 256, 64, device=dev, dtype=dt, requires_grad=True)
            k = torch.randn(4, 8, 256, 64, device=dev, dtype=dt, requires_grad=True)
            v = torch.randn(4, 8, 256, 64, device=dev, dtype=dt, requires_grad=True)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o.float().sum().backward()
            torch.cuda.synchronize()
            ok(f"{name}: forward + backward")
        except Exception as e:
            fail(f"{name}: {type(e).__name__}: {e}")
            failures += 1

    # ------------------------------------------------------------------ summary
    header("Summary")
    if failures == 0:
        print("  All checks passed. Your GPU works with PyTorch.")
        print("  Next: run bench_tokens_per_sec.py")
    else:
        print(f"  {failures} check(s) failed. See messages above.")
    print(f"  Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
