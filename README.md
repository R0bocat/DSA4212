# GPU / PyTorch setup (AMD Radeon RX 6900 XT, Windows)

Official PyTorch ROCm Windows wheels did not detect this GPU (gfx1030 / RDNA2); AMD's nightly gfx103X wheels do, via the
community project https://github.com/ssubedir/RCOm-windows-gfx1030.

## Versions used (from runs/*/summary.json)
- PyTorch `2.10.0+rocm7.12.0a20260204`, HIP `7.2.0`, Python `3.12.16`
- AMD driver 26.10.1, HIP SDK TODO, Windows 11

## Setup that was used
```
git clone https://github.com/ssubedir/RCOm-windows-gfx1030
cd RCOm-windows-gfx1030
.\scripts\setup-nightly.bat
.\activate.bat
```

## Evidence that it works and measurements that fixed the budget (save the real outputs here)
- `check_gpu_output.txt` : `python code/check_gpu.py`
- `bench_fp16.txt`, `bench_fp32.txt`, `bench_bf16.txt` : `python code/bench_tokens_per_sec.py --dtype fp16` (and fp32, bf16).
  bf16 was 7-11x slower than fp32 on this GPU, fp16 fastest (about 131k tokens/s for the 13.9M model), which is why fp16 is used.
- `pip_freeze.txt` : `pip freeze`
