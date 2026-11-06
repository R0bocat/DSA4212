"""
score_similarity.py - rank every document of the general corpus by how PubMed-like it is.

Score = NLL under the general-only model  -  NLL under the PubMed-only model   (nats/token, "sim")
Positive/large = the PubMed model explains the text much better than the general model = domain-like text.
This is the cross-entropy-difference criterion of Moore & Lewis (2010) for selecting in-domain data.

Efficiency: reads the token files directly (no text, no re-tokenising), scores only the first `--window` tokens of each
document, and runs BOTH models on each batch. About 190M tokens per model for two FineWeb shards, i.e. roughly 15-25 minutes
on your GPU. PAUSE THE RUN QUEUE while this runs: sharing the GPU would slow training runs and can trip their 45-minute limit.

  python score_similarity.py --train-bins data/fineweb_v1/train.bin data/fineweb_v2/train.bin \\
      --pm-run runs/S_Uall_d1.0_s1337 --gen-run runs/S_Uall_d0.0_s1337 \\
      --ref-bins pubmed=data/pubmed_v2/val.bin fineweb=data/fineweb_v1/val.bin \\
      --tokenizer tokenizers/mix8192.json --out data/similarity/scores.npz
  add  --max-docs 20000  for a 1-minute dry run.
"""
import argparse
import json
import os
import time

import numpy as np

from similarity_utils import doc_table, score_docs


def load_model(run_dir, device):
    """Rebuild a trained model from runs/<name>/summary.json + best.pt (same class that trained it)."""
    import torch
    from train import GPT, Config

    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    ckpt = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} not found (checkpoints are not in the repository: re-run that training run)")
    cfg = Config(**summary["config"])
    model = GPT(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval().to(device), cfg, summary


def _hidden(model, x):
    """Final hidden states (after the last LayerNorm) for token ids x of shape (B, T): the model's own forward pass."""
    import torch
    pos = torch.arange(x.size(1), device=x.device)
    h = model.wte(x) + model.wpe(pos)
    for blk in model.blocks:
        h = blk(h)
    return model.ln_f(h)


def make_nll_fn(model, device):
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def fn(x, y):
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.head(_hidden(model, xt))
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)), yt.view(-1), reduction="none")
        return nll.view(yt.shape).cpu().numpy()

    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-bins", nargs="+", required=True, help="train.bin files of the general corpus")
    ap.add_argument("--pm-run", required=True, help="run folder of the PubMed-only model (best.pt + summary.json)")
    ap.add_argument("--gen-run", required=True, help="run folder of the general-only model")
    ap.add_argument("--ref-bins", nargs="*", default=[], help="label=path.bin reference sets scored the same way")
    ap.add_argument("--tokenizer", default=None, help="only used to print example documents")
    ap.add_argument("--out", default="data/similarity/scores.npz")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--max-docs", type=int, default=0, help="score only a random subset per file (dry run)")
    ap.add_argument("--show", type=int, default=4, help="example documents to print from each end of the ranking")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm, cfg_pm, _ = load_model(a.pm_run, device)
    gm, cfg_gm, _ = load_model(a.gen_run, device)
    assert cfg_pm.block_size >= a.window and cfg_gm.block_size >= a.window, "window longer than the models' context"
    fns = [make_nll_fn(gm, device), make_nll_fn(pm, device)]  # order: general, pubmed
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def run_file(path, limit=0):
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        starts, lengths = doc_table(tokens)
        t0 = time.time()
        if limit:  # dry run: score a random subset by restricting the table
            keep = np.sort(rng.choice(len(starts), size=min(limit, len(starts)), replace=False))
            sub_s, sub_l = starts[keep], lengths[keep]
        else:
            sub_s, sub_l = starts, lengths

        def log(done, total):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {os.path.basename(os.path.dirname(path))}: {done:,}/{total:,} docs  ({rate:,.0f} docs/s, ETA {(total - done) / max(rate, 1e-9) / 60:.1f} min)", flush=True)

        idx, nll = score_docs(tokens, sub_s, sub_l, fns, a.window, a.batch, a.min_len, log)
        return tokens, sub_s[idx], sub_l[idx], nll[:, 0], nll[:, 1], len(starts)

    print(f"window {a.window} tokens | models: general={a.gen_run}  pubmed={a.pm_run}")
    bin_idx, start, length, nll_gen, nll_pm = [], [], [], [], []
    mems = []
    for b, path in enumerate(a.train_bins):
        print(f"scoring {path}")
        tokens, s, l, ng, npm, n_all = run_file(path, a.max_docs)
        mems.append(tokens)
        bin_idx.append(np.full(len(s), b, dtype=np.int8))
        start.append(s); length.append(l); nll_gen.append(ng); nll_pm.append(npm)
        print(f"  {len(s):,} of {n_all:,} documents eligible (>= {a.min_len} tokens)")
    bin_idx, start, length = np.concatenate(bin_idx), np.concatenate(start), np.concatenate(length)
    nll_gen, nll_pm = np.concatenate(nll_gen), np.concatenate(nll_pm)
    sim = nll_gen - nll_pm

    refs = {}
    for item in a.ref_bins:
        label, path = item.split("=", 1)
        _, _, _, ng, npm, _ = run_file(path, a.max_docs)
        refs[f"ref_{label}"] = ng - npm
        print(f"reference '{label}': mean sim {np.mean(ng - npm):+.3f} over {len(ng):,} docs")

    np.savez(a.out, bin_paths=np.array(a.train_bins), bin_idx=bin_idx, start=start, length=length,
             nll_gen=nll_gen, nll_pm=nll_pm, sim=sim, window=a.window, **refs)
    q = np.quantile(sim, [0.01, 0.25, 0.5, 0.75, 0.99])
    print(f"\nsaved {a.out}: {len(sim):,} docs. sim quantiles 1/25/50/75/99%: " + "  ".join(f"{v:+.2f}" for v in q))

    if a.tokenizer and a.show:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        order = np.argsort(-sim)
        for title, sel in (("MOST PubMed-like", order[:a.show]), ("LEAST PubMed-like", order[-a.show:])):
            print(f"\n--- {title} FineWeb-Edu documents ---")
            for j in sel:
                txt = tok.decode(mems[bin_idx[j]][start[j]:start[j] + 60].tolist())
                print(f"[sim {sim[j]:+.2f}] {txt[:230].replace(chr(10), ' ')}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
