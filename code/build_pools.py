"""
build_pools.py - turn the similarity scores into training pools for the experiment.

  python build_pools.py --scores data/similarity/scores.npz --eval-bins data/pubmed_v2/val.bin data/pubmed_v2/test.bin \\
         --general-dir data/fineweb_v1 --out data/fw_pools --k 4 --tokenizer tokenizers/mix8192.json

What it does (numpy only, a few minutes for ~860M tokens):
  1. DECONTAMINATION: any general-corpus document that shares >= --max-hits sampled 12-token shingles with the PubMed
     validation/test sets is dropped from every pool. The most PubMed-like documents are exactly where copies of abstracts
     would hide, and they would inflate the benefit of the "similar" pool on the test set.
  2. Ranks the remaining documents by similarity and cuts them into k pools holding EQUAL numbers of tokens
     (q1 = most PubMed-like ... qk = least), plus a random pool 'rnd' of the same size as the control.
  3. Writes <out>/<pool>/train.bin (documents shuffled) and copies val.bin / test.bin / meta.json of the general corpus,
     so every pool folder can be passed to train.py as --general-dir. Writes <out>/pools_summary.json.
"""
import argparse
import json
import os
import shutil

import numpy as np

from similarity_utils import (contaminated_hits, doc_table, equal_mass_pools, random_pool, reference_hashes,
                              write_pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--eval-bins", nargs="+", required=True, help="PubMed val.bin / test.bin to decontaminate against")
    ap.add_argument("--general-dir", required=True, help="folder with the general corpus val.bin/test.bin/meta.json")
    ap.add_argument("--out", default="data/fw_pools")
    ap.add_argument("--k", type=int, default=4, help="number of similarity levels (equal-token pools)")
    ap.add_argument("--prefix", default="q", help="pool names are <prefix>1..<prefix>k (+ <prefix>rnd); use 'd' for deciles")
    ap.add_argument("--write", nargs="*", default=None, help="only write these pools to disk (all are still summarised)")
    ap.add_argument("--max-hits", type=int, default=3, help="drop docs with at least this many sampled shingle matches")
    ap.add_argument("--shingle", type=int, default=12)
    ap.add_argument("--sample-mod", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tokenizer", default=None, help="only used to print example documents")
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    z = np.load(a.scores)
    bins = [np.memmap(str(p), dtype=np.uint16, mode="r") for p in z["bin_paths"]]
    bin_idx, start, length, sim = z["bin_idx"].astype(np.int64), z["start"], z["length"], z["sim"]
    n = len(sim)
    print(f"{n:,} scored documents, {length.sum() / 1e6:,.0f}M tokens in {len(bins)} file(s)")

    # ---- 1. decontamination against the PubMed evaluation sets
    ref = reference_hashes([np.memmap(p, dtype=np.uint16, mode="r") for p in a.eval_bins], a.shingle, a.sample_mod)
    print(f"evaluation sets: {len(ref):,} distinct sampled shingles")
    hits = np.zeros(n, dtype=np.int64)
    for b, tokens in enumerate(bins):
        all_starts, _ = doc_table(tokens)
        counts = contaminated_hits(tokens, all_starts, ref, a.shingle, a.sample_mod)
        rows = np.flatnonzero(bin_idx == b)
        hits[rows] = counts[np.searchsorted(all_starts, start[rows])]
    flagged = hits >= a.max_hits
    print(f"decontamination: {flagged.sum():,} docs ({100 * flagged.mean():.3f}%) have >= {a.max_hits} matching shingles -> excluded")

    # ---- 2. ranking and equal-token pools
    ok = np.flatnonzero(~flagged)
    order = ok[np.argsort(-sim[ok], kind="stable")]               # most similar first
    pools = {f"{a.prefix}{i + 1}": p for i, p in enumerate(equal_mass_pools(order, length, a.k))}
    mass = int(np.mean([length[p].sum() for p in pools.values()]))
    pools["rnd" if a.prefix == "q" else f"{a.prefix}rnd"] = random_pool(ok, length, mass, rng)
    # where would the flagged docs have ranked? (are copies concentrated among the most similar documents?)
    cut = np.quantile(sim[ok], np.linspace(0, 1, a.k + 1)[1:-1])   # ascending similarity boundaries
    flag_by_level = np.bincount(np.searchsorted(cut, sim[flagged]), minlength=a.k)[::-1]   # level 1 = most similar

    # ---- 3. write pools
    os.makedirs(a.out, exist_ok=True)
    summary = {"k": a.k, "window": int(z["window"]), "max_hits": a.max_hits, "n_docs_scored": int(n),
               "n_flagged": int(flagged.sum()), "flagged_by_similarity_level": [int(v) for v in flag_by_level],
               "refs": {k[4:]: {"mean_sim": float(np.mean(z[k])), "n": int(len(z[k]))} for k in z.files if k.startswith("ref_")},
               "pools": {}}
    print(f"\n{'pool':<5}{'docs':>10}{'tokens(M)':>11}{'mean sim':>10}{'median':>9}{'nll_pm':>8}{'nll_gen':>9}")
    for name, ids in pools.items():
        ntok = int(length[ids].sum() + len(ids))                      # tokens incl. the trailing EOS of every document
        if a.write is None or name in a.write:
            d = os.path.join(a.out, name)
            os.makedirs(d, exist_ok=True)
            assert write_pool(os.path.join(d, "train.bin"), bins, bin_idx, start, length, ids, rng) == ntok
            for f in ("val.bin", "test.bin", "meta.json"):
                shutil.copy(os.path.join(a.general_dir, f), os.path.join(d, f))
        w = length[ids].astype(np.float64)
        info = {"docs": int(len(ids)), "tokens": int(ntok), "mean_sim": float(np.average(sim[ids], weights=w)),
                "median_sim": float(np.median(sim[ids])), "mean_nll_pm": float(np.average(z["nll_pm"][ids], weights=w)),
                "mean_nll_gen": float(np.average(z["nll_gen"][ids], weights=w))}
        summary["pools"][name] = info
        print(f"{name:<5}{info['docs']:>10,}{ntok / 1e6:>11.1f}{info['mean_sim']:>+10.3f}{info['median_sim']:>+9.3f}"
              f"{info['mean_nll_pm']:>8.3f}{info['mean_nll_gen']:>9.3f}")
    for k_, v in summary["refs"].items():
        print(f"reference '{k_}': mean sim {v['mean_sim']:+.3f} (n={v['n']:,})")
    print(f"flagged documents by similarity level (1 = most similar): {summary['flagged_by_similarity_level']}")
    summ_name = "pools_summary.json" if a.prefix == "q" else f"pools_summary_{a.prefix}.json"
    with open(os.path.join(a.out, summ_name), "w") as f:
        json.dump(summary, f, indent=1)

    if a.tokenizer and a.show:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        for name in (f"{a.prefix}1", f"{a.prefix}{a.k}"):
            print(f"\n--- examples from {name} ---")
            for j in rng.choice(pools[name], size=min(a.show, len(pools[name])), replace=False):
                print(f"[sim {sim[j]:+.2f}] " + tok.decode(bins[bin_idx[j]][start[j]:start[j] + 60].tolist())[:230].replace("\n", " "))
    print(f"\nwrote pools to {a.out}/  (use e.g. --general-dir {a.out}/q1 in train.py)")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
