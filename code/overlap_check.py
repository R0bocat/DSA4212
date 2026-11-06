"""
overlap_check.py - is the domain TEST/VAL text also present in the general TRAINING corpus?

If PubMed test abstracts appear (verbatim or nearly) inside the FineWeb-Edu training slice, the "general" model has
effectively seen the test set and the mixture results are contaminated.

It hashes every 8-word shingle, keeps a fixed 1/16 of them (the same rule on both sides, so a duplicated passage is
still caught with very high probability), and reports how many target documents share any shingle with the reference.

  python overlap_check.py --reference data/fineweb_v1/clean_train.jsonl \
         --targets data/pubmed_v2/clean_val.jsonl data/pubmed_v2/clean_test.jsonl
"""
import argparse
import json
import os
import re
import time
from array import array

import numpy as np

WORD = re.compile(r"\w+")
MULT = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
                 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB, 0xBF58476D1CE4E5B9], dtype=np.uint64)


def sampled_shingles(text, n, mod):
    words = WORD.findall(text.lower())
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    wh = np.fromiter((hash(w) & 0xFFFFFFFFFFFFFFFF for w in words), dtype=np.uint64, count=len(words))
    m = len(words) - n + 1
    with np.errstate(over="ignore"):
        s = np.zeros(m, dtype=np.uint64)
        for k in range(n):
            s += wh[k:k + m] * MULT[k % len(MULT)]
        s ^= s >> np.uint64(29)
        s *= np.uint64(0xBF58476D1CE4E5B9)
        s ^= s >> np.uint64(32)
    return s[(s % np.uint64(mod)) == 0]


def in_sorted(values, sorted_ref):
    """Membership test against an already-sorted array: O(k log M) per call.
    (np.isin re-sorts the whole reference on every call, ~15 s per document at 18M entries.)"""
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    idx = np.searchsorted(sorted_ref, values)
    return sorted_ref[np.minimum(idx, len(sorted_ref) - 1)] == values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="jsonl whose text must NOT contain the targets")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8, help="shingle length in words")
    ap.add_argument("--sample-mod", type=int, default=16, help="keep 1 in this many shingles")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--min-hits", type=int, default=3, help="a doc is 'heavily overlapping' with at least this many matches...")
    ap.add_argument("--min-frac", type=float, default=0.3, help="...and at least this fraction of its sampled shingles matching")
    ap.add_argument("--flagged-out", default=None, help="write 'file<TAB>doc id' of heavily overlapping docs here")
    a = ap.parse_args()

    t0 = time.time()
    print(f"hashing reference {a.reference} ...")
    buf = array("Q")
    with open(a.reference, encoding="utf-8") as f:
        for i, line in enumerate(f):
            buf.extend(sampled_shingles(json.loads(line)["text"], a.n, a.sample_mod).tolist())
            if i and i % 100000 == 0:
                print(f"  {i:,} docs, {len(buf) / 1e6:,.1f}M sampled shingles")
    ref = np.unique(np.frombuffer(buf, dtype=np.uint64))
    print(f"reference: {len(ref):,} distinct sampled shingles ({time.time() - t0:.0f}s)")

    for path in a.targets:
        n_docs = n_hit_docs = n_sh = n_sh_hit = 0
        examples, strong = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                sh = sampled_shingles(d["text"], a.n, a.sample_mod)
                hits = int(in_sorted(sh, ref).sum())
                n_docs += 1
                if n_docs % 1000 == 0:
                    print(f"  checked {n_docs:,} target docs ({time.time() - t0:.0f}s)")
                n_sh += len(sh)
                n_sh_hit += hits
                if hits >= a.min_hits and hits / max(len(sh), 1) >= a.min_frac:
                    strong.append((d["id"], hits, len(sh)))
                if hits:
                    n_hit_docs += 1
                    if len(examples) < a.examples:
                        examples.append((d["id"], hits, len(sh)))
        print(f"\n{path}")
        print(f"  docs sharing >=1 sampled shingle with the reference: {n_hit_docs:,} / {n_docs:,} "
              f"({100 * n_hit_docs / max(n_docs, 1):.2f}%)")
        print(f"  sampled shingles found in the reference: {n_sh_hit:,} / {n_sh:,} ({100 * n_sh_hit / max(n_sh, 1):.3f}%)")
        for ex in examples:
            print(f"  e.g. {ex[0]}: {ex[1]} of {ex[2]} sampled shingles match")
        print(f"  HEAVILY overlapping docs (>={a.min_hits} hits and >={a.min_frac:.0%} of shingles): {len(strong)}")
        for ex in strong[:10]:
            print(f"    {ex[0]}: {ex[1]} of {ex[2]}")
        if a.flagged_out and strong:
            with open(a.flagged_out, "a", encoding="utf-8") as fo:
                for ex in strong:
                    fo.write(f"{path}\t{ex[0]}\n")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
