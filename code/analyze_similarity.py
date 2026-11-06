"""
analyze_similarity.py - figures for the similarity extension.

  python run_queue.py --summarize
  python analyze_similarity.py runs/results.csv --pools data/fw_pools/pools_summary.json --scores data/similarity/scores.npz

fig7_similarity.png : PubMed test bpb (left) and FineWeb-Edu test bpb (right) against how PubMed-like the general pool is.
                      One line per (U, delta) configuration; seeds shown as small dots. Reference band = the same cell
                      trained with the full, unselected general corpus (3 seeds where available).
fig8_sim_distribution.png : what the similarity score means - distribution for FineWeb-Edu documents and PubMed documents,
                      with the pool boundaries.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = 1_000_000
POOL_RE = re.compile(r"^P_U(?P<U>\w+?)_d(?P<d>[\d.]+)_(?P<pool>[qd]\d+|d?rnd)_s(?P<seed>\d+)$")
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def records_from_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            yield dict(name=r["name"], U=int(r["U"]), delta=float(r["delta"]), seed=int(r["seed"]), tokens=int(r["tokens"]),
                       dropout=float(r["dropout"]), dom=float(r["best_test_domain_bpb"]),
                       gen=float(r["best_test_general_bpb"]) if r["best_test_general_bpb"] else np.nan)


def records_from_runs(path):
    """Read runs/<name>/summary.json directly, so no results.csv (which can be stale) is needed."""
    for p in sorted(glob.glob(os.path.join(path, "*", "summary.json"))):
        with open(p) as f:
            s = json.load(f)
        a = s["args"]
        if "best_test_domain_bpb" not in s:
            continue
        yield dict(name=os.path.basename(os.path.dirname(p)), U=int(a["domain_unique_tokens"]), delta=float(a["domain_frac"]),
                   seed=int(a["seed"]), tokens=int(a["tokens"]), dropout=float(a["dropout"]), dom=float(s["best_test_domain_bpb"]),
                   gen=float(s["best_test_general_bpb"]) if "best_test_general_bpb" in s else np.nan)


def load(path):
    records = list(records_from_runs(path) if os.path.isdir(path) else records_from_csv(path))
    pools, ref = [], collections.defaultdict(list)
    for row in records:
        m = POOL_RE.match(row["name"])
        if m:
            row["pool"] = m.group("pool")
            pools.append(row)
        elif row["name"].startswith("S_") and row["tokens"] == 300 * M and row["dropout"] == 0:
            ref[(row["U"], row["delta"])].append(row)
    return pools, ref, records


def check_stale(path):
    """Warn when results.csv is older than the run folders it is supposed to summarise."""
    if os.path.isdir(path):
        return
    runs_dir = os.path.dirname(os.path.abspath(path))
    newer = [p for p in glob.glob(os.path.join(runs_dir, "*", "summary.json")) if os.path.getmtime(p) > os.path.getmtime(path) + 60]
    if newer:
        print(f"WARNING: {path} is older than {len(newer)} finished run(s) (e.g. {os.path.basename(os.path.dirname(newer[0]))}).\n"
              f"         Rebuild it with:  python run_queue.py --summarize   (or pass the runs folder instead of the csv)\n")


def fig_similarity(pool_rows, ref, summ, out):
    cfgs = sorted({(r["U"], r["delta"]) for r in pool_rows})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ci, (U, d) in enumerate(cfgs):
        color = COLORS[ci % len(COLORS)]
        rows = [r for r in pool_rows if (r["U"], r["delta"]) == (U, d)]
        by_pool = collections.defaultdict(list)
        for r in rows:
            by_pool[r["pool"]].append(r)
        for a_, key in ((ax[0], "dom"), (ax[1], "gen")):
            qs = sorted((p for p in by_pool if not p.endswith("rnd") and p in summ["pools"]), key=lambda p: summ["pools"][p]["mean_sim"])
            rnd = next((p for p in by_pool if p.endswith("rnd") and p in summ["pools"]), None)
            xs = [summ["pools"][p]["mean_sim"] for p in qs]
            ys = [st.mean(r[key] for r in by_pool[p]) for p in qs]
            a_.plot(xs, ys, marker="o", color=color, lw=1.6, label=f"U={U // M}M, δ={d:g}")
            for p in qs:
                x = summ["pools"][p]["mean_sim"]
                if len(by_pool[p]) > 1:
                    a_.scatter([x] * len(by_pool[p]), [r[key] for r in by_pool[p]], color=color, s=12, alpha=0.5, zorder=3)
                a_.annotate(p, (x, st.mean(r[key] for r in by_pool[p])), textcoords="offset points", xytext=(4, 5), fontsize=7, color=color)
            if rnd:  # quantity-matched random control
                a_.scatter([summ["pools"][rnd]["mean_sim"]], [st.mean(r[key] for r in by_pool[rnd])], marker="D", s=55,
                           facecolors="white", edgecolors=color, zorder=4, label="random pool (same size)" if ci == 0 and key == "dom" else None)
            full = ref.get((U, d))
            if full:  # same cell with the full, unselected corpus
                vals = [r[key] for r in full]
                m = st.mean(vals)
                a_.axhline(m, color=color, ls=":", lw=1)
                if len(vals) > 1:
                    a_.axhspan(m - st.stdev(vals), m + st.stdev(vals), color=color, alpha=0.08, lw=0)
    pm = summ.get("refs", {}).get("pubmed")
    for a_ in ax:  # PubMed itself is far off to the right of every pool: say so instead of hiding the distance
        if pm:
            a_.annotate(f"PubMed documents themselves: {pm['mean_sim']:+.2f}  →", xy=(1.0, 0.015), xycoords="axes fraction",
                        ha="right", fontsize=8, color="#d62728")
    for a_, t in ((ax[0], "PubMed test bits/byte (lower = better domain model)"), (ax[1], "FineWeb-Edu test bits/byte (general ability)")):
        a_.set_xlabel("similarity of the general pool to PubMed\n(mean log-likelihood ratio, nats/token; right = more PubMed-like)")
        a_.set_ylabel(t)
    ax[0].set_title("Does a more PubMed-like general pool help the domain?\n(dotted = full unselected FineWeb-Edu, band = ±1 seed SD)", fontsize=9)
    ax[1].set_title("...and what does it cost in general ability?", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_distribution(scores_path, summ, out):
    z = np.load(scores_path)
    sim, length = z["sim"], z["length"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    lo, hi = np.quantile(sim, [0.002, 0.998])
    bins = np.linspace(lo, hi, 90)
    ax.hist(sim, bins=bins, weights=length / length.sum(), color="#1f77b4", alpha=0.6, label="FineWeb-Edu documents (token-weighted)")
    for k in z.files:
        if k.startswith("ref_"):
            ax.hist(np.clip(z[k], lo, hi), bins=bins, weights=np.full(len(z[k]), 1 / len(z[k])), histtype="step", lw=1.6,
                    color="#d62728" if "pubmed" in k else "#555555", label=f"{k[4:]} validation documents (reference)")
    order = np.argsort(-sim)
    cum = np.cumsum(length[order]) / length.sum()
    k = summ["k"]
    for i in range(1, k):
        ax.axvline(sim[order][np.searchsorted(cum, i / k)], color="k", ls="--", lw=0.8)
    ax.set_xlabel("similarity score: NLL(general-only model) − NLL(PubMed-only model), nats/token")
    ax.set_ylabel("share of tokens / documents")
    ax.set_title("Where each pool sits (dashed = boundaries of the equal-token pools, q1 = right-most)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="runs/results.csv, or the runs folder itself (reads every summary.json; never stale)")
    ap.add_argument("--pools", required=True, help="pools_summary.json from build_pools.py")
    ap.add_argument("--scores", default=None, help="scores.npz from score_similarity.py (for the distribution figure)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="fig7_similarity.png", help="file name of the main figure (use another for deciles)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(a.pools) as f:
        summ = json.load(f)
    check_stale(a.results)
    pool_rows, ref, records = load(a.results)
    if pool_rows:
        fig_similarity(pool_rows, ref, summ, os.path.join(a.out, a.name))
        print(f"{len(pool_rows)} pool runs; wrote {a.name}")
        print(f"{'config':<14}{'pool':<6}{'similarity':>11}{'domain bpb':>12}{'general bpb':>13}{'n':>3}")
        for U, d in sorted({(r['U'], r['delta']) for r in pool_rows}):
            by = collections.defaultdict(list)
            for r in pool_rows:
                if (r["U"], r["delta"]) == (U, d):
                    by[r["pool"]].append(r)
            for p in sorted(by, key=lambda p: -summ["pools"].get(p, {}).get("mean_sim", 0)):
                print(f"U={U // M}M d={d:<5g} {p:<6}{summ['pools'].get(p, {}).get('mean_sim', float('nan')):>+11.3f}"
                      f"{st.mean(r['dom'] for r in by[p]):>12.4f}{st.mean(r['gen'] for r in by[p]):>13.4f}{len(by[p]):>3}")
    else:
        p_names = [r["name"] for r in records if r["name"].startswith("P_")]
        print(f"no pool runs matched. Read {len(records)} runs from {a.results}; {len(p_names)} of them start with 'P_'.")
        if p_names:
            print("  names that did not match the expected pattern, e.g.:", p_names[:3])
        else:
            print("  none of the runs is a pool run: the file is probably out of date. Run `python run_queue.py --summarize`,\n"
                  "  or pass the runs folder itself:  python analyze_similarity.py runs --pools ...")
    if a.scores:
        fig_distribution(a.scores, summ, os.path.join(a.out, "fig8_sim_distribution.png"))
        print("wrote fig8_sim_distribution.png")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
