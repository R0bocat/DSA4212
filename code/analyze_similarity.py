"""
analyze_similarity.py - figures for the similarity extension.

  python run_queue.py --summarize
  python analyze_similarity.py runs/results.csv --pools data/fw_pools/pools_summary.json --scores data/similarity/scores.npz

fig7_similarity.png : one row per (U, delta) configuration, two columns (domain bpb, general bpb). The x axis is the
                      pool ranking from least to most similar to the domain, plus the random control; the dashed line is
                      the random pool, the dotted line the same cell trained on the full unselected corpus.
fig8_sim_distribution.png : what the similarity score means - the token-weighted distribution of FineWeb-Edu documents,
                      the equal-token pool boundaries, and where the PubMed / FineWeb-Edu validation documents sit.
                      Pass --scores only for the quartile run, and give --dist-name a distinct value for any other
                      pool set, otherwise the two runs overwrite the same file.
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
GREY = "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9})


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
    """One row per (U, delta) configuration, two columns: domain bpb and general bpb.
    x = the four equal-token pools ordered from least to most PubMed-like, plus the random control."""
    cfgs = sorted({(r["U"], r["delta"]) for r in pool_rows})
    if not cfgs:
        return
    pools = summ["pools"]
    qs_all = sorted((p for p in pools if not p.endswith("rnd")), key=lambda p: pools[p]["mean_sim"])
    rnd = next((p for p in pools if p.endswith("rnd")), None)
    if not qs_all:
        return
    nrow = len(cfgs)
    fig, axes = plt.subplots(nrow, 2, figsize=(11.2, 3.4 * nrow), squeeze=False)
    for i, (U, d) in enumerate(cfgs):
        rows = [r for r in pool_rows if (r["U"], r["delta"]) == (U, d)]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["pool"]].append(r)
        qs = [p for p in qs_all if p in by]  # only the pools that were actually trained
        xs = list(range(len(qs)))
        x_rnd = len(qs) + 0.7
        color = COLORS[i % len(COLORS)]
        for j, (key, ylab) in enumerate((("dom", "PubMed test bits/byte"), ("gen", "FineWeb-Edu test bits/byte"))):
            a = axes[i][j]
            ys = [st.mean(r[key] for r in by[p]) for p in qs if p in by]
            xk = [x for x, p in zip(xs, qs) if p in by]
            a.plot(xk, ys, marker="o", color=color, lw=1.9, ms=8, mec="white", mew=1.0, zorder=3)
            for x, y, p in zip(xk, ys, [p for p in qs if p in by]):
                a.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8,
                           color=color, zorder=5,
                           bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
            if rnd and rnd in by:
                yr = st.mean(r[key] for r in by[rnd])
                a.axhline(yr, color=GREY, ls=(0, (4, 3)), lw=1.2, zorder=1)
                a.scatter([x_rnd], [yr], marker="D", s=70, facecolors="white", edgecolors=GREY, lw=1.5, zorder=4)
                a.annotate(f"random\n{yr:.3f}", (x_rnd, yr), textcoords="offset points", xytext=(0, 13), ha="center",
                           fontsize=8, color=GREY, zorder=5,
                           bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
            full = ref.get((U, d))
            if full:
                vals = [r[key] for r in full if not np.isnan(r[key])]
                if vals:
                    a.axhline(st.mean(vals), color=color, ls=":", lw=1.1, alpha=0.8, zorder=1)
            a.set_xticks(xs + ([x_rnd] if rnd else []))
            a.set_xticklabels([f"{p}\n{pools[p]['mean_sim']:+.2f}" for p in qs] +
                              ([f"random\n{pools[rnd]['mean_sim']:+.2f}"] if rnd else []), fontsize=8.5)
            a.set_xlim(-0.5, x_rnd + 0.9)
            a.set_ylabel(ylab)
            a.grid(alpha=0.15, lw=0.6, axis="y")
            a.margins(y=0.26)
            if i == 0:
                a.set_title("Domain ability" if key == "dom" else "General ability")
            if i == nrow - 1:
                a.set_xlabel("general pool, least → most PubMed-like")
        axes[i][0].set_title(f"U = {U // M}M, δ = {d:g}   —   domain ability", fontsize=9.5, color=color)
        axes[i][1].set_title(f"U = {U // M}M, δ = {d:g}   —   general ability", fontsize=9.5, color=color)
    if any(ref.get(cfg) for cfg in cfgs):
        axes[0][0].plot([], [], color=COLORS[0], ls=":", lw=1.1, label="full unselected corpus")
        axes[0][0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("A more PubMed-like general pool helps the domain but costs general ability\n"
                 "dashed = random pool of the same size, dotted = full unselected corpus", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_distribution(scores_path, summ, out):
    z = np.load(scores_path)
    sim, length = z["sim"], z["length"]
    pools = summ["pools"]
    qs = sorted((p for p in pools if not p.endswith("rnd")), key=lambda p: pools[p]["mean_sim"])
    order = np.argsort(-sim)
    cum = np.cumsum(length[order]) / length.sum()
    k = summ["k"]
    edges = sorted(sim[order][np.searchsorted(cum, i / k)] for i in range(1, k))
    lo, hi = np.quantile(sim, [0.001, 0.999])
    refs = {k: float(np.mean(z[k])) for k in z.files if k.startswith("ref_")}
    hi = max([hi] + [v + 0.12 for v in refs.values()])
    fig, (strip, ax) = plt.subplots(2, 1, figsize=(9.6, 5.4), height_ratios=[1.5, 7], sharex=True,
                                    gridspec_kw={"hspace": 0.06})
    ax.hist(sim, bins=np.linspace(lo, hi, 150), weights=length / length.sum(), color="#1f77b4", alpha=0.55, lw=0,
            label="FineWeb-Edu documents (token-weighted)")
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 0.4)
    ax.set_xlim(lo, hi)
    spans = [lo] + edges + [hi]
    # with many pools the bands are narrow, so stagger the labels over two rows
    for i, p in enumerate(qs):
        a0, a1 = spans[i], spans[i + 1]
        strip.axvspan(a0, a1, color=plt.cm.Blues(0.25 + 0.15 * (i % 4)), lw=0)
        strip.text((a0 + a1) / 2, 0.72 if i % 2 == 0 else 0.28, f"{p}\n{pools[p]['mean_sim']:+.2f}",
                   ha="center", va="center", fontsize=8 if len(qs) > 6 else 8.5)
        ax.axvline(a0, color=GREY, ls=(0, (4, 3)), lw=0.9, zorder=1)
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.set_title("Where the equal-token pools sit in the similarity distribution", fontsize=10)
    for sp in ("left", "bottom"):
        strip.spines[sp].set_visible(False)
    rnd = next((p for p in pools if p.endswith("rnd")), None)
    if rnd:
        xr = pools[rnd]["mean_sim"]
        ax.axvline(xr, color=GREY, ls=(0, (4, 3)), lw=1.0, zorder=1)
    for key, color, name in (("ref_fineweb", "#333333", "FineWeb-Edu"), ("ref_pubmed", "#d62728", "PubMed")):
        if key not in refs:
            continue
        ax.axvline(refs[key], color=color, lw=1.8, zorder=5)
    handles = [plt.Line2D([], [], color="#1f77b4", lw=6, alpha=0.55,
                          label="FineWeb-Edu documents (token-weighted)")]
    if "ref_fineweb" in refs:
        handles.append(plt.Line2D([], [], color="#333333", lw=1.8,
                                  label=f"FineWeb-Edu validation docs (mean {refs['ref_fineweb']:+.2f})"))
    if "ref_pubmed" in refs:
        handles.append(plt.Line2D([], [], color="#d62728", lw=1.8,
                                  label=f"PubMed validation docs (mean {refs['ref_pubmed']:+.2f})"))
    if rnd:
        handles.append(plt.Line2D([], [], color=GREY, ls=(0, (4, 3)), lw=1.0,
                                  label=f"random pool (mean {pools[rnd]['mean_sim']:+.2f})"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xlabel("similarity score:  NLL(general-only model) − NLL(PubMed-only model)   [nats/token]")
    ax.set_ylabel("share of tokens (log scale)")
    ax.grid(alpha=0.15, lw=0.6, axis="y")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="runs/results.csv, or the runs folder itself (reads every summary.json; never stale)")
    ap.add_argument("--pools", required=True, help="pools_summary.json from build_pools.py")
    ap.add_argument("--scores", default=None, help="scores.npz from score_similarity.py (for the distribution figure)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="fig7_similarity.png", help="file name of the main figure (use another for deciles)")
    ap.add_argument("--dist-name", default="fig8_sim_distribution.png",
                    help="file name of the distribution figure; give it a distinct name when --pools is a different "
                         "pool set (e.g. the deciles), otherwise the two runs overwrite each other")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(a.pools) as f:
        summ = json.load(f)
    check_stale(a.results)
    pool_rows, ref, records = load(a.results)
    pool_rows = [r for r in pool_rows if r["pool"] in summ["pools"]]
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
        fig_distribution(a.scores, summ, os.path.join(a.out, a.dist_name))
        print(f"wrote {a.dist_name}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
