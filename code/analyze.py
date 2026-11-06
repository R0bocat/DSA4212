"""
analyze.py - turn runs/results.csv into the report's tables and figures.

  python run_queue.py --summarize          # rebuilds runs/results.csv
  python analyze.py runs/results.csv       # writes figures/*.png and prints the numbers behind them

All bpb numbers are TEST bits-per-byte at the best-validation checkpoint (selected on validation, reported on test).
Naming convention (from run_queue.py): S_* = mixture grid, C_U* = domain-only baselines, pilot_*/C_lr* are ignored.
"""
import argparse
import collections
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = 1_000_000
U_ORDER = [3 * M, 10 * M, 40 * M, 0]  # the main grid; 0 = all available domain data (abundant)
EXT_U = [1 * M, 20 * M]              # extra scarcity levels used only as an out-of-sample test of the delta* rule
COLORS = {1 * M: "#8c564b", 3 * M: "#d62728", 10 * M: "#ff7f0e", 20 * M: "#bcbd22", 40 * M: "#2ca02c", 0: "#1f77b4"}
MARKERS = {1 * M: "v", 3 * M: "s", 10 * M: "o", 20 * M: "P", 40 * M: "^", 0: "D"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def lab(U):
    return "abundant" if U == 0 else f"{U // M}M"


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r["name"].startswith("S_") or r["name"].startswith("C_U")):
                continue
            rows.append(dict(
                name=r["name"], U=int(r["U"]), delta=float(r["delta"]), tokens=int(r["tokens"]), lr=float(r["lr"]),
                dropout=float(r["dropout"]), seed=int(r["seed"]), epochs=float(r["domain_epochs"]),
                best_step=int(r["best_step"]), dom=float(r["best_test_domain_bpb"]), dom_final=float(r["final_test_domain_bpb"]),
                gen=float(r["best_test_general_bpb"]) if r["best_test_general_bpb"] else np.nan,
                gen_final=float(r["final_test_general_bpb"]) if r["final_test_general_bpb"] else np.nan))
    return rows


def cells(rows):
    """Mixture-grid cells at the standard 300M budget, pooled over seeds: (U, delta) -> list of runs."""
    c = collections.defaultdict(list)
    for r in rows:
        if r["name"].startswith("S_") and r["tokens"] == 300 * M and r["dropout"] == 0:
            c[(r["U"], r["delta"])].append(r)
    return c


def baselines(rows, grid):
    """Best non-mixture alternative per U (test domain bpb): every domain-only control (any dropout / epoch count / LR),
    the full-budget domain-only runs (mean over seeds) and the general-only run. Taking the minimum over several noisy
    runs makes the baseline slightly optimistic, which is the conservative direction for the claim that mixing helps."""
    gen = grid.get((0, 0.0))
    gen_cand = [(st.mean(r["dom"] for r in gen), "general-only")] if gen else []
    best = {}
    for U in {r["U"] for r in rows}:
        cand = [(r["dom"], r["name"]) for r in rows if r["name"].startswith("C_U") and r["U"] == U]
        if (U, 1.0) in grid:
            rs = grid[(U, 1.0)]
            cand.append((st.mean(r["dom"] for r in rs), f"S_U{lab(U)}_d1.0 (mean of {len(rs)})"))
        if cand:  # only compare against general-only when domain-only runs exist for this U
            cand += gen_cand
            d, n = min(cand)
            best[U] = {"dom": d, "name": n}
    return best


def best_mixtures(rows, grid):
    """Best mixture per U over every variant: seed-averaged no-dropout cells and single dropout runs (300M tokens).
    Both sides of the comparison (baseline and mixture) get to use dropout. Taking the minimum over many noisy
    configurations is slightly optimistic for the mixture (winner's curse, roughly 0.003-0.005 bpb)."""
    out = {}
    for (U, d), rs in grid.items():
        if 0 < d < 1:
            v = st.mean(r["dom"] for r in rs)
            if U not in out or v < out[U][0]:
                out[U] = (v, f"δ={d:g}")
    for r in rows:
        if r["name"].startswith("S_") and r["tokens"] == 300 * M and r["dropout"] > 0 and 0 < r["delta"] < 1:
            if r["U"] not in out or r["dom"] < out[r["U"]][0]:
                out[r["U"]] = (r["dom"], f"δ={r['delta']:g}, drop {r['dropout']:g}")
    return out


def noise(grid):
    sds = [st.stdev([r["dom"] for r in rs]) for rs in grid.values() if len(rs) >= 2]
    n = sum(len(rs) - 1 for rs in grid.values() if len(rs) >= 2)
    pooled = (sum(st.variance([r["dom"] for r in rs]) * (len(rs) - 1) for rs in grid.values() if len(rs) >= 2) / n) ** 0.5 if n else np.nan
    return pooled, n, sds


YMAX = 1.40


def fig_main(grid, base, general_only, sd, out, best_mix):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    for U in U_ORDER:
        pts = sorted((d, st.mean(r["dom"] for r in rs)) for (u, d), rs in grid.items() if u == U and d > 0)
        if not pts:
            continue
        a.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[U], color=COLORS[U], label=f"U={lab(U)}", lw=1.6)
        if U in base:
            a.axhline(base[U]["dom"], color=COLORS[U], ls=":", lw=1.1, alpha=0.8)
        for (u, d), rs in grid.items():
            if u != U:
                continue
            if len(rs) > 1:  # show individual seeds where there are several
                a.scatter([d] * len(rs), [r["dom"] for r in rs], color=COLORS[U], s=14, alpha=0.5, zorder=3)
            fin, best = st.mean(r["dom_final"] for r in rs), st.mean(r["dom"] for r in rs)
            if fin - best > 0.003:  # overfit: the final weights are worse than the best-validation checkpoint
                if fin < YMAX:
                    a.scatter([d], [fin], facecolors="none", edgecolors=COLORS[U], s=46, marker=MARKERS[U], zorder=4)
                else:
                    a.annotate(f"final {fin:.2f} ↑", (d, YMAX - 0.004), fontsize=6.5, color=COLORS[U], ha="center", va="top")
    if general_only is not None:
        a.axhline(general_only, color="k", ls="--", lw=1.1)
        a.text(0.031, general_only + 0.006, "general-only (no domain data)", fontsize=8)
    a.set_xscale("log")
    a.set_xticks([0.03, 0.1, 0.3, 0.6, 1.0])
    a.set_xticklabels(["0.03", "0.1", "0.3", "0.6", "1.0"])
    a.set_xlabel("domain share δ of training sequences")
    a.set_ylabel("PubMed test bits/byte (lower is better)")
    a.set_ylim(top=YMAX)
    a.set_title("Domain performance vs mixture share (filled = best-val checkpoint,\nhollow = final weights if overfit; dotted = best domain-only baseline)", fontsize=9)
    a.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.12, 0.86))
    a.text(0.02, 0.27, f"seed SD ≈ {sd:.4f} bpb\n(pooled over 6 cells, 12 dof)", fontsize=8, color="gray", transform=a.transAxes)

    b = ax[1]
    Us = [U for U in sorted(set(U_ORDER + EXT_U), key=lambda u: (u == 0, u)) if U in base and U in best_mix]
    gains = [base[U]["dom"] - best_mix[U][0] for U in Us]
    bestd = [best_mix[U][1] for U in Us]
    bars = b.bar(range(len(Us)), gains, color=[COLORS[U] for U in Us], width=0.6)
    for bar, U in zip(bars, Us):
        if U in EXT_U:
            bar.set_hatch("//")
    b.axhline(0, color="k", lw=0.8)
    b.axhspan(-2 * sd * 2**0.5, 2 * sd * 2**0.5, color="gray", alpha=0.15, lw=0)
    for i, (g, d) in enumerate(zip(gains, bestd)):
        b.text(i, g + (0.005 if g >= 0 else -0.006), f"{g:+.3f}\n({d})", ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
    b.set_ylim(min(min(gains) - 0.07, -0.04), max(gains) + 0.045)
    b.set_xticks(range(len(Us)))
    b.set_xticklabels([lab(U) for U in Us])
    b.set_xlabel("unique domain tokens U")
    b.set_ylabel("gain over best domain-only baseline (bpb)")
    b.set_title("Gain from general data: best mixture vs best non-mixture\n(both may use dropout; grey = ±2σ of a difference; hatched = validation levels)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(grid, general_only, general_only_gen, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for U in U_ORDER:
        pts = sorted((d, st.mean(r["dom"] for r in rs), st.mean(r["gen"] for r in rs) - general_only_gen)
                     for (u, d), rs in grid.items() if u == U and d > 0)
        if not pts:
            continue
        ax.plot([p[2] for p in pts], [p[1] for p in pts], marker=MARKERS[U], color=COLORS[U], label=f"U={lab(U)}", lw=1.2)
        for d, y, x in pts:
            ax.annotate(f"{d:g}", (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7, color=COLORS[U])
    if general_only is not None:
        ax.scatter([0], [general_only], marker="*", s=150, color="k", zorder=5, label="general-only")
    ax.set_xscale("symlog", linthresh=0.02)
    ticks = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("extra FineWeb-Edu test bpb vs general-only run (symlog)")
    ax.set_ylabel("PubMed (domain) test bits/byte")
    ax.set_title("Trade-off between domain and general ability\n(labels = δ; bottom-left is best)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_repetition(grid, rows, base, out):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    a = ax[0]
    for d, ls in ((0.1, "-"), (0.3, "-")):
        pts = sorted((st.mean(r["epochs"] for r in rs), st.mean(r["dom"] for r in rs)) for (u, dd), rs in grid.items() if dd == d and u != 0)
        fresh = grid.get((0, d))
        if fresh:
            pts = [(st.mean(r["epochs"] for r in fresh), st.mean(r["dom"] for r in fresh))] + pts
        a.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ls=ls, label=f"δ={d}")
    a.set_xscale("log")
    a.set_xticks([0.2, 0.5, 1, 2, 5, 10, 30])
    a.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "30"])
    a.set_xlabel("epochs over the domain data")
    a.set_ylabel("PubMed test bits/byte")
    a.set_title("Cost of repeating domain data at fixed δ", fontsize=10)
    a.legend(frameon=False, fontsize=8, title="U varies along each line", title_fontsize=8)
    b = ax[1]
    for U in U_ORDER:
        ctl = sorted((r["epochs"], r["dom"], r["lr"]) for r in rows if r["name"].startswith("C_U") and r["U"] == U and r["dropout"] == 0)
        if not ctl:
            continue
        lo = collections.defaultdict(lambda: np.inf)
        for e, y, _ in ctl:
            lo[e] = min(lo[e], y)
        es = sorted(lo)
        b.plot(es, [lo[e] for e in es], marker=MARKERS[U], color=COLORS[U], label=f"U={lab(U)}", lw=1.2)
    b.set_xscale("log")
    b.set_xticks([2, 3, 5, 10, 20, 30])
    b.set_xticklabels(["2", "3", "5", "10", "20", "30"])
    b.set_xlabel("epochs (each run has its own full LR schedule)")
    b.set_ylabel("PubMed test bits/byte")
    b.set_title("Fair domain-only baselines: best of lr and 2·lr", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_rule(grid, sd, out, B=300 * M, e_star=10, u_abundant=550 * M):
    """Left: the new scarcity levels with the rule's predicted best delta. Right: observed optimum plateau vs the rule."""
    thr = 2 * sd * 2 ** 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    for U in EXT_U:
        pts = sorted((d, st.mean(r["dom"] for r in rs), st.mean(r["epochs"] for r in rs)) for (u, d), rs in grid.items() if u == U)
        if not pts:
            continue
        a.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[U], color=COLORS[U], label=f"U={lab(U)}", lw=1.6)
        for d, y, ep in pts:
            a.annotate(f"{ep:.0f} ep", (d, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color=COLORS[U])
        a.axvline(min(1.0, e_star * U / B), color=COLORS[U], ls="--", lw=1)
    a.set_xscale("log")
    a.set_xticks([0.01, 0.03, 0.1, 0.3, 0.67, 1.0])
    a.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.67", "1"])
    a.set_xlabel("domain share δ")
    a.set_ylabel("PubMed test bits/byte")
    a.set_title("Out-of-sample levels (dashed = predicted best δ)", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    b = ax[1]
    Ug = np.logspace(np.log10(0.6e6), np.log10(700e6), 100)
    b.plot(Ug / M, np.minimum(1.0, e_star * Ug / B), "k--", lw=1.2, label=f"pre-registered: δ* = min(1, {e_star}·U/B)")
    b.plot(Ug / M, np.minimum(1.0, 15 * Ug / B), color="gray", ls=":", lw=1.4, label="refit on the finer grid: 15·U/B")
    for U in sorted({u for (u, d) in grid}, key=lambda u: (u == 0, u)):
        means = {d: st.mean(r["dom"] for r in rs) for (u, d), rs in grid.items() if u == U}
        if len(means) < 3:  # need at least 3 deltas to say where the optimum is
            continue
        best = min(means, key=means.get)
        tie = [d for d, v in means.items() if v <= means[best] + thr]
        x = (u_abundant if U == 0 else U) / M
        b.plot([x, x], [min(tie), max(tie)], color=COLORS[U], lw=4, alpha=0.45, solid_capstyle="round")
        b.scatter([x], [best], color=COLORS[U] if U not in EXT_U else "white", edgecolors=COLORS[U], s=50, zorder=3, marker=MARKERS[U])
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_yticks([0.03, 0.1, 0.3, 1.0])
    b.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    b.set_xlabel("unique domain tokens U (millions; abundant plotted at 550M)")
    b.set_ylabel("best domain share δ")
    b.set_title("Observed optimum (bar = δ values tied with the best within 2σ)\nhollow = new levels, never used to set the rule", fontsize=9)
    b.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_horizon(rows, grid, out, U=10 * M):
    """Optimum at half the horizon (150M tokens) vs the standard 300M, at U=10M, in delta and in domain epochs."""
    short = sorted((r["delta"], r["dom"], r["epochs"]) for r in rows
                   if r["name"].startswith("S_") and r["U"] == U and r["tokens"] == 150 * M and r["dropout"] == 0)
    full = sorted((d, st.mean(r["dom"] for r in rs), st.mean(r["epochs"] for r in rs)) for (u, d), rs in grid.items() if u == U and d > 0)
    if not short:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for a_, xi, xlabel in ((ax[0], 0, "domain share δ"), (ax[1], 2, "epochs over the domain data")):
        a_.plot([p[xi] for p in full], [p[1] for p in full], marker="o", color="#ff7f0e", label="300M tokens (45 min)")
        a_.plot([p[xi] for p in short], [p[1] for p in short], marker="s", color="#9467bd", label="150M tokens")
        a_.set_xscale("log")
        a_.set_xlabel(xlabel)
        a_.set_ylabel("PubMed test bits/byte")
        a_.legend(frameon=False, fontsize=8)
    ax[0].set_xticks([0.03, 0.1, 0.2, 0.3, 0.45, 0.67, 1.0])
    ax[0].set_xticklabels(["0.03", "0.1", "0.2", "0.3", "0.45", "0.67", "1"])
    ax[1].set_xticks([3, 5, 10, 15, 30])
    ax[1].set_xticklabels(["3", "5", "10", "15", "30"])
    ax[0].set_title(f"U={U // M}M: best δ is larger at the shorter horizon (150M grid is coarse: 3 points)", fontsize=9)
    ax[1].set_title("in epochs the optimum is similar (about 10 vs 13)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_heatmap(grid, base, out):
    Us = [U for U in U_ORDER if U in base and any(u == U for (u, d) in grid)]
    ds = [d for d in (0.03, 0.1, 0.3, 0.6, 1.0) if any(dd == d for (_, dd) in grid)]
    Z = np.full((len(Us), len(ds)), np.nan)
    for i, U in enumerate(Us):
        for j, d in enumerate(ds):
            if (U, d) in grid:
                Z[i, j] = st.mean(r["dom"] for r in grid[(U, d)]) - base[U]["dom"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lim = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_yticks(range(len(Us)))
    ax.set_yticklabels([lab(U) for U in Us])
    for i in range(len(Us)):
        for j in range(len(ds)):
            if not np.isnan(Z[i, j]):
                ep = st.mean(r["epochs"] for r in grid[(Us[i], ds[j])])
                ax.text(j, i, f"{Z[i, j]:+.3f}\n{ep:.1f} ep", ha="center", va="center", fontsize=7,
                        color="white" if abs(Z[i, j]) > 0.6 * lim else "black")
    ax.set_xlabel("domain share δ")
    ax.set_ylabel("unique domain tokens U")
    ax.set_title("Test bpb minus best domain-only baseline for that U (blue = mixture better)\ncell text: Δbpb and epochs over the domain data", fontsize=9)
    fig.colorbar(im, ax=ax, label="Δ bpb")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.results)
    grid = cells(rows)
    base = baselines(rows, grid)
    gen_only = grid.get((0, 0.0))
    general_only = st.mean(r["dom"] for r in gen_only) if gen_only else None
    sd, dof, sds = noise(grid)

    print(f"seed noise: pooled SD = {sd:.4f} bpb over {dof} degrees of freedom (cells with repeats: {len(sds)})")
    print("best domain-only baseline per U:", {lab(U): (b["name"], round(b["dom"], 4)) for U, b in base.items()})
    best_mix = best_mixtures(rows, grid)
    fig_main({k: v for k, v in grid.items() if k[1] > 0}, base, general_only, sd, os.path.join(a.out, "fig1_main.png"), best_mix)
    gen_only_gen = st.mean(r["gen"] for r in gen_only) if gen_only else 0.0
    fig_tradeoff(grid, general_only, gen_only_gen, os.path.join(a.out, "fig2_tradeoff.png"))
    fig_repetition({k: v for k, v in grid.items() if k[1] > 0}, rows, base, os.path.join(a.out, "fig3_repetition.png"))
    fig_heatmap({k: v for k, v in grid.items() if k[1] > 0}, base, os.path.join(a.out, "fig4_heatmap.png"))
    n_fig = 4
    if any(u in EXT_U for (u, d) in grid):
        fig_rule({k: v for k, v in grid.items() if k[1] > 0}, sd, os.path.join(a.out, "fig5_rule.png"))
        n_fig = 5

    if fig_horizon(rows, grid, os.path.join(a.out, "fig6_horizon.png")):
        n_fig += 1

    print("\ngain over the best non-mixture baseline (test domain bpb; positive = mixing helps; both sides may use dropout):")
    for U in sorted(base, key=lambda u: (u == 0, u)):
        if U in best_mix:
            v, label_ = best_mix[U]
            print(f"  U={lab(U):>8}: baseline {base[U]['dom']:.4f} [{base[U]['name']}]  best mixture {v:.4f} ({label_})  gain {base[U]['dom'] - v:+.4f}")
    missing = [lab(U) for U in sorted(best_mix, key=lambda u: (u == 0, u)) if U not in base]
    if missing:
        print(f"  (no domain-only baseline yet for U={', '.join(missing)}, so no gain is computed)")
    print(f"wrote {n_fig} figures to {a.out}/")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
