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
EXT_U = [1 * M, 20 * M]              # extra scarcity levels used only as an out-of-sample test of the constant-epochs fit
COLORS = {1 * M: "#8c564b", 3 * M: "#d62728", 10 * M: "#ff7f0e", 20 * M: "#bcbd22", 40 * M: "#2ca02c", 0: "#1f77b4"}
MARKERS = {1 * M: "v", 3 * M: "s", 10 * M: "o", 20 * M: "P", 40 * M: "^", 0: "D"}
GREY = "#555555"
NOISE = "#999999"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9})


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


def _series(grid, U):
    """(delta, mean bpb, seed SD) for one scarcity level, sorted by delta."""
    out = []
    for (u, d), rs in grid.items():
        if u != U or d <= 0:
            continue
        v = [r["dom"] for r in rs]
        out.append((d, st.mean(v), st.stdev(v) if len(v) > 1 else 0.0))
    return sorted(out)


def _best_label(label):
    """'δ=0.1, drop 0.1' -> 'δ=0.1, dropout 0.1'."""
    parts = [p.strip() for p in label.split(",")]
    return parts[0] + (", dropout " + parts[1].split()[-1] if len(parts) > 1 else "")


def fig_main(grid, base, general_only, sd, out, best_mix):
    """Left: domain bpb against delta, one line per scarcity level. Right: the best gain over the domain-only baseline."""
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})

    a = ax[0]
    for U in U_ORDER:
        pts = _series(grid, U)
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        es = np.array([p[2] for p in pts])
        a.fill_between(xs, ys - es, ys + es, color=COLORS[U], alpha=0.15, lw=0, zorder=1)
        a.plot(xs, ys, marker=MARKERS[U], color=COLORS[U], lw=1.9, ms=6.5, mec="white", mew=0.9, zorder=3, label=lab(U))
        i = int(np.argmin(ys))
        a.scatter([xs[i]], [ys[i]], s=200, facecolors="none", edgecolors=COLORS[U], lw=1.8, zorder=4)
    if general_only is not None:
        a.axhline(general_only, color=GREY, ls=(0, (4, 3)), lw=1.2, zorder=2)
        a.text(0.0105, general_only + 0.008, "general-only (no domain data)", fontsize=8.5, color=GREY)
    a.set_xscale("log")
    a.set_xlim(0.008, 1.3)
    a.set_ylim(0.965, 1.40)
    a.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
    a.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
    a.set_xlabel("domain share δ (fraction of training tokens from PubMed)")
    a.set_ylabel("PubMed test bits/byte  (lower is better)")
    a.set_title("Domain accuracy vs. domain share, one line per scarcity level")
    handles, labels = a.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#333333", mew=1.6, ms=10))
    labels.append("best δ for that U")
    a.legend(handles, labels, frameon=False, ncol=2, loc="lower left", handletextpad=0.4, columnspacing=1.0,
             borderaxespad=0.6)

    b = ax[1]
    Us = [U for U in sorted(U_ORDER, key=lambda u: (u == 0, u)) if U in base and U in best_mix]
    gains = [base[U]["dom"] - best_mix[U][0] for U in Us]
    ypos = np.arange(len(Us))[::-1]
    b.barh(ypos, gains, color=[COLORS[U] for U in Us], height=0.62, edgecolor="white", lw=0.8)
    b.axvline(0, color="#333333", lw=1.0)
    band = 2 * sd * 2 ** 0.5
    b.axvspan(-band, band, color=NOISE, alpha=0.20, lw=0)
    for y, g, U in zip(ypos, gains, Us):
        b.text(g + (0.004 if g >= 0 else -0.004), y, f"{g:+.3f}  {_best_label(best_mix[U][1])}",
               ha="left" if g >= 0 else "right", va="center", fontsize=8.5, color=COLORS[U])
    b.set_yticks(ypos)
    b.set_yticklabels([lab(U) for U in Us])
    b.set_ylim(-0.6, len(Us) - 0.4)
    b.set_xlim(min(gains) - 0.075, max(gains) + 0.075)
    b.set_ylabel("unique domain tokens U")
    b.set_xlabel("gain over best domain-only baseline (bpb)")
    b.set_title("Best gain from mixing in general data")
    b.text(0.5, -0.30, f"grey band = ±2σ of a difference (σ = {sd:.4f} bpb)", transform=b.transAxes,
           ha="center", va="top", fontsize=8, color="#666666")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(grid, general_only, general_only_gen, out):
    """Small multiples: domain bpb against general bpb for each scarcity level, points labelled by delta."""
    Us = [U for U in sorted(U_ORDER, key=lambda u: (u == 0, u))
          if any(u == U and 0 < d < 1 for (u, d) in grid)]
    if not Us:
        return
    ncol = 2
    nrow = int(np.ceil(len(Us) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, U in enumerate(Us):
        a = axes[k]
        pts = sorted((d, st.mean(r["gen"] for r in rs), st.mean(r["dom"] for r in rs))
                     for (u, d), rs in grid.items() if u == U and 0 < d < 1)
        pts = [p for p in pts if not np.isnan(p[1])]
        if not pts:
            a.set_visible(False)
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        a.plot(xs, ys, color=COLORS[U], lw=1.3, alpha=0.5, zorder=1)
        a.scatter(xs, ys, s=70, color=COLORS[U], marker=MARKERS[U], edgecolors="white", linewidths=1.0, zorder=3)
        best = min(pts, key=lambda p: p[2])
        # label the two ends and the best point, but never the same point twice
        labelled = []
        for p in (pts[0], best, pts[-1]):
            if any(p[0] == q[0] for q in labelled):
                continue
            labelled.append(p)
            a.annotate(f"δ={p[0]:g}", (p[1], p[2]), textcoords="offset points", xytext=(9, 7), fontsize=8,
                       color=COLORS[U], zorder=6,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        if len(pts) > 2:
            a.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                       arrowprops=dict(arrowstyle="-|>", color=COLORS[U], lw=1.2, shrinkA=6, shrinkB=6), zorder=2)
        a.scatter([best[1]], [best[2]], s=230, facecolors="none", edgecolors=COLORS[U], lw=1.8, zorder=4)
        a.scatter([general_only_gen], [general_only], marker="*", s=230, color="#111111", zorder=5)
        a.axhline(general_only, color=GREY, ls=(0, (4, 3)), lw=1.0, zorder=1)
        a.set_title(f"U = {lab(U)}")
        a.grid(alpha=0.15, lw=0.6)
        a.margins(x=0.22, y=0.20)
    for k in range(len(Us), len(axes)):
        axes[k].set_visible(False)
    for k in range(0, len(axes), ncol):
        axes[k].set_ylabel("PubMed test bits/byte")
    for k in range(len(axes) - ncol, len(axes)):
        axes[k].set_xlabel("FineWeb-Edu test bits/byte")
    fig.suptitle("Domain/general trade-off: right = worse general ability, down = better domain ability\n"
                 "star = general-only run, dashed line = its domain bpb, circle = best δ for that U", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_repetition(grid, out, sd=None):
    """Left: domain bpb against epochs over the domain data, one line per scarcity level.
    Right: the same curves shifted to their own minimum, which shows where the repetition cost sets in."""
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9), sharex=True)
    handles, labels = [], []

    a = ax[0]
    for U in U_ORDER:
        pts = sorted((st.mean(r["epochs"] for r in rs), st.mean(r["dom"] for r in rs))
                     for (u, d), rs in grid.items() if u == U and d > 0)
        if not pts:
            continue
        ln, = a.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[U], color=COLORS[U], lw=1.8, ms=6.5,
                     mec="white", mew=0.9, label=lab(U))
        handles.append(ln)
        labels.append(lab(U))
        i = int(np.argmin([p[1] for p in pts]))
        a.scatter([pts[i][0]], [pts[i][1]], s=190, facecolors="none", edgecolors=COLORS[U], lw=1.7, zorder=4)
    a.axvspan(8, 15, color=NOISE, alpha=0.16, lw=0, zorder=0)
    a.text(11, 1.395, "8–15 epochs: the optimum for U ≤ 10M", ha="center", fontsize=8, color="#555555")
    a.set_xscale("log")
    a.set_xlim(0.012, 200)
    a.set_ylim(0.96, 1.42)
    a.set_xticks([0.02, 0.1, 1, 10, 100])
    a.set_xticklabels(["0.02", "0.1", "1", "10", "100"])
    a.set_xlabel("epochs over the domain data  (δ·B / U)")
    a.set_ylabel("PubMed test bits/byte")
    a.set_title("Domain accuracy vs. how often the domain data is repeated")

    b = ax[1]
    for U in U_ORDER:
        pts = sorted((st.mean(r["epochs"] for r in rs), st.mean(r["dom"] for r in rs))
                     for (u, d), rs in grid.items() if u == U and d > 0)
        if not pts:
            continue
        lo = min(p[1] for p in pts)
        b.plot([p[0] for p in pts], [p[1] - lo for p in pts], marker=MARKERS[U], color=COLORS[U], lw=1.8, ms=6.5,
               mec="white", mew=0.9)
    b.axvspan(8, 15, color=NOISE, alpha=0.16, lw=0, zorder=0)
    b.axhline(0, color="#333333", lw=1.0)
    if sd:
        b.axhspan(0, 2 * sd * 2 ** 0.5, color=NOISE, alpha=0.22, lw=0, zorder=0)
        b.annotate("2σ of a difference", xy=(0.985, 2 * sd * 2 ** 0.5), xycoords=("axes fraction", "data"),
                   textcoords="offset points", xytext=(0, 3), ha="right", fontsize=8, color="#555555")
    b.set_xscale("log")
    b.set_xlim(0.012, 200)
    b.set_ylim(-0.012, 0.30)
    b.set_xticks([0.02, 0.1, 1, 10, 100])
    b.set_xticklabels(["0.02", "0.1", "1", "10", "100"])
    b.set_xlabel("epochs over the domain data  (δ·B / U)")
    b.set_ylabel("PubMed bpb − best for that U")
    b.set_title("Shifted to each level's own best: the cost sets in past ~15 epochs")
    b.annotate("← too little domain data seen", xy=(0.015, 0.985), xycoords="axes fraction",
               fontsize=8.5, color="#555555", ha="left", va="top")
    b.annotate("too many epochs (repetition) →", xy=(0.985, 0.985), xycoords="axes fraction", fontsize=8.5,
               color="#555555", ha="right", va="top")

    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               title="unique domain tokens", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fit_epochs(grid, B=300 * M, holdout=(), sd=None):
    """Constant-epochs fit through the observed optima: delta* = E*U/B, fitted in log space so every level counts
    equally in relative terms. Levels whose optimum is delta=1 are saturated (censored) and carry no information
    about E, so they are left out, as are the `holdout` levels that the left panels test.

    With `sd` the per-level 2-sigma tie band is converted into a range of epochs, and the ranges of the levels the
    fit used are intersected: that intersection is the band of E consistent with all of them. Returns
    (E, [(U, best delta), ...], (E_lo, E_hi)) where the list is the points the fit used."""
    pts = []
    for U in sorted({u for (u, d) in grid if u > 0 and u not in holdout}):
        means = {d: st.mean(r["dom"] for r in rs) for (u, d), rs in grid.items() if u == U}
        if len(means) < 3:
            continue
        best = min(means, key=means.get)
        if best < 1.0:
            pts.append((U, best))
    if not pts:
        return np.nan, pts, (np.nan, np.nan)
    e_fit = float(np.exp(np.mean([np.log(d * B / U) for U, d in pts])))
    if sd is None:
        return e_fit, pts, (np.nan, np.nan)
    thr = 2 * sd * 2 ** 0.5
    lo, hi = -np.inf, np.inf
    for U, _ in pts:
        series = sorted((d, st.mean(r["dom"] for r in rs)) for (u, d), rs in grid.items() if u == U)
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        level = min(ys) + thr
        cross = []
        for i in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
                cross.append(xs[i] + (level - y0) / (y1 - y0) * (xs[i + 1] - xs[i]))
        if not cross:
            continue
        lo = max(lo, min(cross) * B / U)
        hi = min(hi, max(cross) * B / U)
    return e_fit, pts, (lo, hi)


def fig_rule(grid, sd, out, B=300 * M, u_abundant=550 * M):
    """Left: the out-of-sample scarcity levels with the fitted delta. Right: observed optimum vs a constant-epochs
    fit, drawn as a band of epochs consistent with the in-sample levels. No rule is proposed, so the figure shows
    one empirical fit and its uncertainty instead of competing lines."""
    thr = 2 * sd * 2 ** 0.5
    e_fit, fit_pts, (e_lo, e_hi) = fit_epochs(grid, B, holdout=EXT_U, sd=sd)
    band = np.isfinite(e_lo) and np.isfinite(e_hi) and e_hi > e_lo
    fig = plt.figure(figsize=(12.4, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], hspace=0.42, wspace=0.24)
    left = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]

    for a, U in zip(left, EXT_U):
        pts = sorted((d, st.mean(r["dom"] for r in rs)) for (u, d), rs in grid.items() if u == U)
        if not pts:
            a.set_visible(False)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        best = min(ys)
        a.axhspan(best, best + thr, color=COLORS[U], alpha=0.16, lw=0, zorder=0)
        a.plot(xs, ys, marker=MARKERS[U], color=COLORS[U], lw=1.9, ms=7, mec="white", mew=0.9, zorder=3)
        i = int(np.argmin(ys))
        a.scatter([xs[i]], [ys[i]], s=190, facecolors="none", edgecolors=COLORS[U], lw=1.8, zorder=4)
        pred = min(1.0, e_fit * U / B)
        a.axvline(pred, color=COLORS[U], ls="--", lw=1.3, zorder=1)
        a.annotate(f"fit δ*={pred:.2f}", (pred, 0.97), xycoords=("data", "axes fraction"), textcoords="offset points",
                   xytext=(-5, 0), fontsize=8, color=COLORS[U], rotation=90, va="top", ha="right")
        a.annotate(f"best δ={xs[i]:g}", (xs[i], ys[i]), textcoords="offset points", xytext=(10, -12), fontsize=8,
                   color=COLORS[U])
        a.set_xscale("log")
        a.set_xlim(0.008, 1.3)
        a.set_ylim(best - 0.035, max(ys) + 0.02)
        a.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
        a.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
        a.set_ylabel("PubMed bpb")
        a.set_title(f"U = {lab(U)} (out of sample)", fontsize=9.5)
        a.grid(alpha=0.15, lw=0.6)
    left[1].set_xlabel("domain share δ")
    left[0].text(0.98, 0.06, "shaded = δ within 2σ of the best", transform=left[0].transAxes, ha="right", va="bottom",
                 fontsize=8, color="#666666")

    b = fig.add_subplot(gs[:, 1])
    Ug = np.logspace(np.log10(0.6e6), np.log10(700e6), 100)
    if band:
        b.fill_between(Ug / M, np.minimum(1.0, e_lo * Ug / B), np.minimum(1.0, e_hi * Ug / B),
                       color="#333333", alpha=0.13, lw=0, zorder=0)
    b.plot(Ug / M, np.minimum(1.0, e_fit * Ug / B), "k--", lw=1.4, zorder=1)
    label = f"constant epochs\nδ* = min(1, {e_fit:.0f}·U/B)"
    if band:
        label += f"\nband: E = {e_lo:.0f}–{e_hi:.0f} (2σ)"
    b.annotate(label, (0.62, 0.30), fontsize=8.5, color="#222222", ha="left", va="bottom")
    for U in sorted({u for (u, d) in grid}, key=lambda u: (u == 0, u)):
        means = {d: st.mean(r["dom"] for r in rs) for (u, d), rs in grid.items() if u == U}
        if len(means) < 3:
            continue
        best = min(means, key=means.get)
        tie = [d for d, v in means.items() if v <= means[best] + thr]
        x = (u_abundant if U == 0 else U) / M
        b.plot([x, x], [min(tie), max(tie)], color=COLORS[U], lw=5, alpha=0.35, solid_capstyle="round", zorder=2)
        b.scatter([x], [best], color=COLORS[U] if U not in EXT_U else "white", edgecolors=COLORS[U], s=80,
                  zorder=3, marker=MARKERS[U], lw=1.7)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_yticks([0.03, 0.1, 0.3, 1.0])
    b.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    b.set_xlabel("unique domain tokens U (millions; abundant plotted at 550M)")
    b.set_ylabel("best domain share δ")
    b.set_title("Observed optimum vs a constant-epochs fit")
    b.text(0.98, 0.03, "bars = δ tied with the best within 2σ\nhollow = out-of-sample levels\n"
                       f"band = epochs consistent with the {len(fit_pts)} unsaturated in-sample levels\n"
                       "(40M is already at δ=1, so it does not constrain E)",
           transform=b.transAxes, ha="right", va="bottom", fontsize=8, color="#666666")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_horizon(rows, grid, out, U=10 * M):
    """U=10M at 300M vs 150M training tokens: the best delta moves, the best epoch count stays in the same band."""
    short = sorted((r["delta"], r["dom"], r["epochs"]) for r in rows
                   if r["name"].startswith("S_") and r["U"] == U and r["tokens"] == 150 * M and r["dropout"] == 0)
    full = sorted((d, st.mean(r["dom"] for r in rs), st.mean(r["epochs"] for r in rs))
                  for (u, d), rs in grid.items() if u == U and d > 0)
    if not short:
        return False
    fig, a = plt.subplots(figsize=(7.8, 4.8))
    a.axvspan(8, 15, color=NOISE, alpha=0.16, lw=0, zorder=0)
    for pts, color, marker, name, off in ((full, "#e08214", "o", "300M tokens (full budget)", (16, -14)),
                                          (short, "#54278f", "s", "150M tokens (half budget)", (-16, -20))):
        xs = [p[2] for p in pts]
        ys = [p[1] for p in pts]
        a.plot(xs, ys, marker=marker, color=color, lw=1.9, ms=7, mec="white", mew=0.9, label=name, zorder=3)
        i = int(np.argmin(ys))
        a.scatter([xs[i]], [ys[i]], s=210, facecolors="none", edgecolors=color, lw=1.8, zorder=4)
        a.annotate(f"best δ={pts[i][0]:g} ({pts[i][2]:.0f} ep)", (xs[i], ys[i]), textcoords="offset points",
                   xytext=off, ha="left" if off[0] > 0 else "right", fontsize=8.5, color=color, fontweight="semibold",
                   zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        for k, (d, y, ep) in enumerate(pts):
            if k == i or k not in (0, len(pts) - 1):
                continue  # the best point has its own label; the interior points are read off the curve
            a.annotate(f"δ={d:g}", (ep, y), textcoords="offset points", xytext=(0, 10), ha="center", va="bottom",
                       fontsize=7.5, color=color, alpha=0.9, zorder=6,
                       bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.8))
    a.set_xscale("log")
    a.set_xlim(0.7, 45)
    a.set_ylim(1.065, 1.25)
    a.set_xticks([1, 3, 5, 10, 15, 30])
    a.set_xticklabels(["1", "3", "5", "10", "15", "30"])
    a.set_xlabel("epochs over the domain data  (δ·B / U)")
    a.set_ylabel("PubMed test bits/byte")
    a.set_title("U = 10M: halving the token budget moves the best δ, not the best epoch count")
    a.legend(frameon=False, loc="upper right")
    a.grid(alpha=0.15, lw=0.6)
    fig.text(0.5, -0.02, "shaded = 8–15 epochs. The 150M grid has only 3 points, so its optimum is bracketed to ~10–14 epochs.",
             ha="center", va="top", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_heatmap(grid, base, out):
    """The same numbers as fig1, as a grid: gain in bpb over the best domain-only baseline for each (U, delta)."""
    Us = [U for U in U_ORDER if U in base and any(u == U for (u, d) in grid)]
    ds = [d for d in (0.03, 0.1, 0.3, 0.6, 1.0) if any(dd == d for (_, dd) in grid)]
    Z = np.full((len(Us), len(ds)), np.nan)
    E = np.full((len(Us), len(ds)), np.nan)
    for i, U in enumerate(Us):
        for j, d in enumerate(ds):
            if (U, d) in grid:
                Z[i, j] = base[U]["dom"] - st.mean(r["dom"] for r in grid[(U, d)])  # positive = mixing helps
                E[i, j] = st.mean(r["epochs"] for r in grid[(U, d)])
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    lim = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"{d:g}" for d in ds])
    ax.set_yticks(range(len(Us)))
    ax.set_yticklabels([lab(U) for U in Us])
    ax.set_xticks(np.arange(-0.5, len(ds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(Us), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.8)
    ax.tick_params(which="minor", length=0)
    for i in range(len(Us)):
        for j in range(len(ds)):
            if np.isnan(Z[i, j]):
                continue
            dark = abs(Z[i, j]) > 0.6 * lim
            ax.text(j, i - 0.14, f"{Z[i, j]:+.3f}", ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white" if dark else "#222222")
            ax.text(j, i + 0.24, f"{E[i, j]:.1f} ep", ha="center", va="center", fontsize=7.5,
                    color="white" if dark else "#444444")
        jb = int(np.nanargmax(Z[i]))
        ax.add_patch(plt.Rectangle((jb - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=2.0, zorder=5))
    ax.set_xlabel("domain share δ")
    ax.set_ylabel("unique domain tokens U")
    ax.set_title("Gain over the best domain-only baseline\nbold = Δ bpb (red = mixing helps), small = domain epochs, "
                 "boxed = best δ", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Δ PubMed bpb  (positive = mixing helps)", fontsize=9)
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
    fig_repetition({k: v for k, v in grid.items() if k[1] > 0}, os.path.join(a.out, "fig3_repetition.png"), sd)
    fig_heatmap({k: v for k, v in grid.items() if k[1] > 0}, base, os.path.join(a.out, "fig4_heatmap.png"))
    n_fig = 4
    if any(u in EXT_U for (u, d) in grid):
        e_fit, fit_pts, (e_lo, e_hi) = fit_epochs({k: v for k, v in grid.items() if k[1] > 0}, holdout=EXT_U, sd=sd)
        print("\nconstant-epochs fit (delta* = E*U/B, out-of-sample levels held out):")
        print(f"  E = {e_fit:.1f} epochs from " + ", ".join(f"U={lab(U)} (best delta={d:g})" for U, d in fit_pts))
        if np.isfinite(e_lo) and np.isfinite(e_hi):
            print(f"  epochs consistent with all of them at 2 sigma: E = {e_lo:.1f}-{e_hi:.1f}")
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
