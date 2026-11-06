"""
run_queue.py - run the whole experiment grid unattended, in priority order, skipping runs that already finished.

  python run_queue.py --lr 1e-3 --dry-run            # show the plan and a time estimate, run nothing
  python run_queue.py --lr 1e-3                      # run everything not yet done (Ctrl+C any time; re-run to resume)
  python run_queue.py --lr 1e-3 --max-priority 4     # only the most important runs
  python run_queue.py --summarize                    # (re)build runs/results.csv from all finished runs

A run counts as finished when runs/<name>/summary.json exists. A run interrupted half way simply restarts.
Stdout of each run goes to runs/<name>/stdout.txt. A failed run is reported and the queue moves on.

Experiment grid (small model, 300M training tokens per full run = the 45 minute budget):
  U    = unique domain tokens available (3M / 10M / 40M / all = abundant)
  delta = share of training sequences drawn from the domain corpus (the rest is general text)
  Priority 1-4   core: general-only, scarcity x mixture grid, noise seeds
  Priority 5     fair baselines at U=10M: domain-only with fewer epochs, at lr and 2*lr
  Priority 6-7   remaining scarcity levels and the naive full-budget domain-only runs
  Priority 8     fair baselines at U=3M and U=40M
  Priority 9     complete the U x delta grid at the 300M cap (7 missing cells)
  Priority 10-14 phase 2: dropout control, seeds on key comparisons, prospective test of the delta* rule (new
                 U=1M and U=20M), finer delta points, shorter horizon (150M tokens)
  Priority 15    level playing field: tune the domain-only baselines under dropout and give the mixtures dropout too
  Priority 16    close the gaps: seeds for U=10M, finish U=20M, U=1M refinements
  Priority 17-19 similarity extension (needs the pools): U=10M/delta 0.3, U=3M/delta 0.1, seed-1 replicate of the first
  Priority 20    optional double-horizon runs (600M tokens): exceed the cap, off by default (use --max-priority 20)
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

M = 1_000_000
BUDGET = 300 * M
TOK_PER_STEP = 32 * 256
FULL_TOK_PER_SEC = 130_000


def label(U):
    return "all" if U == 0 else f"{U // M}M"


def build_plan(lr, pools=("q1", "q2", "q3", "q4", "rnd")):
    """Returns a list of dicts: priority, name, U, delta, tokens, lr, seed, dropout."""
    plan = []

    def add(prio, name, U, delta, tokens, lr_, seed=1337, dropout=0.0, time_limit=45, size="small", pool=None):
        plan.append(dict(priority=prio, name=name, U=U, delta=delta, tokens=tokens, lr=lr_, seed=seed,
                         dropout=dropout, time_limit=time_limit, size=size, pool=pool))

    def mix(prio, U, delta, seed=1337, tokens=BUDGET, time_limit=45):
        tag = "" if tokens == BUDGET else f"_B{tokens // M}M"
        add(prio, f"S_U{label(U)}_d{delta}{tag}_s{seed}", U, delta, tokens, lr, seed, time_limit=time_limit)

    def baseline(prio, U, epochs):
        for tag, lr_ in (("lr1x", lr), ("lr2x", 2 * lr)):
            add(prio, f"C_U{label(U)}_ep{epochs}_{tag}", U, 1.0, epochs * U, lr_)

    # 1. general-only (no domain data at all)
    mix(1, 0, 0.0)
    # 2. U=10M across the mixture range
    for d in (0.3, 0.1, 0.6):
        mix(2, 10 * M, d)
    # 3. U=40M and the abundant-data domain-only reference. delta=0.3 is the central cell: if your tuning run at the
    #    final LR exists, rename its folder to S_U40M_d0.3_s1337 and the queue will treat it as done.
    for d in (0.1, 0.3, 0.6):
        mix(3, 40 * M, d)
    mix(3, 0, 1.0)
    # 4. noise estimate: two extra seeds of the central cell (seed 1337 = the tuning run)
    for s in (1, 2):
        mix(4, 40 * M, 0.3, seed=s)
    # 5. fair domain-only baselines at U=10M: fewer epochs, each with its own full LR schedule, at lr and 2*lr
    for ep in (3, 6, 12):
        baseline(5, 10 * M, ep)
    # 6. the heavily scarce level (3M): small domain shares only, 30 epochs at delta=0.3
    for d in (0.03, 0.1, 0.3):
        mix(6, 3 * M, d)
    # 7. naive full-budget domain-only runs and the abundant 30% mixture
    mix(7, 10 * M, 1.0)
    mix(7, 40 * M, 1.0)
    mix(7, 0, 0.3)
    # 8. fair baselines at the other scarcity levels
    for ep in (10, 30):
        baseline(8, 3 * M, ep)
    for ep in (2, 4):
        baseline(8, 40 * M, ep)

    # ---------------- phase 2: added after the first results (see the analysis) ----------------
    # 9. COMPLETE THE GRID: every U x delta cell at the 300M-token cap, including the cells expected to overfit,
    #    so no curve is cut short and nothing is hidden. (3M/0.6 and 3M/1.0 see 60 and 100 domain epochs.)
    for U, d in [(3 * M, 0.6), (3 * M, 1.0), (10 * M, 0.03), (40 * M, 0.03), (0, 0.03), (0, 0.1), (0, 0.6)]:
        mix(9, U, d)
    # 10. Is it just regularisation? Domain-only with dropout, at the best epoch count / LR found for each U.
    #     (These stay below the cap: a baseline is allowed to stop early.)
    for p in (0.1, 0.2, 0.3):
        add(10, f"C_U10M_ep12_lr2x_drop{p}", 10 * M, 1.0, 12 * 10 * M, 2 * lr, dropout=p)
        add(10, f"C_U3M_ep10_lr1x_drop{p}", 3 * M, 1.0, 10 * 3 * M, lr, dropout=p)
    # 11. Seeds on the comparisons that decide the conclusions (mixture vs domain-only at 40M; best delta at 10M and 3M)
    for U, d in [(40 * M, 0.6), (40 * M, 1.0), (10 * M, 0.3), (10 * M, 0.6), (3 * M, 0.1)]:
        for sd in (1, 2):
            mix(11, U, d, seed=sd)
    # 12. Prospective test of the rule delta* ~ min(1, 10*U/B) (fitted AFTER seeing the grid, so it needs new data):
    #     U=1M predicts delta*=0.03 (9 epochs; 0.01 -> 3 epochs, 0.1 -> 30 epochs); U=20M predicts delta*=0.67.
    #     These are a separate validation set, not part of the main grid.
    for d in (0.01, 0.03, 0.1):
        mix(12, 1 * M, d)
    baseline(12, 1 * M, 10)
    for d in (0.3, 0.67):
        mix(12, 20 * M, d)
    # 13. Finer delta where the optimum is sharp (extra points on the U=3M and U=10M curves)
    for d in (0.05, 0.15, 0.2):
        mix(13, 3 * M, d)
    for d in (0.2, 0.45):
        mix(13, 10 * M, d)
    # 14. Shorter horizon at U=10M (stays under the cap). Rule predicts delta*=0.67 at 150M tokens.
    for d in (0.3, 0.67, 1.0):
        mix(14, 10 * M, d, tokens=150 * M)
    # 15. LEVEL PLAYING FIELD after the dropout result: dropout 0.1 improved the domain-only baselines by 0.025-0.03 bpb, so
    #     (a) tune the baselines further (more epochs under dropout, a finer dropout value, dropout at U=40M) and
    #     (b) give the mixtures the same help, so the final comparison is best-regularised vs best-regularised.
    for ep in (6, 24, 30):
        add(15, f"C_U10M_ep{ep}_lr2x_drop0.1", 10 * M, 1.0, ep * 10 * M, 2 * lr, dropout=0.1)
    add(15, "C_U10M_ep12_lr2x_drop0.15", 10 * M, 1.0, 12 * 10 * M, 2 * lr, dropout=0.15)
    for ep in (20, 30):
        add(15, f"C_U3M_ep{ep}_lr1x_drop0.1", 3 * M, 1.0, ep * 3 * M, lr, dropout=0.1)
    add(15, "C_U3M_ep10_lr1x_drop0.15", 3 * M, 1.0, 10 * 3 * M, lr, dropout=0.15)
    add(15, "C_U40M_ep7.5_lr1x_drop0.1", 40 * M, 1.0, 300 * M, lr, dropout=0.1)
    for U, d in [(10 * M, 0.3), (10 * M, 0.6), (3 * M, 0.1), (3 * M, 0.3)]:
        add(15, f"S_U{label(U)}_d{d}_drop0.1_s1337", U, d, BUDGET, lr, dropout=0.1)
    # ---------------- after the 90-run results (see the analysis) ----------------
    # 16. Close the gaps in the main story.
    #   (a) U=10M: the mixture's edge over the best regularised baseline is only ~0.02 and rests on single seeds
    for sd in (1, 2):
        mix(16, 10 * M, 0.45, seed=sd)
        add(16, f"C_U10M_ep30_lr2x_drop0.1_s{sd}", 10 * M, 1.0, 300 * M, 2 * lr, seed=sd, dropout=0.1)
    #   (b) U=20M: only delta 0.3 and 0.67 exist; add delta 0.5 and 1.0 plus domain-only baselines to finish this level
    mix(16, 20 * M, 1.0)
    mix(16, 20 * M, 0.5)
    add(16, "C_U20M_ep10_lr1x", 20 * M, 1.0, 10 * 20 * M, lr)
    add(16, "C_U20M_ep15_lr1x_drop0.1", 20 * M, 1.0, 300 * M, lr, dropout=0.1)
    #   (c) U=1M: delta 0.05 (15 epochs, between the tied 0.03 and 0.1) and the dropout variant of the 30-epoch cell
    mix(16, 1 * M, 0.05)
    add(16, "S_U1M_d0.1_drop0.1_s1337", 1 * M, 0.1, BUDGET, lr, dropout=0.1)
    # 17-19. SIMILARITY EXTENSION: does the benefit of general data depend on how PubMed-like it is?
    #   Same budget, same U and delta; only the general pool changes (equal-token pools q1 = most PubMed-like ... q4 =
    #   least, and a random pool of the same size as the quantity-matched control). Needs data/fw_pools/<pool>/ built by
    #   build_pools.py; runs whose pool folder is missing are skipped, not failed.
    for prio, U, d, sd in [(17, 10 * M, 0.3, 1337), (18, 3 * M, 0.1, 1337), (19, 10 * M, 0.3, 1)]:
        for pl in pools:
            add(prio, f"P_U{label(U)}_d{d}_{pl}_s{sd}", U, d, BUDGET, lr, seed=sd, pool=pl)
    # 21. OPTIONAL follow-up if the quartile result shows a signal: top vs bottom DECILE vs a random decile (~86M tokens each,
    #     so the general stream is repeated ~2.4x for every pool alike). Build with: build_pools.py --k 10 --prefix d --write d1 d10 drnd
    for pl in ("d1", "d10", "drnd"):
        add(21, f"P_U10M_d0.3_{pl}_s1337", 10 * M, 0.3, BUDGET, lr, seed=1337, pool=pl)
    # 20. OPTIONAL, excluded by --max-priority 14: double horizon. These exceed the 45-minute / 300M-token cap on
    #     purpose (about 80 min each) and would have to be reported as a separate, clearly labelled extension.
    for d in (0.1, 0.17, 0.3):
        mix(20, 10 * M, d, tokens=600 * M, time_limit=0)
    return plan


def eval_interval(tokens):
    steps = tokens // TOK_PER_STEP
    return 2000 if tokens >= 200 * M else max(100, int(round(steps / 20 / 100)) * 100)


def command(run, a):
    gdir = os.path.join(a.pools_dir, run["pool"]) if run.get("pool") else a.general_dir
    return [sys.executable, a.train_script,
            "--domain-dir", a.domain_dir, "--general-dir", gdir,
            "--domain-frac", str(run["delta"]), "--domain-unique-tokens", str(run["U"]),
            "--tokens", str(run["tokens"]), "--lr", repr(run["lr"]), "--seed", str(run["seed"]),
            "--dropout", str(run["dropout"]), "--eval-interval", str(eval_interval(run["tokens"])),
            "--time-limit-min", str(run["time_limit"]), "--size", run["size"], "--name", run["name"],
            "--out-dir", a.out_dir]


def est_minutes(run):
    steps = run["tokens"] // TOK_PER_STEP
    return run["tokens"] / FULL_TOK_PER_SEC / 60 + steps / eval_interval(run["tokens"]) * 8.4 / 60 + 0.3


def summarize(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        a = s["args"]
        rows.append({
            "name": os.path.basename(os.path.dirname(path)), "size": a["size"], "U": a["domain_unique_tokens"],
            "delta": a["domain_frac"], "tokens": a["tokens"], "lr": a["lr"], "dropout": a["dropout"], "seed": a["seed"],
            "domain_epochs": round(s.get("domain_epochs", 0), 2), "best_step": s.get("best_step"),
            "minutes": round(s.get("total_minutes", 0), 1),
            "best_val_domain_bpb": s.get("best_val_domain_bpb"), "best_test_domain_bpb": s.get("best_test_domain_bpb"),
            "final_test_domain_bpb": s.get("final_test_domain_bpb"),
            "best_test_general_bpb": s.get("best_test_general_bpb"), "final_test_general_bpb": s.get("final_test_general_bpb"),
            "truncated": s.get("truncated_by_time"), "diverged": s.get("diverged")})
    if not rows:
        print("no finished runs found")
        return
    out = os.path.join(out_dir, "results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'name':<26}{'U':>6}{'delta':>7}{'ep':>6}{'best':>7}{'test dom':>10}{'test gen':>10}{'min':>6}")
    for r in rows:
        td, tg = r["best_test_domain_bpb"], r["best_test_general_bpb"]
        flag = " TRUNCATED" if r["truncated"] else (" DIVERGED" if r["diverged"] else "")
        print(f"{r['name']:<26}{label(r['U']):>6}{r['delta']:>7}{r['domain_epochs']:>6}{r['best_step']:>7}"
              f"{(f'{td:.4f}' if td else '-'):>10}{(f'{tg:.4f}' if tg else '-'):>10}{r['minutes']:>6}{flag}")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr", type=float, help="peak LR chosen for 300M-token runs (required unless --summarize)")
    ap.add_argument("--max-priority", type=int, default=99)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--domain-dir", default="data/pubmed_v2")
    ap.add_argument("--general-dir", default="data/fineweb_v1")
    ap.add_argument("--pools-dir", default="data/fw_pools", help="folder with the similarity pools (build_pools.py)")
    ap.add_argument("--only", nargs="*", default=[], help="run only runs whose name contains one of these strings")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--train-script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    a = ap.parse_args()

    if a.summarize:
        summarize(a.out_dir)
        return
    if a.lr is None:
        raise SystemExit("--lr is required (pick it from the learning-rate checks first)")

    plan = [r for r in build_plan(a.lr) if r["priority"] <= a.max_priority and (not a.only or any(o in r["name"] for o in a.only))]
    done = [r for r in plan if os.path.exists(os.path.join(a.out_dir, r["name"], "summary.json"))]
    todo = [r for r in plan if r not in done]
    hours = sum(est_minutes(r) for r in todo) / 60
    print(f"{len(plan)} runs planned, {len(done)} already finished, {len(todo)} to run (~{hours:.1f} GPU-hours)")
    print(f"{'prio':<5}{'name':<26}{'U':>6}{'delta':>7}{'tokens':>8}{'lr':>9}{'est min':>9}  status")
    for r in plan:
        st = "done" if r in done else "todo"
        print(f"{r['priority']:<5}{r['name']:<26}{label(r['U']):>6}{r['delta']:>7}{r['tokens'] // M:>7}M{r['lr']:>9.0e}{est_minutes(r):>9.0f}  {st}")
    if a.dry_run:
        return

    t_start = time.time()
    for i, r in enumerate(todo, 1):
        if r.get("pool") and not os.path.exists(os.path.join(a.pools_dir, r["pool"], "train.bin")):
            print(f"\n[{i}/{len(todo)}] {r['name']}  SKIPPED: pool folder {os.path.join(a.pools_dir, r['pool'])} not built yet", flush=True)
            continue
        os.makedirs(os.path.join(a.out_dir, r["name"]), exist_ok=True)
        log_path = os.path.join(a.out_dir, r["name"], "stdout.txt")
        print(f"\n[{i}/{len(todo)}] {r['name']}  (~{est_minutes(r):.0f} min)  started {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            rc = subprocess.run(command(r, a), stdout=log, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        sp = os.path.join(a.out_dir, r["name"], "summary.json")
        if rc == 0 and os.path.exists(sp):
            with open(sp) as f:
                s = json.load(f)
            print(f"    done in {mins:.1f} min | test domain bpb {s.get('best_test_domain_bpb', float('nan')):.4f}"
                  f" | test general bpb {s.get('best_test_general_bpb', float('nan')):.4f}"
                  + (" | TRUNCATED BY TIME" if s.get("truncated_by_time") else "")
                  + (" | DIVERGED" if s.get("diverged") else ""), flush=True)
        else:
            print(f"    FAILED (exit code {rc}) after {mins:.1f} min - see {log_path}", flush=True)
    print(f"\nqueue finished in {(time.time() - t_start) / 3600:.1f} h")
    summarize(a.out_dir)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
