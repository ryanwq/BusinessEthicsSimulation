"""
Ratcheting innovation experiment — all four mindsets × four strategies.

Extends Stage 3C by applying the ratcheting aspiration mechanism to WAC, SF,
and Compliance (not just CEI). Runs T=3000 steps to reveal long-run plateaus.

Usage
-----
  python ratcheting_experiment.py              # 30 reps (pilot)
  python ratcheting_experiment.py --reps 250   # publication run
  python ratcheting_experiment.py --reps 250 --output my_results
"""
import argparse
import os
import csv
import time
import concurrent.futures
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from s3c_simulation import run_replication_s3c

# ── Parameters ────────────────────────────────────────────────────────────────
N, G, G_b, M, R = 24, 8, 4, 4, 120
T               = 3000
N_CONFLICT      = 12
ALPHA           = 0.50
BETA            = 0.10
SAMPLE_EVERY    = 20
OUTPUT_DIR      = "ratcheting_results"

MINDSETS   = ["wac", "sf", "compliance", "cei"]
STRATEGIES = ["baseline", "myopic", "spatial", "temporal"]

MINDSET_FULL = {
    "wac":        "Win At Any Cost",
    "sf":         "Survival First",
    "compliance": "Compliance",
    "cei":        "Continuous Ethical Improvement",
}
STRAT_LABELS = {
    "baseline": "Baseline",
    "spatial":  "Spatial",
    "myopic":   "Myopic",
    "temporal": "Temporal",
}
STRAT_COLORS = {
    "baseline": "#1A1A2E",
    "spatial":  "#E74C3C",
    "myopic":   "#F39C12",
    "temporal": "#2980B9",
}


# ── Single replication ────────────────────────────────────────────────────────

def _run_one(args):
    mindset, strategy, rep, seed = args
    (perf, n_t, biz_idx, eth_idx,
     theta_g, A_g_init, A_g_final,
     acc, innov_cnt, casc_cnt, innov_steps,
     above_all_count, above_goal_counts, ratchet_cnt) = run_replication_s3c(
        N, G, G_b, M, R, T, strategy, N_CONFLICT, ALPHA,
        mindset, True, "ratcheting", seed, BETA
    )
    omega_b = perf[:, biz_idx].mean(axis=1)
    omega_e = perf[:, eth_idx].mean(axis=1)
    return dict(
        mindset=mindset, strategy=strategy, rep=rep,
        t_steps=list(range(0, T, SAMPLE_EVERY)),
        omega_b_series=omega_b[::SAMPLE_EVERY].tolist(),
        omega_e_series=omega_e[::SAMPLE_EVERY].tolist(),
        n_t_series=n_t[::SAMPLE_EVERY].tolist(),
        final_omega_b=float(omega_b[-1]),
        final_omega_e=float(omega_e[-1]),
        n_final=int(n_t[-1]),
        innov_count=innov_cnt,
        ratchet_count=ratchet_cnt,
    )


# ── Main run ──────────────────────────────────────────────────────────────────

def main(reps=30, output_dir=OUTPUT_DIR, base_seed=700_000):
    os.makedirs(output_dir, exist_ok=True)
    conditions = [(ms, st) for ms in MINDSETS for st in STRATEGIES]
    tasks = [
        (ms, st, rep, base_seed + ci * reps + rep)
        for ci, (ms, st) in enumerate(conditions)
        for rep in range(reps)
    ]

    n_cells = len(conditions)
    print(f"\nRatcheting experiment — {reps} reps × {n_cells} conditions = {len(tasks):,} total")
    print(f"  T={T} steps, N={N}, G={G}, α={ALPHA}, β={BETA}")
    t0 = time.time()

    all_results = []
    n_workers = min(10, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_one, task): task for task in tasks}
        done = 0
        tick = max(1, len(tasks) // 10)
        for fut in concurrent.futures.as_completed(futures):
            all_results.append(fut.result())
            done += 1
            if done % tick == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)}  ({elapsed:.0f}s elapsed, "
                      f"ETA {(len(tasks)-done)/done*elapsed:.0f}s)")

    print(f"  Done in {time.time()-t0:.0f}s")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pr_path = os.path.join(output_dir, "ratcheting_per_rep.csv")
    pr_fields = ['mindset', 'strategy', 'rep', 'final_omega_b', 'final_omega_e',
                 'n_final', 'innov_count', 'ratchet_count']
    with open(pr_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=pr_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print(f"  Per-rep CSV: {pr_path}")

    ts_rows = []
    for r in all_results:
        for i, t in enumerate(r['t_steps']):
            ts_rows.append({'mindset': r['mindset'], 'strategy': r['strategy'],
                            'rep': r['rep'], 't': t,
                            'omega_b': r['omega_b_series'][i],
                            'omega_e': r['omega_e_series'][i],
                            'n_t':     r['n_t_series'][i]})
    ts_path = os.path.join(output_dir, "ratcheting_timeseries.csv")
    with open(ts_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mindset','strategy','rep','t','omega_b','omega_e','n_t'])
        w.writeheader()
        w.writerows(ts_rows)
    print(f"  Timeseries CSV: {ts_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'':30s}  {'omega_b':>8}  {'omega_e':>8}  {'N_t':>7}  {'ratchets':>9}")
    pr_df = pd.DataFrame(all_results)
    for ms in MINDSETS:
        for st in STRATEGIES:
            row = pr_df[(pr_df['mindset']==ms) & (pr_df['strategy']==st)]
            if row.empty:
                continue
            print(f"  {ms:12s} {st:12s}  "
                  f"{row['final_omega_b'].mean():8.4f}  "
                  f"{row['final_omega_e'].mean():8.4f}  "
                  f"{row['n_final'].mean():7.1f}  "
                  f"{row['ratchet_count'].mean():9.1f}")
        print()

    # ── Figure ────────────────────────────────────────────────────────────────
    make_figure(ts_path, output_dir)
    print(f"\nAll outputs saved to: {output_dir}/")


def make_figure(ts_path, output_dir):
    df = pd.read_csv(ts_path)

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                         'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'legend.fontsize': 11})

    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True, sharey=True)

    for row, ms in enumerate(MINDSETS):
        ax_b, ax_e = axes[row, 0], axes[row, 1]
        for st in STRATEGIES:
            sub = (df[(df['mindset']==ms) & (df['strategy']==st)]
                   .groupby('t')[['omega_b','omega_e']].mean())
            t_vals = sub.index.values
            lbl = STRAT_LABELS[st] if row == 0 else ""
            ax_b.plot(t_vals, sub['omega_b'].values,
                      color=STRAT_COLORS[st], linewidth=1.5, label=lbl)
            ax_e.plot(t_vals, sub['omega_e'].values,
                      color=STRAT_COLORS[st], linewidth=1.5)

        ax_b.set_ylabel(MINDSET_FULL[ms], fontsize=12)
        ax_e.set_ylabel("")
        for col in range(2):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].set_ylim(0.25, 0.55)

    axes[0, 0].set_title("Business performance", fontsize=13)
    axes[0, 1].set_title("Ethical performance",  fontsize=13)
    axes[-1, 0].set_xlabel("Time step", fontsize=12)
    axes[-1, 1].set_xlabel("Time step", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strategy", fontsize=11,
               loc="upper right", bbox_to_anchor=(0.99, 0.99))

    fig.tight_layout()
    path = os.path.join(output_dir, "ratcheting_trajectories.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ratcheting innovation — all mindsets × all strategies")
    ap.add_argument("--reps",   type=int, default=30,
                    help="Replications per cell (default 30)")
    ap.add_argument("--output", default=OUTPUT_DIR,
                    help="Output directory")
    args = ap.parse_args()
    main(reps=args.reps, output_dir=args.output)
