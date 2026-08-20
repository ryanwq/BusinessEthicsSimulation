"""
Paper-ready trajectory figure for Stage 2B (mindset comparison).

Produces s2b_fig2_trajectories.png from the saved s2b_timeseries.csv.
The R0 goal-blind reference is re-run on first call and cached to CSV
so subsequent calls are fast.

Usage
-----
  python make_s2b_fig2.py                     # uses s2b_results/ by default
  python make_s2b_fig2.py --data s2b_results  # explicit path
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1b_simulation import run_replication_s1b

# ── Constants ─────────────────────────────────────────────────────────────────
N, G, G_b, M, R, T = 24, 8, 4, 4, 120, 250
N_CONFLICT, ALPHA   = 12, 0.50

MINDSETS   = ["wac", "sf", "compliance", "cei"]
STRATEGIES = ["baseline", "myopic", "spatial", "temporal"]

MINDSET_FULL = {
    "wac":        "Win At Any Cost",
    "sf":         "Survival First",
    "compliance": "Compliance",
    "cei":        "Continuous Ethical Improvement",
}
STRAT_LABEL  = ["Baseline", "Myopic", "Spatial diff.", "Temporal diff."]
STRAT_COLORS = ["#185FA5", "#D85A30", "#3B6D11", "#534AB7"]


# ── R0 reference (goal-blind baseline, cached after first run) ────────────────

def _compute_r0(reps=250):
    r0_b = np.zeros((reps, T))
    r0_e = np.zeros((reps, T))
    for rep in range(reps):
        seed = 999_000 + rep
        perf, b_idx, e_idx = run_replication_s1b(
            N, G, G_b, M, R, T, "baseline", N_CONFLICT, ALPHA, seed
        )
        r0_b[rep] = perf[:, b_idx].mean(axis=1)
        r0_e[rep] = perf[:, e_idx].mean(axis=1)
    return r0_b.mean(axis=0), r0_e.mean(axis=0)


def _load_or_compute_r0(data_dir, reps=250):
    cache = os.path.join(data_dir, "s2b_r0_reference.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        return df['omega_b'].values, df['omega_e'].values
    print("  Running R0 reference (250 reps, ~10 s)...")
    r0_mb, r0_me = _compute_r0(reps)
    pd.DataFrame({'t': np.arange(T), 'omega_b': r0_mb, 'omega_e': r0_me}).to_csv(cache, index=False)
    print("  R0 done — cached to", cache)
    return r0_mb, r0_me


# ── Figure ────────────────────────────────────────────────────────────────────

def make_paper_figure(data_dir, output_dir=None):
    """Generate s2b_fig2_trajectories.png.

    Parameters
    ----------
    data_dir  : directory containing s2b_timeseries.csv
    output_dir: where to save the PNG (defaults to data_dir)
    """
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "s2b_timeseries.csv")
    df = pd.read_csv(csv_path)
    r0_mb, r0_me = _load_or_compute_r0(data_dir)

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                         'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'legend.fontsize': 11})
    steps = np.arange(T)
    fig, axes = plt.subplots(len(MINDSETS), 2,
                             figsize=(11, 2.8 * len(MINDSETS)),
                             sharex=True, sharey=True)

    for mi, ms in enumerate(MINDSETS):
        for si, st in enumerate(STRATEGIES):
            sub = df[(df['mindset'] == ms) & (df['strategy'] == st)].sort_values('t')
            mb  = sub['omega_b'].values
            me  = sub['omega_e'].values
            lbl = STRAT_LABEL[si] if mi == 0 else ""
            axes[mi, 0].plot(steps, mb, color=STRAT_COLORS[si], linewidth=1.3, label=lbl)
            axes[mi, 1].plot(steps, me, color=STRAT_COLORS[si], linewidth=1.3)

        r0_lbl = "R0 (Baseline, goal-blind)" if mi == 0 else ""
        axes[mi, 0].plot(steps, r0_mb, "k--", linewidth=0.9, alpha=0.5, label=r0_lbl)
        axes[mi, 1].plot(steps, r0_me, "k--", linewidth=0.9, alpha=0.5)

        axes[mi, 0].set_ylabel(MINDSET_FULL[ms], fontsize=12)
        axes[mi, 1].set_ylabel("")
        for col in range(2):
            axes[mi, col].grid(True, alpha=0.25)

    axes[0, 0].set_title("Business performance", fontsize=13)
    axes[0, 1].set_title("Ethical performance",  fontsize=13)
    axes[-1, 0].set_xlabel("Time step", fontsize=12)
    axes[-1, 1].set_xlabel("Time step", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strategy", fontsize=11,
               loc="upper right", bbox_to_anchor=(0.99, 0.96))

    fig.tight_layout()
    out_path = os.path.join(output_dir, "s2b_fig2_trajectories.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate Stage 2B trajectory figure")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "s2b_results"),
                    help="Directory containing s2b_timeseries.csv")
    ap.add_argument("--output", default=None,
                    help="Output directory (defaults to --data)")
    args = ap.parse_args()
    make_paper_figure(data_dir=args.data, output_dir=args.output)
