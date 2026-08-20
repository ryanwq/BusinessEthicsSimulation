"""
Paper-ready trajectory figure for Stage 3C (innovation study).

Produces s3c_fig2_trajectories.png from the saved s3c_timeseries.csv.
Includes only No-Innovation and Fixed-Innovation conditions (excludes
CEI Ratcheting data even if it is present in the CSV).

Usage
-----
  python make_s3c_fig2.py                     # uses s3c_results/ by default
  python make_s3c_fig2.py --data s3c_results  # explicit path
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
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

INNOV_CONDITIONS = [(False, None), (True, "fixed")]
INNOV_LABEL = {(False, None): "NoI", (True, "fixed"): "Fixed"}
INNOV_LS    = {(False, None): "-",   (True, "fixed"): "--"}


# ── Figure ────────────────────────────────────────────────────────────────────

def make_paper_figure(data_dir, output_dir=None):
    """Generate s3c_fig2_trajectories.png.

    Parameters
    ----------
    data_dir  : directory containing s3c_timeseries.csv
    output_dir: where to save the PNG (defaults to data_dir)
    """
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "s3c_timeseries.csv")
    ts = pd.read_csv(csv_path, low_memory=False)

    ts['hi'] = ts['has_innovation'].astype(bool)
    ts['it'] = ts['innovation_type'].fillna('none')
    means = (ts.groupby(['mindset','strategy','hi','it','t'])[['omega_b','omega_e']]
               .mean().reset_index())

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                         'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'legend.fontsize': 11})
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey=True)

    for mi, ms in enumerate(MINDSETS):
        ax_b, ax_e = axes[mi, 0], axes[mi, 1]
        for si, st in enumerate(STRATEGIES):
            for (hi, it) in INNOV_CONDITIONS:
                it_key = 'none' if it is None else it
                sub = means[(means['mindset']==ms) & (means['strategy']==st)
                            & (means['hi']==hi)    & (means['it']==it_key)].sort_values('t')
                if sub.empty:
                    continue
                lbl = f"{STRAT_LABEL[si]} {INNOV_LABEL[(hi,it)]}" if mi == 0 else ""
                ls  = INNOV_LS[(hi, it)]
                ax_b.plot(sub['t'].values, sub['omega_b'].values,
                          color=STRAT_COLORS[si], ls=ls, lw=1.4, label=lbl)
                ax_e.plot(sub['t'].values, sub['omega_e'].values,
                          color=STRAT_COLORS[si], ls=ls, lw=1.4)

        ax_b.set_ylabel(MINDSET_FULL[ms], fontsize=12)
        ax_e.set_ylabel("")
        for col in range(2):
            axes[mi, col].grid(True, alpha=0.25)
            axes[mi, col].set_ylim(0.30, 0.44)

    axes[0, 0].set_title("Business performance", fontsize=13)
    axes[0, 1].set_title("Ethical performance",  fontsize=13)
    axes[-1, 0].set_xlabel("Step", fontsize=12)
    axes[-1, 1].set_xlabel("Step", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strategy", fontsize=11,
               loc="upper right", bbox_to_anchor=(0.99, 0.99))

    fig.tight_layout()
    out_path = os.path.join(output_dir, "s3c_fig2_trajectories.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate Stage 3C trajectory figure")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "s3c_results"),
                    help="Directory containing s3c_timeseries.csv")
    ap.add_argument("--output", default=None,
                    help="Output directory (defaults to --data)")
    args = ap.parse_args()
    make_paper_figure(data_dir=args.data, output_dir=args.output)
