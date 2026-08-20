"""
Stage 1B experiment: decision-level partition model sensitivity analysis.

9 conditions (3 conflict fractions × 3 intensities) × 4 strategies × R reps.

Usage
-----
    python3 s1b_experiment.py              # 30-rep pilot
    python3 s1b_experiment.py --reps 250   # full run
    python3 s1b_experiment.py --output s1b_results --reps 250

Outputs
-------
    <output>/s1b_per_rep.csv        One row per replication.
    <output>/s1b_timeseries.csv     Mean Omega_b/e trajectories.
    <output>/s1b_fig1_frontier.png  Business-ethics frontier.
    <output>/s1b_fig2_trajectories.png  Omega_b/e over time.
    <output>/s1b_fig3_diff.png      Omega_b - Omega_e boxplots.
    <output>/s1b_fig4_performance.png   Final Omega_b and Omega_e bar chart.
    <output>/s1b_fig5_correlations.png  Empirical correlation boxplots vs Stage 1.
"""
import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from s1b_simulation import run_replication_s1b

# ── parameters ────────────────────────────────────────────────────────────────
N, G, G_b, M, R, T = 24, 8, 4, 4, 120, 250

STRATEGIES  = ["baseline", "myopic", "spatial", "temporal"]
STRAT_LABEL = ["Baseline", "Myopic", "Spatial diff.", "Temporal diff."]
COLORS_S    = ["#185FA5", "#D85A30", "#3B6D11", "#534AB7"]

# 9 sensitivity conditions: (n_conflict, alpha, label)
CONDITIONS = [
    ( 6, 0.25, "D1\nf=.25\nα=.25"),
    ( 6, 0.50, "D2\nf=.25\nα=.50"),
    ( 6, 1.00, "D3\nf=.25\nα=1.0"),
    (12, 0.25, "D4\nf=.50\nα=.25"),
    (12, 0.50, "D5\nf=.50\nα=.50"),
    (12, 1.00, "D6\nf=.50\nα=1.0"),
    (18, 0.25, "D7\nf=.75\nα=.25"),
    (18, 0.50, "D8\nf=.75\nα=.50"),
    (18, 1.00, "D9\nf=.75\nα=1.0"),
]
COND_KEYS   = [f"D{i+1}" for i in range(9)]
COND_LABELS = [c[2] for c in CONDITIONS]
COND_SHORT  = COND_KEYS

# Conflict-fraction groups (for visual grouping in figures)
# D1-D3: f=0.25, D4-D6: f=0.50, D7-D9: f=0.75
FRAC_COLORS = ["#BDD7EE", "#6BAED6", "#2171B5",   # low conflict (light→dark blue)
               "#C7E9C0", "#74C476", "#238B45",   # medium conflict (light→dark green)
               "#FCBBA1", "#FB6A4A", "#CB181D"]   # high conflict (light→dark red)


def pearson_r(series_a, series_b):
    """First-differenced Pearson r between two 1-D series."""
    da = np.diff(series_a)
    db = np.diff(series_b)
    if da.std() < 1e-12 or db.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def run_condition(strategy, n_conflict, alpha, reps, base_seed):
    """
    Run `reps` replications for one (strategy, condition) cell.

    Returns dict with:
      omega_b       : (reps, T)
      omega_e       : (reps, T)
      corr_bb_reps  : (reps,) — within-business correlation per rep
      corr_ee_reps  : (reps,) — within-ethical correlation per rep
      corr_be_reps  : (reps,) — cross-type correlation per rep
    """
    omega_b = np.zeros((reps, T))
    omega_e = np.zeros((reps, T))
    corr_bb = np.zeros(reps)
    corr_ee = np.zeros(reps)
    corr_be = np.zeros(reps)

    bb_pairs = list(combinations(range(G_b), 2))
    be_pairs = [(bi, ei) for bi in range(G_b) for ei in range(G - G_b)]
    ee_pairs = list(combinations(range(G - G_b), 2))

    for rep in range(reps):
        seed = base_seed + rep
        perf, b_idx, e_idx = run_replication_s1b(
            N, G, G_b, M, R, T, strategy, n_conflict, alpha, seed
        )

        pb = perf[:, b_idx]   # (T, G_b)
        pe = perf[:, e_idx]   # (T, G_e)
        omega_b[rep] = pb.mean(axis=1)
        omega_e[rep] = pe.mean(axis=1)

        corr_bb[rep] = (np.mean([pearson_r(pb[:, i], pb[:, j]) for i, j in bb_pairs])
                        if bb_pairs else 0.0)
        corr_ee[rep] = (np.mean([pearson_r(pe[:, i], pe[:, j]) for i, j in ee_pairs])
                        if ee_pairs else 0.0)
        corr_be[rep] = np.mean([pearson_r(pb[:, bi], pe[:, ei]) for bi, ei in be_pairs])

    return {
        "omega_b":      omega_b,
        "omega_e":      omega_e,
        "corr_bb_reps": corr_bb,
        "corr_ee_reps": corr_ee,
        "corr_be_reps": corr_be,
    }


def main(reps=30, output_dir="s1b_pilot", base_seed=0):
    os.makedirs(output_dir, exist_ok=True)
    total = len(STRATEGIES) * len(CONDITIONS)
    print(f"Stage 1B: 4 strategies × 9 conditions × {reps} reps = {total * reps} replications")

    results = {}   # (strategy, cond_idx) -> dict
    per_rep_rows = []
    done = 0

    for si, strategy in enumerate(STRATEGIES):
        for ci, (n_conflict, alpha, _) in enumerate(CONDITIONS):
            done += 1
            seed_off = (si * len(CONDITIONS) + ci) * reps
            print(f"  [{done:2d}/{total}] {STRAT_LABEL[si]:16s} × {COND_KEYS[ci]} "
                  f"(n_conflict={n_conflict}, α={alpha:.2f}) ...", end=" ", flush=True)

            res = run_condition(strategy, n_conflict, alpha, reps, base_seed + seed_off)
            results[(si, ci)] = res

            final_b = res["omega_b"][:, -1]
            final_e = res["omega_e"][:, -1]
            diff    = final_b - final_e
            print(f"Ω_b={final_b.mean():.3f}  Ω_e={final_e.mean():.3f}  "
                  f"Δ={diff.mean():+.3f}  "
                  f"r_be={res['corr_be_reps'].mean():.3f}")

            for rep in range(reps):
                per_rep_rows.append({
                    "strategy":   strategy,
                    "cond":       COND_KEYS[ci],
                    "n_conflict": n_conflict,
                    "alpha":      alpha,
                    "rep":        rep,
                    "final_omega_b": float(res["omega_b"][rep, -1]),
                    "final_omega_e": float(res["omega_e"][rep, -1]),
                    "diff_be":       float(res["omega_b"][rep, -1] - res["omega_e"][rep, -1]),
                    "corr_bb":    float(res["corr_bb_reps"][rep]),
                    "corr_ee":    float(res["corr_ee_reps"][rep]),
                    "corr_be":    float(res["corr_be_reps"][rep]),
                })

    # ── CSVs ─────────────────────────────────────────────────────────────────
    per_rep_path = os.path.join(output_dir, "s1b_per_rep.csv")
    with open(per_rep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_rep_rows[0].keys())
        w.writeheader()
        w.writerows(per_rep_rows)
    print(f"\nPer-rep data: {per_rep_path}")

    ts_rows = []
    for si, strategy in enumerate(STRATEGIES):
        for ci in range(len(CONDITIONS)):
            res = results[(si, ci)]
            mb  = res["omega_b"].mean(axis=0)
            me  = res["omega_e"].mean(axis=0)
            for t in range(T):
                ts_rows.append({
                    "strategy": strategy,
                    "cond":     COND_KEYS[ci],
                    "t":        t,
                    "omega_b":  float(mb[t]),
                    "omega_e":  float(me[t]),
                })
    ts_path = os.path.join(output_dir, "s1b_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ts_rows[0].keys())
        w.writeheader()
        w.writerows(ts_rows)
    print(f"Time-series data: {ts_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    make_fig1(results, output_dir)
    make_fig2(results, output_dir)
    make_fig3(results, output_dir)
    make_fig4(results, output_dir)
    make_fig5(results, output_dir)
    print(f"\nAll figures saved to {output_dir}/")


# ── Figure 1: Business-ethics frontier ────────────────────────────────────────

def make_fig1(results, output_dir):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(14, 4), sharey=True, sharex=True)

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        for ci in range(len(CONDITIONS)):
            res   = results[(si, ci)]
            fb    = res["omega_b"][:, -1].mean()
            fe    = res["omega_e"][:, -1].mean()
            color = FRAC_COLORS[ci]
            ax.scatter(fb, fe, color=color, s=60, zorder=3)
            ax.annotate(COND_KEYS[ci], (fb, fe), textcoords="offset points",
                        xytext=(4, 3), fontsize=7)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.4, label="Ω_b = Ω_e")
        ax.set_xlabel("Final Ω_b", fontsize=9)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Final Ω_e", fontsize=9)

    # Legend for conflict fraction
    from matplotlib.patches import Patch
    patches = [
        Patch(color=FRAC_COLORS[0], label="f = 0.25 (D1–D3)"),
        Patch(color=FRAC_COLORS[3], label="f = 0.50 (D4–D6)"),
        Patch(color=FRAC_COLORS[6], label="f = 0.75 (D7–D9)"),
    ]
    fig.legend(handles=patches, fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stage 1B: Business-ethics performance frontier by strategy",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s1b_fig1_frontier.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 saved.")


# ── Figure 2: Trajectories for selected conditions ─────────────────────────────

def make_fig2(results, output_dir):
    # Show D1, D5, D9 across all 4 strategies — 4 rows × 2 cols (Ω_b, Ω_e)
    highlight = [0, 4, 8]   # D1, D5, D9 indices
    hl_colors = ["#2171B5", "#238B45", "#CB181D"]
    hl_labels = ["D1 (low f, mild α)", "D5 (med f, mod α)", "D9 (high f, strong α)"]
    steps = np.arange(T)

    fig, axes = plt.subplots(len(STRATEGIES), 2,
                             figsize=(10, 2.8 * len(STRATEGIES)), sharex=True)

    for si in range(len(STRATEGIES)):
        for idx, (ci, color, label) in enumerate(zip(highlight, hl_colors, hl_labels)):
            res = results[(si, ci)]
            mb  = res["omega_b"].mean(axis=0)
            me  = res["omega_e"].mean(axis=0)
            axes[si, 0].plot(steps, mb, color=color, linewidth=1.4,
                             label=label if si == 0 else "")
            axes[si, 1].plot(steps, me, color=color, linewidth=1.4)

        axes[si, 0].set_ylabel(f"{STRAT_LABEL[si]}\nΩ_b", fontsize=8)
        axes[si, 1].set_ylabel("Ω_e", fontsize=8)
        for col in range(2):
            axes[si, col].grid(True, alpha=0.25)

    axes[0, 0].set_title("Business performance (Ω_b)", fontsize=9)
    axes[0, 1].set_title("Ethical performance (Ω_e)", fontsize=9)
    axes[-1, 0].set_xlabel("Time step", fontsize=9)
    axes[-1, 1].set_xlabel("Time step", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Condition", loc="center right",
               fontsize=8, bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Stage 1B: Performance trajectories (D1, D5, D9)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s1b_fig2_trajectories.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 2 saved.")


# ── Figure 3: Ω_b − Ω_e boxplots across all conditions ───────────────────────

def make_fig3(results, output_dir):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(14, 4), sharey=True)

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        positions = np.arange(len(CONDITIONS))
        data = [results[(si, ci)]["omega_b"][:, -1] - results[(si, ci)]["omega_e"][:, -1]
                for ci in range(len(CONDITIONS))]
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], FRAC_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(COND_SHORT, fontsize=7)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    axes[0].set_ylabel("Ω_b − Ω_e (final step)", fontsize=9)
    fig.suptitle("Stage 1B: Spontaneous business-ethics performance gap\n"
                 "(positive = business advantage, negative = ethical advantage)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s1b_fig3_diff.png"), dpi=150)
    plt.close(fig)
    print("Fig 3 saved.")


# ── Figure 4: Final Ω_b and Ω_e bar chart, all conditions ─────────────────────

def make_fig4(results, output_dir):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(14, 4), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.35

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        mb_means = [results[(si, ci)]["omega_b"][:, -1].mean() for ci in range(len(CONDITIONS))]
        me_means = [results[(si, ci)]["omega_e"][:, -1].mean() for ci in range(len(CONDITIONS))]
        mb_sds   = [results[(si, ci)]["omega_b"][:, -1].std()  for ci in range(len(CONDITIONS))]
        me_sds   = [results[(si, ci)]["omega_e"][:, -1].std()  for ci in range(len(CONDITIONS))]

        ax.bar(x - width/2, mb_means, width, yerr=mb_sds, color="#2171B5", alpha=0.8,
               capsize=3, label="Ω_b")
        ax.bar(x + width/2, me_means, width, yerr=me_sds, color="#238B45", alpha=0.8,
               capsize=3, label="Ω_e")
        ax.set_xticks(x)
        ax.set_xticklabels(COND_SHORT, fontsize=7)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        if si == 0:
            ax.legend(fontsize=8)

    axes[0].set_ylabel("Final mean performance", fontsize=9)
    fig.suptitle("Stage 1B: Final Ω_b and Ω_e by condition and strategy\n"
                 "(bars = mean ± SD across replications)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s1b_fig4_performance.png"), dpi=150)
    plt.close(fig)
    print("Fig 4 saved.")


# ── Figure 5: Empirical correlation boxplots vs Stage 1 reference ─────────────

def make_fig5(results, output_dir):
    """
    Three panels (within-business, within-ethical, cross-type).
    Nine boxplot clusters (D1-D9), grouped by conflict fraction.
    Stage 1 reference correlations loaded from s1_results_v2/s1_per_rep.csv if available.
    """
    corr_types = ["corr_bb_reps", "corr_ee_reps", "corr_be_reps"]
    titles = ["Within-business\ncorrelation",
              "Within-ethical\ncorrelation",
              "Cross-type\ncorrelation"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    # ── load Stage 1 reference correlations if available ─────────────────────
    s1_ref = {}   # condition_label -> {corr_bb: [], corr_ee: [], corr_be: []}
    s1_csv = os.path.join(os.path.dirname(__file__), "s1_results_v2", "s1_per_rep.csv")
    if os.path.exists(s1_csv):
        import csv as csv_mod
        with open(s1_csv) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                key = row["condition"]
                if key not in s1_ref:
                    s1_ref[key] = {"corr_bb": [], "corr_ee": [], "corr_be": []}
                s1_ref[key]["corr_bb"].append(float(row["corr_bb"]))
                s1_ref[key]["corr_ee"].append(float(row["corr_ee"]))
                s1_ref[key]["corr_be"].append(float(row["corr_be"]))
        # Average across strategies (Stage 1 per_rep has one row per rep per strategy)
        s1_means = {}
        for cond, d in s1_ref.items():
            s1_means[cond] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"  Stage 1 reference loaded ({len(s1_means)} conditions).")
    else:
        s1_means = {}
        print(f"  Stage 1 reference not found at {s1_csv}; skipping reference lines.")

    x = np.arange(len(CONDITIONS))

    for ax, corr_key, title, csv_key in zip(
            axes, corr_types, titles,
            ["corr_bb", "corr_ee", "corr_be"]):

        # Aggregate across strategies: pool all reps from all 4 strategies per condition
        data = []
        for ci in range(len(CONDITIONS)):
            pooled = np.concatenate([results[(si, ci)][corr_key]
                                     for si in range(len(STRATEGIES))])
            data.append(pooled)

        bp = ax.boxplot(data, positions=x, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], FRAC_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Stage 1 reference: horizontal dashed lines per S1 condition
        s1_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))
        s1_cond_order = ["C1", "C2", "C3", "C4", "C5"]
        for i, cond in enumerate(s1_cond_order):
            if cond in s1_means:
                val = s1_means[cond][csv_key]
                ax.axhline(val, color=s1_colors[i], linewidth=1.2,
                           linestyle=":", alpha=0.85,
                           label=f"S1 {cond}" if ax == axes[0] else "")

        ax.set_xticks(x)
        ax.set_xticklabels(COND_SHORT, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylabel("First-differenced Pearson r", fontsize=8)

        # Vertical separators between conflict-fraction groups
        for sep in [2.5, 5.5]:
            ax.axvline(sep, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)

        # Group labels at top
        ymax = ax.get_ylim()[1]
        for label, xc in [("f = 0.25", 1), ("f = 0.50", 4), ("f = 0.75", 7)]:
            ax.text(xc, ymax * 0.97, label, ha="center", fontsize=7, color="gray")

    if s1_means:
        axes[0].legend(fontsize=7, loc="upper right", title="Stage 1 ref",
                       title_fontsize=7)

    from matplotlib.patches import Patch
    frac_patches = [
        Patch(color=FRAC_COLORS[0], label="f=0.25"),
        Patch(color=FRAC_COLORS[3], label="f=0.50"),
        Patch(color=FRAC_COLORS[6], label="f=0.75"),
    ]
    fig.legend(handles=frac_patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Stage 1B: Empirical goal-performance correlations\n"
                 "(boxes = distribution across reps × strategies; "
                 "dotted lines = Stage 1 reference means by condition)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s1b_fig5_correlations.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 5 saved.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1B NK landscape sensitivity analysis")
    parser.add_argument("--reps",   type=int, default=30,
                        help="Replications per condition cell (default 30 for pilot)")
    parser.add_argument("--output", type=str, default="s1b_pilot",
                        help="Output directory (default s1b_pilot)")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()
    main(reps=args.reps, output_dir=args.output, base_seed=args.seed)
