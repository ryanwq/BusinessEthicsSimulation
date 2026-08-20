"""
Stage 2B experiment: ethical mindsets on the Stage 1B partition landscape.

Fixed landscape: D5 (n_conflict=12, alpha=0.50, N=24, G=8, G_b=4, G_e=4, M=4, R=120, T=250)
Design: 4 ethical mindsets × 4 strategies × 250 reps = 4,000 replications.
Reference R0: goal-blind rule at D5 (Stage 1B baseline, 250 reps per strategy).

Usage
-----
    python3 s2b_experiment.py              # 15-rep pilot
    python3 s2b_experiment.py --reps 250   # full run

Outputs
-------
    <output>/s2b_per_rep.csv
    <output>/s2b_timeseries.csv
    <output>/s2b_fig1_frontier.png       Business-ethics frontier by strategy
    <output>/s2b_fig2_trajectories.png   Omega_b / Omega_e trajectories by mindset
    <output>/s2b_fig3_diff.png           Omega_b - Omega_e boxplots (H1/H2)
    <output>/s2b_fig4_flips.png          Accepted flips by group type (H5)
    <output>/s2b_fig5_correlations.png   Empirical goal correlations by mindset
    <output>/s2b_fig6_compliance.png     Ethical threshold compliance rate
"""
import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

from s2b_simulation import run_replication_s2b
from s1b_simulation import run_replication_s1b

# ── fixed parameters (D5) ─────────────────────────────────────────────────────
N, G, G_b, M, R, T = 24, 8, 4, 4, 120, 250
N_CONFLICT, ALPHA = 12, 0.50

STRATEGIES   = ["baseline", "myopic", "spatial", "temporal"]
STRAT_LABEL  = ["Baseline", "Myopic", "Spatial diff.", "Temporal diff."]
STRAT_COLORS = ["#185FA5", "#D85A30", "#3B6D11", "#534AB7"]

MINDSETS     = ["wac", "sf", "compliance", "cei"]
MINDSET_LABEL = ["Win at All Costs", "Survival First", "Compliance", "Cont. Ethical Impr."]
MINDSET_SHORT = ["WAC", "SF", "Comp.", "CEI"]
MINDSET_COLORS = ["#C0392B", "#E67E22", "#2980B9", "#27AE60"]


def pearson_r(series_a, series_b):
    da = np.diff(series_a)
    db = np.diff(series_b)
    if da.std() < 1e-12 or db.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def compliance_rate(perf_series, thresholds, ethical_idx):
    """
    Fraction of time steps at which ALL ethical goals are at or above threshold.
    """
    n_above = 0
    for t in range(len(perf_series)):
        if all(perf_series[t, g] >= thresholds[g] for g in ethical_idx):
            n_above += 1
    return n_above / len(perf_series)


# ── single-cell runner ────────────────────────────────────────────────────────

def run_cell(strategy, mindset, reps, base_seed):
    """
    Run `reps` replications for one (strategy, mindset) cell at D5.

    Returns dict with:
      omega_b       : (reps, T)
      omega_e       : (reps, T)
      acc_n, acc_b, acc_e : (reps,) — accepted flips by group type
      corr_bb_reps  : (reps,)
      corr_ee_reps  : (reps,)
      corr_be_reps  : (reps,)
      comp_rate     : (reps,) — fraction of steps all ethical goals above threshold
    """
    omega_b = np.zeros((reps, T))
    omega_e = np.zeros((reps, T))
    acc_n   = np.zeros(reps, dtype=np.int64)
    acc_b   = np.zeros(reps, dtype=np.int64)
    acc_e   = np.zeros(reps, dtype=np.int64)
    corr_bb = np.zeros(reps)
    corr_ee = np.zeros(reps)
    corr_be = np.zeros(reps)
    comp    = np.zeros(reps)

    G_e = G - G_b
    bb_pairs = list(combinations(range(G_b), 2))
    ee_pairs = list(combinations(range(G_e), 2))

    for rep in range(reps):
        seed = base_seed + rep
        perf, b_idx, e_idx, thresholds, acc_counts = run_replication_s2b(
            N, G, G_b, M, R, T, strategy, N_CONFLICT, ALPHA, mindset, seed
        )

        pb = perf[:, b_idx]
        pe = perf[:, e_idx]
        omega_b[rep] = pb.mean(axis=1)
        omega_e[rep] = pe.mean(axis=1)

        acc_n[rep] = acc_counts[0]
        acc_b[rep] = acc_counts[1]
        acc_e[rep] = acc_counts[2]

        corr_bb[rep] = (np.mean([pearson_r(pb[:, i], pb[:, j]) for i, j in bb_pairs])
                        if bb_pairs else 0.0)
        corr_ee[rep] = (np.mean([pearson_r(pe[:, i], pe[:, j]) for i, j in ee_pairs])
                        if ee_pairs else 0.0)
        be_pairs = [(bi, ei) for bi in range(G_b) for ei in range(G_e)]
        corr_be[rep] = np.mean([pearson_r(pb[:, bi], pe[:, ei]) for bi, ei in be_pairs])

        comp[rep] = compliance_rate(perf, thresholds, e_idx)

    return {
        "omega_b":      omega_b,
        "omega_e":      omega_e,
        "acc_n":        acc_n,
        "acc_b":        acc_b,
        "acc_e":        acc_e,
        "corr_bb_reps": corr_bb,
        "corr_ee_reps": corr_ee,
        "corr_be_reps": corr_be,
        "comp_rate":    comp,
    }


def run_reference(strategy, reps, base_seed):
    """Goal-blind reference (Stage 1B D5) for one strategy."""
    omega_b = np.zeros((reps, T))
    omega_e = np.zeros((reps, T))
    for rep in range(reps):
        perf, b_idx, e_idx = run_replication_s1b(
            N, G, G_b, M, R, T, strategy, N_CONFLICT, ALPHA, base_seed + rep
        )
        omega_b[rep] = perf[:, b_idx].mean(axis=1)
        omega_e[rep] = perf[:, e_idx].mean(axis=1)
    return {"omega_b": omega_b, "omega_e": omega_e}


# ── main ──────────────────────────────────────────────────────────────────────

def main(reps=15, output_dir="s2b_pilot", base_seed=0):
    os.makedirs(output_dir, exist_ok=True)
    n_cells = len(STRATEGIES) * len(MINDSETS)
    print(f"Stage 2B: 4 mindsets × 4 strategies × {reps} reps = {n_cells * reps} replications")
    print(f"Fixed landscape: D5 (n_conflict={N_CONFLICT}, α={ALPHA:.2f})")

    results = {}    # (si, mi) -> dict
    ref_r0  = {}    # si -> dict — goal-blind reference
    per_rep_rows = []
    done = 0

    # ── reference condition R0 ────────────────────────────────────────────────
    print("\nRunning reference R0 (goal-blind) ...")
    for si, strategy in enumerate(STRATEGIES):
        seed_r0 = 999_000 + si * reps
        ref_r0[si] = run_reference(strategy, reps, seed_r0)
        fb = ref_r0[si]["omega_b"][:, -1].mean()
        fe = ref_r0[si]["omega_e"][:, -1].mean()
        print(f"  R0 {STRAT_LABEL[si]:16s}: Ω_b={fb:.3f}  Ω_e={fe:.3f}  Δ={fb-fe:+.4f}")

    # ── mindset cells ─────────────────────────────────────────────────────────
    print("\nRunning mindset cells ...")
    for mi, mindset in enumerate(MINDSETS):
        for si, strategy in enumerate(STRATEGIES):
            done += 1
            seed_off = (mi * len(STRATEGIES) + si) * reps + base_seed
            print(f"  [{done:2d}/{n_cells}] {MINDSET_SHORT[mi]:5s} × {STRAT_LABEL[si]:16s} ...",
                  end=" ", flush=True)

            res = run_cell(strategy, mindset, reps, seed_off)
            results[(si, mi)] = res

            fb   = res["omega_b"][:, -1].mean()
            fe   = res["omega_e"][:, -1].mean()
            diff = fb - fe
            total_acc = (res["acc_n"].mean() + res["acc_b"].mean() + res["acc_e"].mean())
            pct_n = res["acc_n"].mean() / total_acc * 100 if total_acc > 0 else 0
            pct_b = res["acc_b"].mean() / total_acc * 100 if total_acc > 0 else 0
            pct_e = res["acc_e"].mean() / total_acc * 100 if total_acc > 0 else 0
            print(f"Ω_b={fb:.3f}  Ω_e={fe:.3f}  Δ={diff:+.4f}  "
                  f"acc N/B/E={pct_n:.0f}%/{pct_b:.0f}%/{pct_e:.0f}%  "
                  f"comp={res['comp_rate'].mean():.3f}")

            for rep in range(reps):
                per_rep_rows.append({
                    "strategy":   strategy,
                    "mindset":    mindset,
                    "rep":        rep,
                    "final_omega_b": float(res["omega_b"][rep, -1]),
                    "final_omega_e": float(res["omega_e"][rep, -1]),
                    "diff_be":       float(res["omega_b"][rep, -1] - res["omega_e"][rep, -1]),
                    "acc_n":     int(res["acc_n"][rep]),
                    "acc_b":     int(res["acc_b"][rep]),
                    "acc_e":     int(res["acc_e"][rep]),
                    "corr_bb":   float(res["corr_bb_reps"][rep]),
                    "corr_ee":   float(res["corr_ee_reps"][rep]),
                    "corr_be":   float(res["corr_be_reps"][rep]),
                    "comp_rate": float(res["comp_rate"][rep]),
                })

    # ── CSVs ─────────────────────────────────────────────────────────────────
    per_rep_path = os.path.join(output_dir, "s2b_per_rep.csv")
    with open(per_rep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_rep_rows[0].keys())
        w.writeheader()
        w.writerows(per_rep_rows)
    print(f"\nPer-rep data: {per_rep_path}")

    ts_rows = []
    for mi, mindset in enumerate(MINDSETS):
        for si, strategy in enumerate(STRATEGIES):
            res = results[(si, mi)]
            for t in range(T):
                ts_rows.append({
                    "strategy": strategy,
                    "mindset":  mindset,
                    "t":        t,
                    "omega_b":  float(res["omega_b"][:, t].mean()),
                    "omega_e":  float(res["omega_e"][:, t].mean()),
                })
    ts_path = os.path.join(output_dir, "s2b_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ts_rows[0].keys())
        w.writeheader()
        w.writerows(ts_rows)
    print(f"Time-series data: {ts_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    make_fig1(results, ref_r0, output_dir)
    make_fig2(results, ref_r0, output_dir)
    make_fig3(results, ref_r0, output_dir)
    make_fig4(results, output_dir)
    make_fig5(results, output_dir)
    make_fig6(results, output_dir)
    print(f"\nAll outputs saved to {output_dir}/")


# ── Figure 1: Business-ethics frontier ────────────────────────────────────────

def make_fig1(results, ref_r0, output_dir):
    """
    One panel per strategy (4). Each point = one mindset's final (Ω_b, Ω_e).
    R0 reference plotted as a grey X. Diagonal = symmetry line.
    """
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(14, 4),
                             sharey=True, sharex=True)

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        # R0 reference
        fb0 = ref_r0[si]["omega_b"][:, -1].mean()
        fe0 = ref_r0[si]["omega_e"][:, -1].mean()
        ax.scatter(fb0, fe0, marker="x", s=80, color="gray", zorder=4,
                   linewidths=2, label="R0 (goal-blind)")

        for mi, mindset in enumerate(MINDSETS):
            res = results[(si, mi)]
            fb  = res["omega_b"][:, -1].mean()
            fe  = res["omega_e"][:, -1].mean()
            ax.scatter(fb, fe, color=MINDSET_COLORS[mi], s=60,
                       zorder=3, label=MINDSET_SHORT[mi])
            ax.annotate(MINDSET_SHORT[mi], (fb, fe),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.4)
        ax.set_xlabel("Final Ω_b", fontsize=9)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Final Ω_e", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Stage 2B: Business-ethics frontier by strategy and mindset\n"
                 "(D5 fixed landscape; diagonal = symmetry; X = goal-blind reference)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s2b_fig1_frontier.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 saved.")


# ── Figure 2: Trajectories by mindset ─────────────────────────────────────────

def make_fig2(results, ref_r0, output_dir):
    """
    4 rows (mindsets) × 2 cols (Ω_b, Ω_e). Lines = strategies. Dashed = R0.
    """
    steps = np.arange(T)
    fig, axes = plt.subplots(len(MINDSETS), 2,
                             figsize=(11, 2.8 * len(MINDSETS)), sharex=True, sharey=True)

    for mi in range(len(MINDSETS)):
        for si in range(len(STRATEGIES)):
            res = results[(si, mi)]
            mb  = res["omega_b"].mean(axis=0)
            me  = res["omega_e"].mean(axis=0)
            lbl = STRAT_LABEL[si] if mi == 0 else ""
            axes[mi, 0].plot(steps, mb, color=STRAT_COLORS[si],
                             linewidth=1.3, label=lbl)
            axes[mi, 1].plot(steps, me, color=STRAT_COLORS[si],
                             linewidth=1.3)

        # R0 reference (dashed, only Baseline strategy for clarity)
        mb0 = ref_r0[0]["omega_b"].mean(axis=0)
        me0 = ref_r0[0]["omega_e"].mean(axis=0)
        axes[mi, 0].plot(steps, mb0, "k--", linewidth=0.9, alpha=0.5,
                         label="R0 (baseline, goal-blind)" if mi == 0 else "")
        axes[mi, 1].plot(steps, me0, "k--", linewidth=0.9, alpha=0.5)

        axes[mi, 0].set_ylabel(f"{MINDSET_SHORT[mi]}\nΩ_b", fontsize=8)
        axes[mi, 1].set_ylabel("Ω_e", fontsize=8)
        for col in range(2):
            axes[mi, col].grid(True, alpha=0.25)

    axes[0, 0].set_title("Business performance (Ω_b)", fontsize=9)
    axes[0, 1].set_title("Ethical performance (Ω_e)", fontsize=9)
    axes[-1, 0].set_xlabel("Time step", fontsize=9)
    axes[-1, 1].set_xlabel("Time step", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strategy", fontsize=8,
               loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Stage 2B: Performance trajectories by mindset\n"
                 "(mean across reps; dashed = goal-blind Baseline reference)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s2b_fig2_trajectories.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 2 saved.")


# ── Figure 3: Ω_b − Ω_e boxplots (H1/H2) ────────────────────────────────────

def make_fig3(results, ref_r0, output_dir):
    """
    4 panels (strategies). Within each panel, 4 boxplots (mindsets).
    Dashed line = R0 (goal-blind) median diff.
    Tests H1 (WAC breaks symmetry) and H2 (CEI restores it).
    """
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(13, 4.5), sharey=True)

    x = np.arange(len(MINDSETS))

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        data = [
            results[(si, mi)]["omega_b"][:, -1] - results[(si, mi)]["omega_e"][:, -1]
            for mi in range(len(MINDSETS))
        ]
        bp = ax.boxplot(data, positions=x, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], MINDSET_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # R0 reference median
        r0_diff = (ref_r0[si]["omega_b"][:, -1] - ref_r0[si]["omega_e"][:, -1]).mean()
        ax.axhline(r0_diff, color="gray", linewidth=1.2, linestyle="--", alpha=0.8,
                   label=f"R0 mean = {r0_diff:+.4f}")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(MINDSET_SHORT, fontsize=8)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25, axis="y")

    axes[0].set_ylabel("Ω_b − Ω_e (final step)", fontsize=9)
    fig.suptitle(
        "Stage 2B: Business-ethics performance gap by mindset and strategy\n"
        "(H1: WAC → large positive gap; H2: CEI → gap near zero)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s2b_fig3_diff.png"), dpi=150)
    plt.close(fig)
    print("Fig 3 saved.")


# ── Figure 4: Accepted flips by group type (H5) ───────────────────────────────

def make_fig4(results, output_dir):
    """
    Grouped bars: for each strategy (4 rows of subplots),
    show 4 mindset groups, each with stacked N/B/E counts.
    The B/E ratio directly explains the Ω_b − Ω_e gap (H5).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    x     = np.arange(len(MINDSETS))
    width = 0.55

    for si, ax in enumerate(axes_flat):
        n_means = np.array([results[(si, mi)]["acc_n"].mean() for mi in range(len(MINDSETS))])
        b_means = np.array([results[(si, mi)]["acc_b"].mean() for mi in range(len(MINDSETS))])
        e_means = np.array([results[(si, mi)]["acc_e"].mean() for mi in range(len(MINDSETS))])

        ax.bar(x, n_means, width, label="Group N (neutral)", color="#AAAAAA", alpha=0.8)
        ax.bar(x, b_means, width, bottom=n_means, label="Group B (business-favoring)",
               color="#2171B5", alpha=0.8)
        ax.bar(x, e_means, width, bottom=n_means + b_means,
               label="Group E (ethical-favoring)", color="#238B45", alpha=0.8)

        # Annotate B/E ratio
        for mi in range(len(MINDSETS)):
            bv = b_means[mi]
            ev = e_means[mi]
            ratio = bv / ev if ev > 0 else float("inf")
            total = n_means[mi] + bv + ev
            ax.text(x[mi], total + 0.5, f"B/E\n{ratio:.2f}",
                    ha="center", va="bottom", fontsize=7, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(MINDSET_SHORT, fontsize=9)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        if si == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.text(0.04, 0.5, "Mean accepted flips per replication", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle(
        "Stage 2B: Accepted flips by decision group type\n"
        "(H5: B/E acceptance ratio predicts Ω_b − Ω_e gap; "
        "goal-blind rule → B/E = 1.0)",
        fontsize=10)
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(os.path.join(output_dir, "s2b_fig4_flips.png"), dpi=150)
    plt.close(fig)
    print("Fig 4 saved.")


# ── Figure 5: Empirical correlations by mindset ───────────────────────────────

def make_fig5(results, output_dir):
    """
    Three panels (within-B, within-E, cross-type).
    Within each panel, 4 boxplot groups (one per mindset), each with 4 strategies pooled.
    """
    corr_keys = ["corr_bb_reps", "corr_ee_reps", "corr_be_reps"]
    titles     = ["Within-business\ncorrelation",
                  "Within-ethical\ncorrelation",
                  "Cross-type\ncorrelation"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(MINDSETS))

    for ax, corr_key, title in zip(axes, corr_keys, titles):
        data = [
            np.concatenate([results[(si, mi)][corr_key]
                            for si in range(len(STRATEGIES))])
            for mi in range(len(MINDSETS))
        ]
        bp = ax.boxplot(data, positions=x, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], MINDSET_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MINDSET_SHORT, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("First-differenced Pearson r", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Stage 2B: Empirical goal-performance correlations by mindset\n"
        "(pooled across strategies; H4: mindsets induce correlation structure)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s2b_fig5_correlations.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Fig 5 saved.")


# ── Figure 6: Ethical compliance rate ─────────────────────────────────────────

def make_fig6(results, output_dir):
    """
    Fraction of time steps at which all ethical goals are at/above threshold.
    4 panels (strategies), 4 groups per panel (mindsets).
    H3: Compliance mindset → highest compliance rate.
    """
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(13, 4.5), sharey=True)
    x = np.arange(len(MINDSETS))

    for si, (ax, strategy) in enumerate(zip(axes, STRATEGIES)):
        means = [results[(si, mi)]["comp_rate"].mean() for mi in range(len(MINDSETS))]
        sds   = [results[(si, mi)]["comp_rate"].std()  for mi in range(len(MINDSETS))]

        bars = ax.bar(x, means, 0.6, yerr=sds, capsize=4,
                      color=MINDSET_COLORS, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(MINDSET_SHORT, fontsize=8)
        ax.set_title(STRAT_LABEL[si], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

        for bar, mean_v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_v + 0.02,
                    f"{mean_v:.2f}", ha="center", fontsize=7)

    axes[0].set_ylabel("Fraction of steps above all thresholds", fontsize=9)
    fig.suptitle(
        "Stage 2B: Ethical threshold compliance rate by mindset and strategy\n"
        "(H3: Compliance mindset → highest rate; WAC → lowest)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "s2b_fig6_compliance.png"), dpi=150)
    plt.close(fig)
    print("Fig 6 saved.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2B NK landscape mindset experiment")
    parser.add_argument("--reps",   type=int, default=15,
                        help="Replications per cell (default 15 for pilot)")
    parser.add_argument("--output", type=str, default="s2b_pilot",
                        help="Output directory (default s2b_pilot)")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()
    main(reps=args.reps, output_dir=args.output, base_seed=args.seed)
