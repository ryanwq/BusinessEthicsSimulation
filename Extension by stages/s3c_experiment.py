"""
Stage 3C experiment: aspiration-level innovation on the D5 partition landscape.

Design: 36 cells
  WAC  × {no-I, +I}         × 4 strategies =  8 cells
  SF   × {no-I, +I}         × 4 strategies =  8 cells
  Comp × {no-I, +I}         × 4 strategies =  8 cells
  CEI  × {no-I, Fixed, Ratch} × 4 strategies = 12 cells
  Total: 36 cells × 250 reps = 9,000 replications (primary run)

Usage
-----
    python3 s3c_experiment.py              # 10-rep pilot, console stats
    python3 s3c_experiment.py --reps 250   # full run, all figures and CSV

Outputs (full run)
------------------
    s3c_results/s3c_per_rep.csv
    s3c_results/s3c_timeseries.csv
    s3c_results/s3c_fig1_frontier.png
    s3c_results/s3c_fig2_trajectories.png
    s3c_results/s3c_fig3_n_growth.png
    s3c_results/s3c_fig4_innovation.png
    s3c_results/s3c_fig5_diff.png
    s3c_results/s3c_fig6_flips.png
    s3c_results/s3c_fig7_compliance.png
    s3c_results/s3c_fig8_below_aspiration.png
    s3c_results/s3c_fig9_ratchet_vs_fixed.png
"""
import argparse
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

from s3c_simulation import run_replication_s3c

# ── Fixed parameters (D5) ─────────────────────────────────────────────────────
N, G, G_b, M, R, T = 24, 8, 4, 4, 120, 500
N_CONFLICT, ALPHA, BETA = 12, 0.50, 0.10
OUTPUT_DIR = "s3c_results"

# ── Cell definitions ──────────────────────────────────────────────────────────
STRATEGIES   = ["baseline", "myopic", "spatial", "temporal"]
STRAT_LABEL  = ["Baseline", "Myopic", "Spatial diff.", "Temporal diff."]
STRAT_COLORS = ["#185FA5", "#D85A30", "#3B6D11", "#534AB7"]

MINDSETS      = ["wac", "sf", "compliance", "cei"]
MINDSET_LABEL = ["Win at All Costs", "Survival First", "Compliance", "Cont. Ethical Impr."]
MINDSET_SHORT = ["WAC", "SF", "Comp.", "CEI"]
MINDSET_COLORS = ["#C0392B", "#E67E22", "#2980B9", "#27AE60"]

# Innovation conditions per mindset
INNOV_CONDITIONS = {
    "wac":        [(False, None),   (True, "fixed")],
    "sf":         [(False, None),   (True, "fixed")],
    "compliance": [(False, None),   (True, "fixed")],
    "cei":        [(False, None),   (True, "fixed")],   # ratcheting available via simulation but excluded from default design
}
INNOV_LABEL = {
    (False, None):         "no innov.",
    (True, "fixed"):       "+I Fixed",
    (True, "ratcheting"):  "+I Ratcheting",
}
INNOV_SHORT = {
    (False, None):         "NoI",
    (True, "fixed"):       "Fixed",
    (True, "ratcheting"):  "Ratch",
}
INNOV_LS = {
    (False, None):         "-",
    (True, "fixed"):       "--",
    (True, "ratcheting"):  ":",
}


def pearson_r(a, b):
    da, db = np.diff(a), np.diff(b)
    if da.std() < 1e-12 or db.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])



def _cell_seed(mi, has_innovation, innov_type, si, reps):
    innov_idx = {(False, None): 0, (True, "fixed"): 1, (True, "ratcheting"): 2}[
        (has_innovation, innov_type)]
    cell_idx = (mi * 3 + innov_idx) * len(STRATEGIES) + si
    return cell_idx * reps


# ── single-cell runner ────────────────────────────────────────────────────────

def run_cell_s3c(strategy, mindset, has_innovation, innovation_type, reps, base_seed,
                 beta=BETA):
    """Run `reps` replications for one cell. Returns list of result dicts."""
    mi = MINDSETS.index(mindset)
    si = STRATEGIES.index(strategy)
    results = []

    for rep in range(reps):
        seed = base_seed + rep
        (perf, n_t, biz_idx, eth_idx,
         theta_g, A_g_init, A_g_final,
         acc, innov_cnt, casc_cnt, innov_steps,
         above_all_count, above_goal_counts, ratchet_cnt) = run_replication_s3c(
            N, G, G_b, M, R, T, strategy, N_CONFLICT, ALPHA,
            mindset, has_innovation, innovation_type, seed, beta
        )

        omega_b = perf[:, biz_idx].mean(axis=1)
        omega_e = perf[:, eth_idx].mean(axis=1)

        # Compliance rate: fraction of steps where ALL ethical goals >= θ_g,E simultaneously
        comp_rate_all = above_all_count / T
        comp_by_goal = {g: above_goal_counts[g] / T for g in eth_idx}

        r_bb = np.mean([pearson_r(perf[:, a], perf[:, b])
                        for a, b in combinations(biz_idx, 2)]) if len(biz_idx) > 1 else 0.0
        r_ee = np.mean([pearson_r(perf[:, a], perf[:, b])
                        for a, b in combinations(eth_idx, 2)]) if len(eth_idx) > 1 else 0.0
        r_be = np.mean([pearson_r(perf[:, a], perf[:, b])
                        for a in biz_idx for b in eth_idx])

        intervals = np.diff(innov_steps).tolist() if len(innov_steps) > 1 else []

        results.append({
            "mindset": mindset, "strategy": strategy,
            "has_innovation": has_innovation, "innovation_type": innovation_type,
            "rep": rep, "seed": seed,
            "mean_omega_b": float(omega_b.mean()),
            "mean_omega_e": float(omega_e.mean()),
            "final_omega_b": float(omega_b[-1]),
            "final_omega_e": float(omega_e[-1]),
            "n_final": int(n_t[-1]),
            "innov_count": innov_cnt,
            "cascade_count": casc_cnt,
            "mean_interval": float(np.mean(intervals)) if intervals else 0.0,
            "comp_rate": comp_rate_all,
            # fixed-width per-goal columns (NaN for non-ethical goals) so CSV fieldnames are stable
            **{f"comp_rate_g{g}": comp_by_goal.get(g, float("nan")) for g in range(G)},
            "acc_N": int(acc[0]), "acc_B": int(acc[1]), "acc_E": int(acc[2]),
            "r_bb": r_bb, "r_ee": r_ee, "r_be": r_be,
            "beta": beta,
            "ratchet_count": ratchet_cnt,
            "omega_b_series": omega_b.tolist(),
            "omega_e_series": omega_e.tolist(),
            "n_t_series": n_t.tolist(),
            **{f"theta_g{g}": float(theta_g[g]) for g in range(G)},
            **{f"A_g_init_{g}": float(A_g_init[g]) for g in range(G)},
            **{f"A_g_final_{g}": float(A_g_final[g]) for g in range(G)},
        })

    return results


# ── full experiment ───────────────────────────────────────────────────────────

def run_all_cells(reps, output_dir, beta=BETA):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    total_cells = sum(len(INNOV_CONDITIONS[m]) for m in MINDSETS) * len(STRATEGIES)
    cell_num = 0
    t0 = time.time()

    for si, strategy in enumerate(STRATEGIES):
        for mi, mindset in enumerate(MINDSETS):
            for (has_innovation, innovation_type) in INNOV_CONDITIONS[mindset]:
                cell_num += 1
                base_seed = _cell_seed(mi, has_innovation, innovation_type, si, reps)
                label = (f"{MINDSET_SHORT[mi]}/"
                         f"{STRAT_LABEL[si]}/"
                         f"{INNOV_LABEL[(has_innovation, innovation_type)]}")
                print(f"  [{cell_num:2d}/{total_cells}] {label} ...", end=" ", flush=True)
                ct = time.time()
                cell_res = run_cell_s3c(strategy, mindset, has_innovation,
                                        innovation_type, reps, base_seed, beta=beta)
                all_results.extend(cell_res)
                elapsed = time.time() - ct
                m_ob = np.mean([r["mean_omega_b"] for r in cell_res])
                m_oe = np.mean([r["mean_omega_e"] for r in cell_res])
                cr   = np.mean([r["comp_rate"]    for r in cell_res])
                ni   = np.mean([r["innov_count"]  for r in cell_res])
                print(f"Ωb={m_ob:.3f} Ωe={m_oe:.3f} cr={cr:.3f} "
                      f"innov={ni:.1f}  ({elapsed:.1f}s)")

    print(f"\nTotal: {len(all_results)} reps, {time.time()-t0:.1f}s")
    return all_results


def print_pilot_summary(results):
    """Print a quick table for pilot runs."""
    from collections import defaultdict
    cells = defaultdict(list)
    for r in results:
        key = (r["mindset"], r["strategy"], r["has_innovation"], r["innovation_type"])
        cells[key].append(r)

    header = f"{'Mindset':<12}{'Strategy':<14}{'Innov':<14}{'Ωb':>6}{'Ωe':>6}{'CR':>6}{'N_fin':>7}{'Innov':>7}"
    print("\n" + header)
    print("-" * len(header))
    for mi, mindset in enumerate(MINDSETS):
        for si, strategy in enumerate(STRATEGIES):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                key = (mindset, strategy, hi, it)
                if key not in cells:
                    continue
                rs = cells[key]
                ob = np.mean([r["mean_omega_b"] for r in rs])
                oe = np.mean([r["mean_omega_e"] for r in rs])
                cr = np.mean([r["comp_rate"]    for r in rs])
                nf = np.mean([r["n_final"]      for r in rs])
                ni = np.mean([r["innov_count"]  for r in rs])
                print(f"{MINDSET_SHORT[mi]:<12}{STRAT_LABEL[si]:<14}"
                      f"{INNOV_SHORT[(hi,it)]:<14}"
                      f"{ob:6.3f}{oe:6.3f}{cr:6.3f}{nf:7.1f}{ni:7.1f}")


def save_csv(results, output_dir):
    series_keys = {"omega_b_series", "omega_e_series", "n_t_series"}
    per_rep_path = os.path.join(output_dir, "s3c_per_rep.csv")
    flat = [{k: v for k, v in r.items() if k not in series_keys} for r in results]
    if flat:
        fieldnames = list(flat[0].keys())
        with open(per_rep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flat)
        print(f"Saved: {per_rep_path}")

    ts_path = os.path.join(output_dir, "s3c_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mindset","strategy","has_innovation","innovation_type","rep","t",
                    "omega_b","omega_e","n_t"])
        for r in results:
            for t_idx, (ob, oe, nt) in enumerate(zip(
                    r["omega_b_series"], r["omega_e_series"], r["n_t_series"])):
                w.writerow([r["mindset"], r["strategy"],
                            r["has_innovation"], r["innovation_type"],
                            r["rep"], t_idx, ob, oe, nt])
    print(f"Saved: {ts_path}")


# ── figures ───────────────────────────────────────────────────────────────────

def _cell_mean(results, mindset, strategy, has_innovation, innovation_type, key):
    vals = [r[key] for r in results
            if r["mindset"] == mindset and r["strategy"] == strategy
            and r["has_innovation"] == has_innovation
            and r["innovation_type"] == innovation_type]
    return np.array(vals) if vals else None


def fig_frontier(results, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True, sharex=True)
    for si, (strategy, ax) in enumerate(zip(STRATEGIES, axes)):
        for mi, mindset in enumerate(MINDSETS):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                ob = _cell_mean(results, mindset, strategy, hi, it, "final_omega_b")
                oe = _cell_mean(results, mindset, strategy, hi, it, "final_omega_e")
                if ob is None:
                    continue
                marker = "o" if not hi else ("^" if it == "fixed" else "D")
                ls_label = f"{MINDSET_SHORT[mi]} {INNOV_SHORT[(hi,it)]}"
                ax.scatter(ob.mean(), oe.mean(), color=MINDSET_COLORS[mi],
                           marker=marker, s=80, label=ls_label, zorder=3)
        ax.set_title(STRAT_LABEL[si])
        ax.set_xlabel("Ω_b")
        ax.axline((0,0), slope=1, color="gray", lw=0.7, ls=":")
    axes[0].set_ylabel("Ω_e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure 1 — Business-ethics frontier (final performance)")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig1_frontier.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_trajectories(results, output_dir):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey="row")
    steps = np.arange(T)
    for mi, mindset in enumerate(MINDSETS):
        ax_b, ax_e = axes[mi, 0], axes[mi, 1]
        for si, strategy in enumerate(STRATEGIES):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                all_ob = [r["omega_b_series"] for r in results
                          if r["mindset"]==mindset and r["strategy"]==strategy
                          and r["has_innovation"]==hi and r["innovation_type"]==it]
                if not all_ob:
                    continue
                ob_mean = np.mean(all_ob, axis=0)
                all_oe = [r["omega_e_series"] for r in results
                          if r["mindset"]==mindset and r["strategy"]==strategy
                          and r["has_innovation"]==hi and r["innovation_type"]==it]
                oe_mean = np.mean(all_oe, axis=0)
                lbl = f"{STRAT_LABEL[si]} {INNOV_SHORT[(hi,it)]}"
                ls  = INNOV_LS[(hi, it)]
                ax_b.plot(steps, ob_mean, color=STRAT_COLORS[si], ls=ls,
                          lw=1.4, label=lbl)
                ax_e.plot(steps, oe_mean, color=STRAT_COLORS[si], ls=ls,
                          lw=1.4, label=lbl)
        ax_b.set_ylabel(f"{MINDSET_SHORT[mi]}\nΩ_b")
        ax_e.set_ylabel("Ω_e")
    axes[0, 0].set_title("Business performance Ω_b")
    axes[0, 1].set_title("Ethical performance Ω_e")
    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:8], labels[:8], loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 2 — Performance trajectories")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig2_trajectories.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_n_growth(results, output_dir):
    innov_cells = [(m, s, hi, it)
                   for m in MINDSETS for s in STRATEGIES
                   for (hi, it) in INNOV_CONDITIONS[m] if hi]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True, sharex=True)
    steps = np.arange(T)
    for mi, mindset in enumerate(MINDSETS):
        ax = axes[mi]
        ax.axhline(N, color="gray", lw=0.8, ls=":", label=f"N₀={N}")
        for si, strategy in enumerate(STRATEGIES):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                if not hi:
                    continue
                all_nt = [r["n_t_series"] for r in results
                          if r["mindset"]==mindset and r["strategy"]==strategy
                          and r["has_innovation"]==hi and r["innovation_type"]==it]
                if not all_nt:
                    continue
                nt_mean = np.mean(all_nt, axis=0)
                lbl = f"{STRAT_LABEL[si]} {INNOV_SHORT[(hi,it)]}"
                ax.plot(steps, nt_mean, color=STRAT_COLORS[si],
                        ls=INNOV_LS[(hi, it)], lw=1.4, label=lbl)
        ax.set_title(MINDSET_SHORT[mi])
        ax.set_xlabel("Step")
    axes[0].set_ylabel("N_t (decisions)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure 3 — N_t growth (innovation conditions only)")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig3_n_growth.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_innovation_counts(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    labels, innov_data, casc_data = [], [], []
    for si, strategy in enumerate(STRATEGIES):
        for mi, mindset in enumerate(MINDSETS):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                if not hi:
                    continue
                vals = [r["innov_count"] for r in results
                        if r["mindset"]==mindset and r["strategy"]==strategy
                        and r["has_innovation"]==hi and r["innovation_type"]==it]
                if not vals:
                    continue
                cv  = [r["cascade_count"] for r in results
                       if r["mindset"]==mindset and r["strategy"]==strategy
                       and r["has_innovation"]==hi and r["innovation_type"]==it]
                labels.append(f"{MINDSET_SHORT[mi]}\n{STRAT_LABEL[si][:4]}\n{INNOV_SHORT[(hi,it)]}")
                innov_data.append(vals)
                casc_data.append(cv)

    axes[0].boxplot(innov_data, labels=labels, patch_artist=True)
    axes[0].set_title("Innovations per replication")
    axes[0].set_ylabel("count")
    axes[1].boxplot(casc_data, labels=labels, patch_artist=True)
    axes[1].set_title("Cascade innovations per replication")
    axes[1].set_ylabel("count")
    for ax in axes:
        ax.tick_params(axis='x', labelsize=7)
    fig.suptitle("Figure 4 — Innovation and cascade counts")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig4_innovation.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_diff(results, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    for si, (strategy, ax) in enumerate(zip(STRATEGIES, axes)):
        labels, data = [], []
        for mi, mindset in enumerate(MINDSETS):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                vals = [(r["final_omega_b"] - r["final_omega_e"]) for r in results
                        if r["mindset"]==mindset and r["strategy"]==strategy
                        and r["has_innovation"]==hi and r["innovation_type"]==it]
                if not vals:
                    continue
                labels.append(f"{MINDSET_SHORT[mi]}\n{INNOV_SHORT[(hi,it)]}")
                data.append(vals)
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(STRAT_LABEL[si])
        ax.tick_params(axis='x', labelsize=7)
    axes[0].set_ylabel("Ω_b − Ω_e (final)")
    fig.suptitle("Figure 5 — Business−Ethical performance gap")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig5_diff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_compliance(results, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    for si, (strategy, ax) in enumerate(zip(STRATEGIES, axes)):
        labels, data = [], []
        for mi, mindset in enumerate(MINDSETS):
            for (hi, it) in INNOV_CONDITIONS[mindset]:
                vals = [r["comp_rate"] for r in results
                        if r["mindset"]==mindset and r["strategy"]==strategy
                        and r["has_innovation"]==hi and r["innovation_type"]==it]
                if not vals:
                    continue
                labels.append(f"{MINDSET_SHORT[mi]}\n{INNOV_SHORT[(hi,it)]}")
                data.append(vals)
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_title(STRAT_LABEL[si])
        ax.tick_params(axis='x', labelsize=7)
    axes[0].set_ylabel("Compliance rate (all ethical goals ≥ θ_g,E)")
    n_cells = sum(len(INNOV_CONDITIONS[m]) for m in MINDSETS) * len(STRATEGIES)
    fig.suptitle(f"Figure 7 — Ethical compliance rates (common θ_g,E scale, all {n_cells} cells)")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig7_compliance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_ratchet_vs_fixed(results, output_dir):
    """Figure 9: CEI+I Fixed vs. Ratcheting comparison."""
    metrics = [("final_omega_b","Final Ω_b"), ("final_omega_e","Final Ω_e"),
               ("innov_count","Innovations"), ("comp_rate","Comp. rate")]
    fig, axes = plt.subplots(len(STRATEGIES), len(metrics), figsize=(16, 14),
                             sharey="col")
    for si, strategy in enumerate(STRATEGIES):
        for mi_m, (key, mname) in enumerate(metrics):
            ax = axes[si, mi_m]
            fixed = [r[key] for r in results
                     if r["mindset"]=="cei" and r["strategy"]==strategy
                     and r["innovation_type"]=="fixed"]
            ratch = [r[key] for r in results
                     if r["mindset"]=="cei" and r["strategy"]==strategy
                     and r["innovation_type"]=="ratcheting"]
            if fixed and ratch:
                ax.boxplot([fixed, ratch], labels=["Fixed","Ratch."], patch_artist=True)
            ax.set_title(f"{STRAT_LABEL[si]} — {mname}" if mi_m == 0 else mname,
                         fontsize=8)
    fig.suptitle("Figure 9 — CEI+I Fixed vs. Ratcheting")
    fig.tight_layout()
    path = os.path.join(output_dir, "s3c_fig9_ratchet_vs_fixed.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10,
                    help="Replications per cell (default 10 = pilot)")
    ap.add_argument("--output", default=OUTPUT_DIR,
                    help="Output directory")
    ap.add_argument("--beta", type=float, default=BETA,
                    help="Aspiration stretch factor (default 0.10)")
    args = ap.parse_args()
    is_pilot = args.reps < 50

    total_cells = sum(len(INNOV_CONDITIONS[m]) for m in MINDSETS) * len(STRATEGIES)
    print(f"\nStage 3C Experiment — {'PILOT' if is_pilot else 'FULL RUN'}")
    print(f"  {args.reps} reps × {total_cells} cells = {args.reps*total_cells:,} total replications")
    print(f"  D5: N={N}, G={G}, G_b={G_b}, M={M}, R={R}, T={T}, β={args.beta}")
    print()

    results = run_all_cells(args.reps, args.output, beta=args.beta)

    print_pilot_summary(results)

    if not is_pilot:
        print("\nSaving outputs...")
        save_csv(results, args.output)
        print("Generating figures...")
        fig_frontier(results, args.output)
        fig_trajectories(results, args.output)
        fig_n_growth(results, args.output)
        fig_innovation_counts(results, args.output)
        fig_diff(results, args.output)
        fig_compliance(results, args.output)
        has_ratcheting = any(r["innovation_type"] == "ratcheting" for r in results)
        if has_ratcheting:
            fig_ratchet_vs_fixed(results, args.output)
        print("\nDone.")
    else:
        print(f"\n[Pilot complete — run with --reps 250 for full results]")


if __name__ == "__main__":
    main()
