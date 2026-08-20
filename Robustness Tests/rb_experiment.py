"""
Robustness experiment runner for the NK Landscape Extension Study.

Runs three variants (A, B, C) of the Stage 3C innovation model, the
supplementary high-beta WAC run, and the stochastic Group-B assignment
supplementary (Variants P10 / P20).  Produces:

  rb_per_rep.csv          — one row per replication (all variants)
  rb_timeseries.csv       — averaged time series per +I condition
  rb_highbeta_wac.csv     — high-beta WAC supplementary
  rb_event_study.csv      — 61-step windows around Group-B events (P-variants)

Usage
-----
  python rb_experiment.py              # pilot: 30 reps
  python rb_experiment.py --reps 120   # full run
  python rb_experiment.py --variant P  # Group-B supplementary only
  python rb_experiment.py --highbeta   # high-beta WAC only
"""
import argparse
import csv
import os
import sys
import time
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.join(_HERE, "..", "Extension by stages")
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rb_simulation import run_replication_rb

# ── fixed parameters ──────────────────────────────────────────────────────────
N, G, G_b, M, R, T = 24, 8, 4, 4, 120, 500
N_CONFLICT, ALPHA, BETA = 12, 0.50, 0.10

STRATEGIES = ["baseline", "myopic", "spatial", "temporal"]
MINDSETS   = ["wac", "sf", "compliance", "cei"]

INNOV_CONDITIONS = {
    "wac":        [(False, None), (True, "fixed")],
    "sf":         [(False, None), (True, "fixed")],
    "compliance": [(False, None), (True, "fixed")],
    "cei":        [(False, None), (True, "fixed")],
}

# ── variant definitions ───────────────────────────────────────────────────────
# Each entry maps to kwargs forwarded to run_replication_rb.
VARIANTS = {
    "baseline":  dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "A75":       dict(innov_pi=0.75, innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "A50":       dict(innov_pi=0.50, innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "B":         dict(innov_pi=1.0,  innov_symmetric=True,  innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "C005":      dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.005, innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "C010":      dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.010, innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "D_K3_005":  dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.005, innov_cost_periods=3, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "D_K3_010":  dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.010, innov_cost_periods=3, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "D_K5_005":  dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.005, innov_cost_periods=5, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "D_K5_010":  dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.010, innov_cost_periods=5, innov_hold_max=0,  innov_tradeoff_prob=0.0),
    "E":         dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=10, innov_tradeoff_prob=0.0),
    "P10":       dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.10),
    "P20":       dict(innov_pi=1.0,  innov_symmetric=False, innov_cost=0.0,   innov_cost_periods=1, innov_hold_max=0,  innov_tradeoff_prob=0.20),
}

VARIANT_LABEL = {
    "baseline":  "Baseline (primary)",
    "A75":       "Variant A (π=0.75)",
    "A50":       "Variant A (π=0.50)",
    "B":         "Variant B (symmetric)",
    "C005":      "Variant C (δ=0.005)",
    "C010":      "Variant C (δ=0.010)",
    "D_K3_005":  "Variant D (K=3, δ=0.005)",
    "D_K3_010":  "Variant D (K=3, δ=0.010)",
    "D_K5_005":  "Variant D (K=5, δ=0.005)",
    "D_K5_010":  "Variant D (K=5, δ=0.010)",
    "E":         "Variant E (hold L~U{0,10})",
    "P10":       "Group-B p=0.10",
    "P20":       "Group-B p=0.20",
}

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Saved: {path}")


# ── single-cell runner ────────────────────────────────────────────────────────

def run_cell(variant_key, mindset, strategy, has_innov, innov_type,
             reps, base_seed=0, beta=BETA):
    """Run one cell; return (per_rep_rows, event_windows).

    event_windows : list of (is_group_b, g_star_ethical, step, window)
                    where window is a (61, n_eth) array of ethical performance
                    in the [-30, +30] step range around the event.
                    Only populated when innov_tradeoff_prob > 0 and has_innov.
    """
    vflags       = VARIANTS[variant_key]
    rows         = []
    event_windows = []   # collected across reps for P-variants

    for rep in range(reps):
        seed   = base_seed + rep
        result, gb_log = run_replication_rb(
            N=N, G=G, G_b=G_b, M=M, R=R, T=T,
            regime=strategy,
            n_conflict=N_CONFLICT, alpha=ALPHA,
            mindset=mindset,
            has_innovation=has_innov,
            innovation_type=innov_type,
            seed=seed,
            beta=beta,
            **vflags,
        )
        (perf, n_t, biz_idx, eth_idx,
         theta_g, A_g_init, A_g_final,
         acc, innov_cnt, casc_cnt, innov_steps,
         above_all_cnt, above_goal_cnts, ratchet_cnt) = result

        final_perf = perf[-1]
        biz_perf   = float(final_perf[biz_idx].mean())
        eth_perf   = float(final_perf[eth_idx].mean())
        avg_perf   = float(final_perf.mean())
        comp_rate  = float(above_all_cnt) / T

        group_b_cnt = sum(1 for (gb, _) in gb_log if gb)

        row = {
            "variant":        variant_key,
            "variant_label":  VARIANT_LABEL[variant_key],
            "mindset":        mindset,
            "strategy":       strategy,
            "has_innovation": int(has_innov),
            "innov_type":     innov_type or "none",
            "innov_pi":       vflags["innov_pi"],
            "innov_symmetric": int(vflags["innov_symmetric"]),
            "innov_cost":     vflags["innov_cost"],
            "innov_tradeoff_prob": vflags["innov_tradeoff_prob"],
            "beta":           beta,
            "rep":            rep,
            "seed":           seed,
            "final_avg_omega":  round(avg_perf, 6),
            "final_biz_omega":  round(biz_perf, 6),
            "final_eth_omega":  round(eth_perf, 6),
            "compliance_rate":  round(comp_rate, 6),
            "innov_count":    innov_cnt,
            "group_b_count":  group_b_cnt,
            "cascade_count":  casc_cnt,
            "ratchet_count":  ratchet_cnt,
        }
        for g in range(G):
            row[f"final_omega_goal_{g}"] = round(float(final_perf[g]), 6)
        rows.append(row)

        # ── event-study windows (P-variants only) ────────────────────────
        if vflags["innov_tradeoff_prob"] > 0.0 and has_innov:
            n_eth = len(eth_idx)
            eth_arr = np.array(eth_idx)
            for (is_gb, g_eth), step in zip(gb_log, innov_steps):
                if not is_gb:
                    continue
                t0 = step - 30
                t1 = step + 31       # exclusive; window length = 61
                # Extract with edge padding
                window = np.full((61, n_eth), np.nan)
                for wi, t in enumerate(range(t0, t1)):
                    if 0 <= t < T:
                        window[wi] = perf[t, eth_arr]
                event_windows.append({
                    "variant":   variant_key,
                    "mindset":   mindset,
                    "strategy":  strategy,
                    "p_value":   vflags["innov_tradeoff_prob"],
                    "g_star_ethical": int(g_eth),
                    "rep":       rep,
                    "step":      step,
                    "window":    window,   # (61, n_eth) — stored in memory only
                })

    return rows, event_windows


def run_cell_timeseries(variant_key, mindset, strategy, has_innov, innov_type,
                         reps, base_seed=0, beta=BETA):
    """Run one cell; return averaged time-series rows."""
    vflags   = VARIANTS[variant_key]
    perf_all = []
    biz_idx_all = []
    eth_idx_all = []
    for rep in range(reps):
        seed = base_seed + rep
        result, _ = run_replication_rb(
            N=N, G=G, G_b=G_b, M=M, R=R, T=T,
            regime=strategy,
            n_conflict=N_CONFLICT, alpha=ALPHA,
            mindset=mindset,
            has_innovation=has_innov,
            innovation_type=innov_type,
            seed=seed,
            beta=beta,
            **vflags,
        )
        perf_all.append(result[0])
        biz_idx_all.append(result[2])
        eth_idx_all.append(result[3])

    perf_stack = np.stack(perf_all)
    mean_perf  = perf_stack.mean(axis=0)

    # Average biz and eth separately across reps (goal assignment varies per rep)
    mean_biz = np.mean([perf_stack[i][:, biz_idx_all[i]].mean(axis=1)
                        for i in range(reps)], axis=0)
    mean_eth = np.mean([perf_stack[i][:, eth_idx_all[i]].mean(axis=1)
                        for i in range(reps)], axis=0)

    rows = []
    for t in range(T):
        rows.append({
            "variant":        variant_key,
            "mindset":        mindset,
            "strategy":       strategy,
            "has_innovation": int(has_innov),
            "innov_type":     innov_type or "none",
            "beta":           beta,
            "time_step":      t,
            "avg_omega":      round(float(mean_perf[t].mean()), 6),
            "biz_omega":      round(float(mean_biz[t]), 6),
            "eth_omega":      round(float(mean_eth[t]), 6),
        })
    return rows


# ── event-study aggregator ────────────────────────────────────────────────────

def aggregate_event_windows(all_windows):
    """
    Aggregate per-event windows into per-(mindset, strategy, p_value, g_star_type)
    mean trajectories.  Returns a list of CSV rows, one per time offset per group.
    """
    from collections import defaultdict
    buckets = defaultdict(list)   # key -> list of (61,) arrays

    for ev in all_windows:
        key = (ev["variant"], ev["mindset"], ev["strategy"],
               ev["p_value"], ev["g_star_ethical"])
        # Mean ethical performance across ethical goals at each time offset
        buckets[key].append(ev["window"].mean(axis=1))   # (61,)

    rows = []
    offsets = list(range(-30, 31))
    for key, windows in buckets.items():
        arr = np.array(windows)          # (n_events, 61)
        mean = np.nanmean(arr, axis=0)   # (61,)
        se   = np.nanstd(arr, axis=0) / np.sqrt((~np.isnan(arr)).sum(axis=0).clip(1))
        variant, mindset, strategy, p_val, g_eth = key
        for i, offset in enumerate(offsets):
            rows.append({
                "variant":        variant,
                "mindset":        mindset,
                "strategy":       strategy,
                "p_value":        p_val,
                "g_star_ethical": g_eth,
                "time_offset":    offset,
                "mean_eth_perf":  round(float(mean[i]), 6) if not np.isnan(mean[i]) else "",
                "se_eth_perf":    round(float(se[i]),   6) if not np.isnan(se[i])   else "",
                "n_events":       int((~np.isnan(arr[:, i])).sum()),
            })
    return rows


# ── main experiment runners ───────────────────────────────────────────────────

def run_variants(variant_keys, reps, output_dir, base_seed=0):
    """Run all mindset × strategy × innov_cond cells for a list of variants."""
    per_rep_rows  = []
    ts_rows       = []
    all_ev_windows = []

    for vk in variant_keys:
        print(f"\n  Variant {vk} ({VARIANT_LABEL[vk]})")
        for mindset in MINDSETS:
            for has_innov, innov_type in INNOV_CONDITIONS[mindset]:
                # No-I cells are identical across draw variants — run once
                if not has_innov and vk != "baseline":
                    continue
                for strategy in STRATEGIES:
                    label = (f"    {vk}/{mindset}/{strategy}/"
                             f"{'I' if has_innov else 'noI'}")
                    print(label, end=" ... ", flush=True)
                    t0 = time.time()
                    rows, ev_wins = run_cell(vk, mindset, strategy, has_innov,
                                             innov_type, reps, base_seed)
                    per_rep_rows.extend(rows)
                    all_ev_windows.extend(ev_wins)
                    if has_innov:
                        ts = run_cell_timeseries(vk, mindset, strategy,
                                                  has_innov, innov_type,
                                                  reps, base_seed)
                        ts_rows.extend(ts)
                    print(f"{time.time()-t0:.1f}s")

    _save_csv(per_rep_rows, os.path.join(output_dir, "rb_per_rep.csv"))
    _save_csv(ts_rows,      os.path.join(output_dir, "rb_timeseries.csv"))

    if all_ev_windows:
        ev_rows = aggregate_event_windows(all_ev_windows)
        _save_csv(ev_rows, os.path.join(output_dir, "rb_event_study.csv"))


def run_highbeta_wac(reps, output_dir, base_seed=0):
    """Supplementary: WAC at β = 0.10, 0.30, 0.50 across all strategies."""
    print("\n  Supplementary: High-β WAC")
    rows = []
    for beta in [0.10, 0.30, 0.50]:
        for strategy in STRATEGIES:
            print(f"    WAC / β={beta} / {strategy} ... ", end="", flush=True)
            t0 = time.time()
            r, _ = zip(*[
                run_replication_rb(
                    N=N, G=G, G_b=G_b, M=M, R=R, T=T,
                    regime=strategy, n_conflict=N_CONFLICT, alpha=ALPHA,
                    mindset="wac", has_innovation=True, innovation_type="fixed",
                    seed=base_seed + rep, beta=beta,
                    **VARIANTS["baseline"],
                )
                for rep in range(reps)
            ])
            for rep, result in enumerate(r):
                (perf, n_t, biz_idx, eth_idx,
                 theta_g, A_g_init, A_g_final,
                 acc, innov_cnt, casc_cnt, innov_steps,
                 above_all_cnt, above_goal_cnts, ratchet_cnt) = result
                rows.append({
                    "variant":       "baseline",
                    "mindset":       "wac",
                    "strategy":      strategy,
                    "beta":          beta,
                    "rep":           rep,
                    "final_avg_omega":  round(float(perf[-1].mean()), 6),
                    "final_biz_omega":  round(float(perf[-1, biz_idx].mean()), 6),
                    "final_eth_omega":  round(float(perf[-1, eth_idx].mean()), 6),
                    "compliance_rate":  round(float(above_all_cnt) / T, 6),
                    "innov_count":   innov_cnt,
                    "group_b_count": innov_cnt,  # WAC always Group B
                })
            print(f"{time.time()-t0:.1f}s")

    _save_csv(rows, os.path.join(output_dir, "rb_highbeta_wac.csv"))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Robustness experiment runner")
    ap.add_argument("--reps",     type=int, default=30)
    ap.add_argument("--variant",  type=str, default="all",
                    choices=["all", "A", "B", "C", "P", "baseline"])
    ap.add_argument("--highbeta", action="store_true")
    ap.add_argument("--output",   type=str, default=_HERE)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.highbeta:
        run_highbeta_wac(args.reps, args.output, args.seed)
        return

    if args.variant == "all":
        keys = list(VARIANTS.keys())
    elif args.variant == "baseline":
        keys = ["baseline"]
    elif args.variant == "A":
        keys = ["A75", "A50"]
    elif args.variant == "B":
        keys = ["B"]
    elif args.variant == "C":
        keys = ["C005", "C010"]
    elif args.variant == "P":
        keys = ["baseline", "P10", "P20"]   # baseline needed as p=0 reference
    else:
        keys = [args.variant]

    t0 = time.time()
    print(f"\nRobustness experiment — {args.reps} reps, variants: {keys}")
    run_variants(keys, args.reps, args.output, args.seed)
    run_highbeta_wac(args.reps, args.output, args.seed)
    print(f"\nDone in {time.time()-t0:.0f}s.  Outputs: {args.output}/")


if __name__ == "__main__":
    main()
