"""
Unified runner for the Ethiraj-Levinthal NK Landscape Study.

Run this script from the ethiraj_levinthal/ folder.  It automatically
finds the simulation code in the Extension by stages/ and robustness tests/
sub-folders — you do not need to move any files.

Stage overview
--------------
  1b          Goal-blind replication of Ethiraj & Levinthal (2003)
  2b          Mindset comparison (WAC, SF, Compliance, CEI) — no innovation
  3c          Innovation study: No-Innovation vs. Fixed aspiration rise
  ratcheting  Ratcheting aspiration extension across all mindsets × strategies
  robustness  Robustness tests for the Stage 3C model (Variants A / B / C / P)

Usage
-----
  python run_study.py                              # all main stages, 30 reps (pilot)
  python run_study.py --full                       # publication rep counts
  python run_study.py --stage 3c                   # one stage only
  python run_study.py --stage robustness           # robustness tests only
  python run_study.py --stage 3c --alpha 0.25      # vary a parameter
  python run_study.py --stage 2b --n 16 --n-conflict 6
  python run_study.py --stage all --full --output my_results

Third-party requirements: numpy, matplotlib, pandas  (see requirements.txt)
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
# This script lives in ethiraj_levinthal/; the simulation code is in two
# sub-folders.  Both are added to sys.path so imports work from here.
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
EXTENSION_DIR  = os.path.join(SCRIPT_DIR, "Extension by stages")
ROBUSTNESS_DIR = os.path.join(SCRIPT_DIR, "robustness tests")

for _p in [EXTENSION_DIR, ROBUSTNESS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(N=24, G=8, G_b=4, M=4, R=120, N_CONFLICT=12, ALPHA=0.50, BETA=0.10)

DEFAULT_T = {'1b': 250, '2b': 250, '3c': 500, 'ratcheting': 3000, 'robustness': 500}

FULL_REPS = {'1b': 250, '2b': 250, '3c': 120, 'ratcheting': 250, 'robustness': 120}

# "all" runs the four main stages; robustness must be requested explicitly
MAIN_STAGES = ['1b', '2b', '3c', 'ratcheting']
ALL_STAGES  = MAIN_STAGES + ['robustness']


# ── Parameter utilities ───────────────────────────────────────────────────────

def _apply_params(module, args, t_override=None):
    """Monkey-patch module-level constants from CLI args (only supplied values)."""
    mapping = [
        ('n',          'N'),
        ('g',          'G'),
        ('g_b',        'G_b'),
        ('m',          'M'),
        ('r',          'R'),
        ('n_conflict', 'N_CONFLICT'),
        ('alpha',      'ALPHA'),
        ('beta',       'BETA'),
    ]
    for arg_attr, mod_attr in mapping:
        val = getattr(args, arg_attr, None)
        if val is not None and hasattr(module, mod_attr):
            setattr(module, mod_attr, val)
    if t_override is not None and hasattr(module, 'T'):
        module.T = t_override


def validate_params(args):
    """Validate parameter combinations and exit with a clear message on error."""
    errors = []
    g   = args.g          if args.g          is not None else DEFAULTS['G']
    g_b = args.g_b        if args.g_b        is not None else DEFAULTS['G_b']
    m   = args.m          if args.m          is not None else DEFAULTS['M']
    n   = args.n          if args.n          is not None else DEFAULTS['N']
    nc  = args.n_conflict if args.n_conflict is not None else DEFAULTS['N_CONFLICT']

    if g_b >= g:
        errors.append(
            f"--g-b ({g_b}) must be less than --g ({g}). "
            f"Ethical goals = G - G_b, which would be {g - g_b}.")
    if g % m != 0:
        errors.append(
            f"--g ({g}) must be divisible by --m ({m}) so each department "
            f"receives an equal number of goals.")
    if nc > n:
        errors.append(
            f"--n-conflict ({nc}) cannot exceed --n ({n}).")
    if args.alpha is not None and not (0.0 < args.alpha < 1.0):
        errors.append(f"--alpha must be strictly between 0 and 1. Got {args.alpha}.")
    if args.beta is not None and args.beta <= 0:
        errors.append(f"--beta must be positive. Got {args.beta}.")
    if args.n is not None and args.n < 4:
        errors.append(f"--n must be at least 4. Got {args.n}.")
    if args.t is not None and args.t < 10:
        errors.append(f"--t must be at least 10. Got {args.t}.")

    if errors:
        print("\nParameter error(s) — cannot run:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def _param_line(args, stage):
    """One-line summary of active parameters for the run banner."""
    t   = args.t          if args.t          is not None else DEFAULT_T[stage]
    n   = args.n          if args.n          is not None else DEFAULTS['N']
    g   = args.g          if args.g          is not None else DEFAULTS['G']
    g_b = args.g_b        if args.g_b        is not None else DEFAULTS['G_b']
    m   = args.m          if args.m          is not None else DEFAULTS['M']
    r   = args.r          if args.r          is not None else DEFAULTS['R']
    nc  = args.n_conflict if args.n_conflict is not None else DEFAULTS['N_CONFLICT']
    a   = args.alpha      if args.alpha      is not None else DEFAULTS['ALPHA']
    b   = args.beta       if args.beta       is not None else DEFAULTS['BETA']
    return (f"N={n}, G={g} (G_b={g_b}, G_e={g-g_b}), M={m}, R={r}, "
            f"N_conflict={nc}, α={a}, β={b}, T={t}")


# ── Core trajectory figure ────────────────────────────────────────────────────

_MINDSETS_ORDERED = ["wac", "sf", "compliance", "cei"]
_MINDSET_LABEL    = {"wac": "WAC", "sf": "SF", "compliance": "Compliance", "cei": "CEI"}
_STRATEGY_COLORS  = {
    "baseline": "#2980B9", "myopic": "#E67E22",
    "spatial":  "#27AE60", "temporal": "#8E44AD",
}
_STRATEGY_LABEL = {
    "baseline": "Baseline", "myopic": "Myopic",
    "spatial":  "Spatial",  "temporal": "Temporal",
}
_STRATEGIES_ORDERED = ["baseline", "myopic", "spatial", "temporal"]


def _fig_split_trajectories(cells, output_dir, filename, title):
    """
    Generate the fundamental 8-panel trajectory figure.

    Layout: 4 rows (WAC / SF / Compliance / CEI)
            × 2 columns (Business performance | Ethical performance)
    Lines:  one per evaluation strategy (4 per panel)
    Y-axis: identical across all 8 panels

    Parameters
    ----------
    cells : list of dicts, each with keys:
        mindset   (str)
        strategy  (str)
        biz_series (array-like, length T)
        eth_series (array-like, length T)
    output_dir : str — where to save the PNG
    filename   : str — PNG filename (no path)
    title      : str — figure suptitle
    """
    if not cells:
        return

    T_val = len(cells[0]["biz_series"])
    steps = np.arange(T_val)

    # Global y limits so all 8 panels share the same scale
    all_vals = []
    for c in cells:
        all_vals.extend(c["biz_series"])
        all_vals.extend(c["eth_series"])
    y_lo = max(0.0,  min(all_vals) - 0.01)
    y_hi = min(1.0,  max(all_vals) + 0.01)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.998)

    # Index cells by (mindset, strategy)
    idx = {(c["mindset"], c["strategy"]): c for c in cells}

    for row_i, mindset in enumerate(_MINDSETS_ORDERED):
        for col_j, (series_key, col_title, ylabel) in enumerate([
            ("biz_series", "Business performance", "Business performance (Ω_B)"),
            ("eth_series", "Ethical performance",  "Ethical performance (Ω_E)"),
        ]):
            ax = axes[row_i, col_j]

            for strategy in _STRATEGIES_ORDERED:
                c = idx.get((mindset, strategy))
                if c is None:
                    continue
                ts = np.asarray(c[series_key])
                ax.plot(steps, ts,
                        color=_STRATEGY_COLORS[strategy],
                        label=_STRATEGY_LABEL[strategy],
                        linewidth=1.3, alpha=0.9)

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(0, T_val - 1)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)

            if row_i == 0:
                ax.set_title(col_title, fontsize=11, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(
                    _MINDSET_LABEL[mindset] + "\n" + ylabel, fontsize=10)
            else:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_i == 3:
                ax.set_xlabel("Time step", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Evaluation strategy", loc="lower center",
               ncol=4, fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _cells_from_s3c_results(results, has_innovation=True, innovation_type="fixed"):
    """Build the cells list from run_all_cells() results (averages across reps)."""
    buckets = defaultdict(lambda: {"biz": [], "eth": []})
    for r in results:
        if r.get("has_innovation") == has_innovation \
                and r.get("innovation_type") == innovation_type:
            key = (r["mindset"], r["strategy"])
            buckets[key]["biz"].append(r["omega_b_series"])
            buckets[key]["eth"].append(r["omega_e_series"])
    cells = []
    for (mindset, strategy), data in buckets.items():
        if data["biz"]:
            cells.append({
                "mindset":    mindset,
                "strategy":   strategy,
                "biz_series": np.mean(data["biz"], axis=0).tolist(),
                "eth_series": np.mean(data["eth"], axis=0).tolist(),
            })
    return cells


def _cells_from_rb_timeseries(ts_rows, variant_key):
    """Build the cells list from rb_timeseries.csv rows for one variant."""
    buckets = defaultdict(lambda: {"biz": [], "eth": [], "t": []})
    for r in ts_rows:
        if r.get("variant") == variant_key and r.get("has_innovation") == "1":
            key = (r["mindset"], r["strategy"])
            buckets[key]["t"].append(int(r["time_step"]))
            buckets[key]["biz"].append(float(r.get("biz_omega", r.get("avg_omega", 0))))
            buckets[key]["eth"].append(float(r.get("eth_omega", r.get("avg_omega", 0))))
    cells = []
    for (mindset, strategy), data in buckets.items():
        order = np.argsort(data["t"])
        cells.append({
            "mindset":    mindset,
            "strategy":   strategy,
            "biz_series": [data["biz"][i] for i in order],
            "eth_series": [data["eth"][i] for i in order],
        })
    return cells


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage_1b(reps, output_dir, args):
    import s1b_experiment
    t = args.t if args.t is not None else DEFAULT_T['1b']
    _apply_params(s1b_experiment, args, t_override=t)
    s1b_experiment.main(reps=reps, output_dir=output_dir)


def run_stage_2b(reps, output_dir, args):
    import s2b_experiment
    t = args.t if args.t is not None else DEFAULT_T['2b']
    _apply_params(s2b_experiment, args, t_override=t)
    s2b_experiment.main(reps=reps, output_dir=output_dir)

    ts_csv = os.path.join(output_dir, "s2b_timeseries.csv")
    if os.path.exists(ts_csv):
        print("  Generating paper trajectory figure...")
        import make_s2b_fig2
        _apply_params(make_s2b_fig2, args, t_override=t)
        make_s2b_fig2.make_paper_figure(data_dir=output_dir)


def run_stage_3c(reps, output_dir, args):
    import s3c_experiment
    t    = args.t    if args.t    is not None else DEFAULT_T['3c']
    beta = args.beta if args.beta is not None else s3c_experiment.BETA
    _apply_params(s3c_experiment, args, t_override=t)

    total_cells = (sum(len(s3c_experiment.INNOV_CONDITIONS[m])
                       for m in s3c_experiment.MINDSETS)
                   * len(s3c_experiment.STRATEGIES))
    print(f"  {reps} reps × {total_cells} cells = {reps * total_cells:,} total replications")

    results = s3c_experiment.run_all_cells(reps, output_dir, beta=beta)
    s3c_experiment.print_pilot_summary(results)
    print("  Saving CSV and generating figures...")
    s3c_experiment.save_csv(results, output_dir)
    s3c_experiment.fig_frontier(results, output_dir)
    s3c_experiment.fig_trajectories(results, output_dir)
    s3c_experiment.fig_n_growth(results, output_dir)
    s3c_experiment.fig_innovation_counts(results, output_dir)
    s3c_experiment.fig_diff(results, output_dir)
    s3c_experiment.fig_compliance(results, output_dir)

    # Core 8-panel trajectory figure — always generated
    print("  Generating trajectory figures...")
    cells = _cells_from_s3c_results(results, has_innovation=True, innovation_type="fixed")
    t_val = args.t if args.t is not None else DEFAULT_T['3c']
    _fig_split_trajectories(
        cells, output_dir,
        "s3c_fig_trajectories_biz_eth.png",
        f"Stage 3C — Business and ethical performance trajectories (+I Fixed, T={t_val})",
    )

    ts_csv = os.path.join(output_dir, "s3c_timeseries.csv")
    if os.path.exists(ts_csv):
        print("  Generating paper trajectory figure...")
        from make_s3c_fig2 import make_paper_figure
        make_paper_figure(data_dir=output_dir)


def run_stage_ratcheting(reps, output_dir, args):
    import ratcheting_experiment
    t = args.t if args.t is not None else DEFAULT_T['ratcheting']
    _apply_params(ratcheting_experiment, args, t_override=t)
    ratcheting_experiment.main(reps=reps, output_dir=output_dir)


def run_stage_robustness(reps, output_dir, args):
    """
    Robustness tests for the Stage 3C model.

    Runs the pre-specified variant set, the high-β WAC supplementary, and
    then generates all robustness figures in output_dir/.

    Notes
    -----
    - Robustness tests always use the publication parameter set (N=24, T=500,
      etc.).  Landscape parameter flags (--n, --g, etc.) are ignored here by
      design: robustness tests are meant to vary the innovation procedure, not
      the landscape.
    - Use --rb-variant to select which tests to run (default: all main variants).
    """
    import rb_experiment
    import rb_figures

    # Map --rb-variant to the list of variant keys rb_experiment understands
    variant_map = {
        'all': ["baseline", "A75", "A50", "B", "C005", "C010"],
        'A':   ["baseline", "A75", "A50"],
        'B':   ["baseline", "B"],
        'C':   ["baseline", "C005", "C010"],
        'D':   ["baseline", "D_K3_005", "D_K3_010", "D_K5_005", "D_K5_010"],
        'E':   ["baseline", "E"],
        'P':   ["baseline", "P10", "P20"],
    }
    rb_variant = args.rb_variant if hasattr(args, 'rb_variant') and args.rb_variant else 'all'
    keys = variant_map.get(rb_variant, ["baseline", "A75", "A50", "B", "C005", "C010"])

    print(f"  Variants: {keys}  +  high-β WAC supplementary")
    rb_experiment.run_variants(keys, reps, output_dir)
    rb_experiment.run_highbeta_wac(reps, output_dir)

    # Load CSVs for figure generation
    per_rep_path = os.path.join(output_dir, "rb_per_rep.csv")
    hb_path      = os.path.join(output_dir, "rb_highbeta_wac.csv")
    ev_path      = os.path.join(output_dir, "rb_event_study.csv")

    if not os.path.exists(per_rep_path):
        print("  [Warning: rb_per_rep.csv not found — figures skipped]")
        return

    with open(per_rep_path, newline="") as f:
        per_rep = list(csv.DictReader(f))
    hb_rows = []
    if os.path.exists(hb_path):
        with open(hb_path, newline="") as f:
            hb_rows = list(csv.DictReader(f))
    ev_rows = []
    if os.path.exists(ev_path):
        with open(ev_path, newline="") as f:
            ev_rows = list(csv.DictReader(f))

    print("  Generating robustness figures...")
    rb_figures.print_ordering_check(per_rep)

    variants_in_data = set(r["variant"] for r in per_rep)
    for vk in rb_experiment.VARIANTS:
        if vk in variants_in_data:
            rb_figures.fig_frontier(per_rep, vk, output_dir)

    rb_figures.fig_compliance(per_rep, output_dir)
    rb_figures.fig_eth_ordering(per_rep, output_dir)
    rb_figures.fig_innov_counts(per_rep, output_dir)
    rb_figures.fig_eth_by_p(per_rep, output_dir)
    rb_figures.fig_compliance_by_p(per_rep, output_dir)
    rb_figures.fig_biz_by_p(per_rep, output_dir)
    rb_figures.fig_recovery_event_study(ev_rows, output_dir)
    rb_figures.fig_highbeta_wac(hb_rows, per_rep, output_dir)

    # Core 8-panel trajectory figures — always generated for every variant run
    if os.path.exists(os.path.join(output_dir, "rb_timeseries.csv")):
        with open(os.path.join(output_dir, "rb_timeseries.csv"), newline="") as f:
            ts_rows = list(csv.DictReader(f))
        for vk in keys:
            cells = _cells_from_rb_timeseries(ts_rows, vk)
            if cells:
                slug  = vk.lower()
                label = rb_experiment.VARIANT_LABEL.get(vk, vk)
                _fig_split_trajectories(
                    cells, output_dir,
                    f"rb_fig_trajectories_biz_eth_{slug}.png",
                    f"Robustness — Business and ethical performance ({label}, +I Fixed, T={DEFAULT_T['robustness']})",
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ethiraj-Levinthal NK Landscape Study — unified runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
STAGES
  1b          Goal-blind replication of Ethiraj & Levinthal (2003)
  2b          Mindset comparison (WAC / SF / Compliance / CEI), no innovation
  3c          Innovation study (No-Innovation vs. Fixed aspiration rise)
  ratcheting  Ratcheting aspiration extension — all mindsets × strategies
  robustness  Robustness tests for the Stage 3C model (Variants A / B / C / P)
  all         Runs 1b + 2b + 3c + ratcheting (robustness must be run separately)

EXAMPLES
  python run_study.py                                  # pilot run, all main stages
  python run_study.py --full                           # publication-quality run
  python run_study.py --stage 3c --reps 50
  python run_study.py --stage 2b --n 16 --n-conflict 6
  python run_study.py --stage 3c --alpha 0.25 --beta 0.20
  python run_study.py --stage ratcheting --t 5000
  python run_study.py --stage robustness --full
  python run_study.py --stage robustness --rb-variant P --reps 120
  python run_study.py --full --output sensitivity/alpha_low --alpha 0.30

PARAMETER DEFAULTS (original published study)
  N=24  G=8 (G_b=4, G_e=4)  M=4  R=120  N_conflict=12  α=0.50  β=0.10
  T per stage: 1b=250, 2b=250, 3c=500, ratcheting=3000
        """,
    )

    # ── Run control ───────────────────────────────────────────────────────────
    run_grp = ap.add_argument_group('Run control')
    run_grp.add_argument('--stage', choices=['all'] + ALL_STAGES, default='all',
                         help='Stage(s) to run. "all" runs the four main stages '
                              '(robustness must be requested separately). Default: all')
    run_grp.add_argument('--reps', type=int, default=30,
                         help='Replications per cell (default: 30). Ignored with --full.')
    run_grp.add_argument('--full', action='store_true',
                         help='Use publication rep counts: 1b/2b/ratcheting=250, 3c/robustness=120')
    run_grp.add_argument('--output', default='results',
                         help='Base output directory (default: results/)')

    # ── Organization structure ────────────────────────────────────────────────
    org = ap.add_argument_group(
        'Organization structure',
        'Controls the size and shape of the simulated organization. '
        'Original values: N=24, G=8, G_b=4, M=4, R=120. '
        'NOTE: ignored for --stage robustness (robustness tests always use '
        'the original parameter set).')
    org.add_argument('--n', type=int, default=None, metavar='INT',
                     help='Total decision bits — overall organization complexity (default: 24)')
    org.add_argument('--g', type=int, default=None, metavar='INT',
                     help='Total number of goals, business + ethical (default: 8)')
    org.add_argument('--g-b', type=int, default=None, dest='g_b', metavar='INT',
                     help='Business goals; ethical goals = G - G_b (default: 4)')
    org.add_argument('--m', type=int, default=None, metavar='INT',
                     help='Departments — must divide evenly into G (default: 4)')
    org.add_argument('--r', type=int, default=None, metavar='INT',
                     help='Interdependence radius — higher = rougher NK landscape (default: 120)')

    # ── Landscape conflict ────────────────────────────────────────────────────
    conf = ap.add_argument_group(
        'Landscape conflict',
        'Controls how many decisions are shared between business and ethical goals. '
        'Original value: N_conflict=12.')
    conf.add_argument('--n-conflict', type=int, default=None, dest='n_conflict', metavar='INT',
                      help='Decisions that simultaneously affect both goal types (default: 12)')

    # ── Search acceptance ─────────────────────────────────────────────────────
    srch = ap.add_argument_group(
        'Search acceptance',
        'Controls how accepting or demanding the organization is when evaluating '
        'candidate moves. Original value: α=0.50.')
    srch.add_argument('--alpha', type=float, default=None, metavar='FLOAT',
                      help='Conflict magnitude: strength of negative contributions from '
                           'conflicted decisions (0–1, default: 0.50)')

    # ── Innovation / aspiration ───────────────────────────────────────────────
    innov = ap.add_argument_group(
        'Innovation and aspiration (Stages 3C and Ratcheting)',
        'Controls how ambitiously the organization raises its performance targets '
        'when innovation is enabled. Original value: β=0.10.')
    innov.add_argument('--beta', type=float, default=None, metavar='FLOAT',
                       help='Aspiration stretch: how far above current performance the '
                            'innovation trigger is set (default: 0.10, i.e. 10%% above)')

    # ── Robustness options ────────────────────────────────────────────────────
    rb = ap.add_argument_group(
        'Robustness options (--stage robustness only)',
        'Selects which robustness variant(s) to run. All variants use the '
        'original publication parameter set.')
    rb.add_argument('--rb-variant', default='all', dest='rb_variant',
                    choices=['all', 'A', 'B', 'C', 'D', 'E', 'P'],
                    help=('Which robustness test(s) to run:\n'
                          '  all  — Variants A (π=0.75, 0.50), B (symmetric), '
                          'C (δ=0.005, 0.010) [default]\n'
                          '  A    — Stochastic success only (π=0.75 and 0.50)\n'
                          '  B    — Symmetric rewiring only\n'
                          '  C    — Innovation cost only (δ=0.005 and 0.010)\n'
                          '  D    — Sustained cost (K=3,5 × δ=0.005,0.010)\n'
                          '  E    — Post-innovation search suspension (hold L~U{0,10})\n'
                          '  P    — Group-B supplementary (p=0.10 and 0.20)'))

    # ── Time horizon ──────────────────────────────────────────────────────────
    t_grp = ap.add_argument_group(
        'Time horizon',
        'Overrides the stage-specific default. Stage defaults: 1b=250, 2b=250, '
        '3c=500, ratcheting=3000.')
    t_grp.add_argument('--t', type=int, default=None, metavar='INT',
                       help='Time steps per replication (overrides stage default)')

    args = ap.parse_args()
    validate_params(args)

    stages = MAIN_STAGES if args.stage == 'all' else [args.stage]

    print(f"\n{'='*65}")
    print("  Ethiraj-Levinthal NK Landscape Study")
    print(f"  Stages : {', '.join(stages)}")
    if args.full:
        print(f"  Mode   : FULL RUN  "
              f"({', '.join(f'{s}={FULL_REPS[s]}' for s in stages)} reps per cell)")
    else:
        print(f"  Mode   : pilot  ({args.reps} reps per cell)")
    print(f"  Output : {os.path.abspath(args.output)}/")
    print(f"{'='*65}")

    t_total = time.time()

    for stage in stages:
        reps = FULL_REPS[stage] if args.full else args.reps
        output_dir = os.path.join(args.output, stage)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'─'*65}")
        print(f"  Stage {stage.upper()}  —  {reps} reps  →  {output_dir}/")
        if stage != 'robustness':
            print(f"  {_param_line(args, stage)}")
        else:
            print(f"  Variant: {args.rb_variant}  (fixed parameters: N=24, T=500, α=0.50, β=0.10)")
        print(f"{'─'*65}")
        t0 = time.time()

        try:
            if stage == '1b':
                run_stage_1b(reps, output_dir, args)
            elif stage == '2b':
                run_stage_2b(reps, output_dir, args)
            elif stage == '3c':
                run_stage_3c(reps, output_dir, args)
            elif stage == 'ratcheting':
                run_stage_ratcheting(reps, output_dir, args)
            elif stage == 'robustness':
                run_stage_robustness(reps, output_dir, args)
        except Exception as exc:
            print(f"\n  *** Stage {stage} failed: {exc}")
            import traceback
            traceback.print_exc()
            print("  Continuing with remaining stages...")
            continue

        print(f"\n  Stage {stage} complete in {time.time()-t0:.0f}s — outputs in {output_dir}/")

    print(f"\n{'='*65}")
    print(f"  All done in {time.time()-t_total:.0f}s.")
    print(f"  Results: {os.path.abspath(args.output)}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
