"""
Robustness figures — reads rb_per_rep.csv, rb_highbeta_wac.csv, and
rb_event_study.csv; produces all figures.

Usage
-----
  python rb_figures.py                  # reads from same directory
  python rb_figures.py --data ./        # explicit data directory
"""
import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── colours / labels ──────────────────────────────────────────────────────────
MINDSET_COLORS = {
    "wac":        "#C0392B",
    "sf":         "#E67E22",
    "compliance": "#2980B9",
    "cei":        "#27AE60",
}
MINDSET_LABEL = {
    "wac": "WAC", "sf": "SF", "compliance": "Comp.", "cei": "CEI",
}
STRAT_MARKERS = {
    "baseline": "o", "myopic": "s", "spatial": "^", "temporal": "D",
}
STRAT_LABEL = {
    "baseline": "Baseline", "myopic": "Myopic",
    "spatial":  "Spatial",  "temporal": "Temporal",
}
VARIANT_TITLE = {
    "baseline": "Baseline (primary)",
    "A75":      "Variant A  π = 0.75",
    "A50":      "Variant A  π = 0.50",
    "B":        "Variant B (symmetric)",
    "C005":     "Variant C  δ = 0.005",
    "C010":     "Variant C  δ = 0.010",
    "P10":      "Group-B  p = 0.10",
    "P20":      "Group-B  p = 0.20",
}

MINDSETS   = ["wac", "sf", "compliance", "cei"]
STRATEGIES = ["baseline", "myopic", "spatial", "temporal"]

VARIANTS_ORDERED = ["baseline", "A75", "A50", "B", "C005", "C010", "P10", "P20"]

# ── data loading ──────────────────────────────────────────────────────────────

def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def _float(row, key, default=float("nan")):
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ── original figures (variants A / B / C) ────────────────────────────────────

def fig_frontier(rows, variant_key, output_dir):
    subset = [r for r in rows
              if r.get("variant") == variant_key and r.get("has_innovation") == "1"]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for mindset in MINDSETS:
        for strategy in STRATEGIES:
            pts = [r for r in subset
                   if r["mindset"] == mindset and r["strategy"] == strategy]
            if not pts:
                continue
            biz = np.mean([_float(r, "final_biz_omega") for r in pts])
            eth = np.mean([_float(r, "final_eth_omega") for r in pts])
            ax.scatter(biz, eth, color=MINDSET_COLORS[mindset],
                       marker=STRAT_MARKERS[strategy], s=60, alpha=0.85, zorder=3)

    mindset_patches = [mpatches.Patch(color=MINDSET_COLORS[m], label=MINDSET_LABEL[m])
                       for m in MINDSETS]
    strat_handles   = [plt.Line2D([0], [0], marker=STRAT_MARKERS[s], color="grey",
                                   linestyle="None", label=STRAT_LABEL[s], markersize=7)
                       for s in STRATEGIES]
    ax.legend(handles=mindset_patches + strat_handles, fontsize=8,
              loc="lower right", ncol=2)
    ax.set_xlabel("Business performance (Ω_B)")
    ax.set_ylabel("Ethical performance (Ω_E)")
    ax.set_title(f"Frontier — {VARIANT_TITLE.get(variant_key, variant_key)}")
    fig.tight_layout()
    slug = variant_key.lower().replace(".", "")
    path = os.path.join(output_dir, f"rb_fig_frontier_{slug}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_compliance(rows, output_dir):
    present = sorted(set(r["variant"] for r in rows if r.get("has_innovation") == "1"),
                     key=lambda v: VARIANTS_ORDERED.index(v) if v in VARIANTS_ORDERED else 99)
    if not present:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x, n_var, width = np.arange(len(MINDSETS)), len(present), 0.7 / len(present)
    for vi, vk in enumerate(present):
        subset = [r for r in rows if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = [float(np.mean([_float(r, "compliance_rate") for r in subset
                                 if r["mindset"] == m]) or float("nan"))
                  for m in MINDSETS]
        ax.bar(x + (vi - (n_var-1)/2)*width, means, width*0.9,
               label=VARIANT_TITLE.get(vk, vk), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Compliance rate"); ax.set_title("Compliance rate across robustness variants (+I)")
    ax.legend(fontsize=7, loc="upper left", ncol=2); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_compliance_all.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_eth_ordering(rows, output_dir):
    present = sorted(set(r["variant"] for r in rows if r.get("has_innovation") == "1"),
                     key=lambda v: VARIANTS_ORDERED.index(v) if v in VARIANTS_ORDERED else 99)
    if not present:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x, n_var, width = np.arange(len(MINDSETS)), len(present), 0.7 / len(present)
    for vi, vk in enumerate(present):
        subset = [r for r in rows if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = [float(np.mean([_float(r, "final_eth_omega") for r in subset
                                 if r["mindset"] == m]) or float("nan"))
                  for m in MINDSETS]
        ax.bar(x + (vi - (n_var-1)/2)*width, means, width*0.9,
               label=VARIANT_TITLE.get(vk, vk), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Final ethical performance (Ω_E)")
    ax.set_title("Ethical performance ordering across robustness variants (+I)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_eth_ordering.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_innov_counts(rows, output_dir):
    present = sorted(set(r["variant"] for r in rows if r.get("has_innovation") == "1"),
                     key=lambda v: VARIANTS_ORDERED.index(v) if v in VARIANTS_ORDERED else 99)
    if not present:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x, n_var, width = np.arange(len(MINDSETS)), len(present), 0.7 / len(present)
    for vi, vk in enumerate(present):
        subset = [r for r in rows if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = [float(np.mean([_float(r, "innov_count") for r in subset
                                 if r["mindset"] == m]) or float("nan"))
                  for m in MINDSETS]
        ax.bar(x + (vi - (n_var-1)/2)*width, means, width*0.9,
               label=VARIANT_TITLE.get(vk, vk), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Mean innovations per replication")
    ax.set_title("Innovation counts across robustness variants (+I)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_innov_counts.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_highbeta_wac(hb_rows, per_rep_rows, output_dir):
    if not hb_rows:
        return
    betas = sorted(set(float(r.get("beta", 0.10)) for r in hb_rows))
    cei_ref_eth  = [_float(r, "final_eth_omega") for r in per_rep_rows
                    if r.get("variant") == "baseline" and r.get("mindset") == "cei"
                    and r.get("has_innovation") == "1"]
    cei_ref_comp = [_float(r, "compliance_rate") for r in per_rep_rows
                    if r.get("variant") == "baseline" and r.get("mindset") == "cei"
                    and r.get("has_innovation") == "1"]
    cei_eth  = float(np.mean(cei_ref_eth))  if cei_ref_eth  else float("nan")
    cei_comp = float(np.mean(cei_ref_comp)) if cei_ref_comp else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ylabel, title, cei_val in [
        (axes[0], "final_eth_omega", "Final ethical performance (Ω_E)",
         "Ethical performance by β — WAC (+I)", cei_eth),
        (axes[1], "compliance_rate", "Compliance rate",
         "Compliance rate by β — WAC (+I)", cei_comp),
        (axes[2], "innov_count",     "Mean innovations per replication",
         "Innovation count by β — WAC (+I)", None),
    ]:
        means, sems = [], []
        for beta in betas:
            pts = [_float(r, metric) for r in hb_rows
                   if abs(float(r.get("beta", 0)) - beta) < 1e-9]
            means.append(float(np.mean(pts)) if pts else float("nan"))
            sems.append(float(np.std(pts)/np.sqrt(len(pts))) if len(pts) > 1 else 0.0)
        ax.errorbar(betas, means, yerr=sems, marker="o", color="#C0392B",
                    capsize=4, linewidth=1.8, label="WAC")
        if cei_val is not None and not np.isnan(cei_val):
            ax.axhline(cei_val, linestyle="--", color="#27AE60", linewidth=1.4,
                       label="CEI β=0.10")
            ax.legend(fontsize=9)
        ax.set_xlabel("β"); ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xticks(betas)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_highbeta_wac.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── new Group-B supplementary figures ────────────────────────────────────────

def fig_eth_by_p(rows, output_dir):
    """Fig S1 — Final ethical performance by mindset for p = 0, 0.10, 0.20."""
    p_variants = ["baseline", "P10", "P20"]
    p_labels   = ["p = 0 (baseline)", "p = 0.10", "p = 0.20"]
    present    = [vk for vk in p_variants
                  if any(r.get("variant") == vk and r.get("has_innovation") == "1"
                         for r in rows)]
    if not present:
        print("  [fig_eth_by_p] no P-variant data — skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(MINDSETS))
    n_p    = len(present)
    width  = 0.7 / n_p
    colors = ["#7F8C8D", "#E67E22", "#C0392B"]

    for vi, vk in enumerate(present):
        subset = [r for r in rows
                  if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = []
        sems   = []
        for mindset in MINDSETS:
            pts = [_float(r, "final_eth_omega") for r in subset
                   if r["mindset"] == mindset]
            means.append(float(np.mean(pts)) if pts else float("nan"))
            sems.append(float(np.std(pts)/np.sqrt(len(pts))) if len(pts) > 1 else 0.0)
        offset = (vi - (n_p-1)/2) * width
        lbl = p_labels[p_variants.index(vk)] if vk in p_variants else vk
        ax.bar(x + offset, means, width*0.9, label=lbl,
               color=colors[vi % len(colors)], alpha=0.85,
               yerr=sems, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Final ethical performance (Ω_E)")
    ax.set_title("Fig S1 — Ethical performance by mindset and Group-B probability (+I)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_s1_eth_by_p.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_compliance_by_p(rows, output_dir):
    """Fig S3 — Compliance rate by mindset for p = 0, 0.10, 0.20."""
    p_variants = ["baseline", "P10", "P20"]
    p_labels   = ["p = 0 (baseline)", "p = 0.10", "p = 0.20"]
    present    = [vk for vk in p_variants
                  if any(r.get("variant") == vk and r.get("has_innovation") == "1"
                         for r in rows)]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x, n_p, width = np.arange(len(MINDSETS)), len(present), 0.7 / len(present)
    colors = ["#7F8C8D", "#E67E22", "#C0392B"]

    for vi, vk in enumerate(present):
        subset = [r for r in rows
                  if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = []
        sems   = []
        for mindset in MINDSETS:
            pts = [_float(r, "compliance_rate") for r in subset
                   if r["mindset"] == mindset]
            means.append(float(np.mean(pts)) if pts else float("nan"))
            sems.append(float(np.std(pts)/np.sqrt(len(pts))) if len(pts) > 1 else 0.0)
        offset = (vi - (n_p-1)/2) * width
        lbl = p_labels[p_variants.index(vk)] if vk in p_variants else vk
        ax.bar(x + offset, means, width*0.9, label=lbl,
               color=colors[vi % len(colors)], alpha=0.85,
               yerr=sems, capsize=3)

    ax.set_xticks(x); ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Compliance rate (fraction of steps all θ_E met)")
    ax.set_title("Fig S3 — Compliance rate by mindset and Group-B probability (+I)")
    ax.set_ylim(0, 1); ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_s3_compliance_by_p.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_biz_by_p(rows, output_dir):
    """Fig S4 — Final business performance by mindset for p = 0, 0.10, 0.20."""
    p_variants = ["baseline", "P10", "P20"]
    p_labels   = ["p = 0 (baseline)", "p = 0.10", "p = 0.20"]
    present    = [vk for vk in p_variants
                  if any(r.get("variant") == vk and r.get("has_innovation") == "1"
                         for r in rows)]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x, n_p, width = np.arange(len(MINDSETS)), len(present), 0.7 / len(present)
    colors = ["#7F8C8D", "#E67E22", "#C0392B"]

    for vi, vk in enumerate(present):
        subset = [r for r in rows
                  if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means, sems = [], []
        for mindset in MINDSETS:
            pts = [_float(r, "final_biz_omega") for r in subset
                   if r["mindset"] == mindset]
            means.append(float(np.mean(pts)) if pts else float("nan"))
            sems.append(float(np.std(pts)/np.sqrt(len(pts))) if len(pts) > 1 else 0.0)
        offset = (vi - (n_p-1)/2) * width
        lbl = p_labels[p_variants.index(vk)] if vk in p_variants else vk
        ax.bar(x + offset, means, width*0.9, label=lbl,
               color=colors[vi % len(colors)], alpha=0.85,
               yerr=sems, capsize=3)

    ax.set_xticks(x); ax.set_xticklabels([MINDSET_LABEL[m] for m in MINDSETS])
    ax.set_ylabel("Final business performance (Ω_B)")
    ax.set_title("Fig S4 — Business performance by mindset and Group-B probability (+I)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_s4_biz_by_p.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def fig_recovery_event_study(ev_rows, output_dir):
    """
    Fig S2 — Ethical performance in the 61-step window around Group-B events.

    One panel per p-value (0.10, 0.20); within each panel, one line per
    mindset.  Time 0 = step of the Group-B innovation event.  The pre-event
    mean (offsets -30 to -1) anchors the baseline; post-event trajectory
    shows divergence by acceptance rule.
    """
    if not ev_rows:
        print("  [fig_recovery_event_study] no event-study data — skipping")
        return

    p_values  = sorted(set(float(r.get("p_value", 0)) for r in ev_rows if r.get("p_value")))
    offsets   = sorted(set(int(r.get("time_offset", 0)) for r in ev_rows))
    if not p_values or not offsets:
        return

    n_panels  = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, p_val in zip(axes, p_values):
        for mindset in MINDSETS:
            means, sems = [], []
            for offset in offsets:
                pts = [_float(r, "mean_eth_perf") for r in ev_rows
                       if abs(float(r.get("p_value", 0)) - p_val) < 1e-9
                       and r.get("mindset") == mindset
                       and int(r.get("time_offset", -999)) == offset
                       and r.get("mean_eth_perf") != ""]
                means.append(float(np.mean(pts)) if pts else float("nan"))
                sems.append(float(np.std(pts)/np.sqrt(len(pts)))
                            if len(pts) > 1 else 0.0)

            means = np.array(means)
            sems  = np.array(sems)
            ax.plot(offsets, means, color=MINDSET_COLORS[mindset],
                    label=MINDSET_LABEL[mindset], linewidth=1.8)
            ax.fill_between(offsets, means - sems, means + sems,
                            color=MINDSET_COLORS[mindset], alpha=0.15)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvspan(-30, -0.5, color="grey", alpha=0.06)
        ax.set_xlabel("Steps relative to Group-B innovation event")
        ax.set_ylabel("Mean ethical performance (Ω_E)")
        ax.set_title(f"Fig S2 — Recovery after Group-B innovation  (p = {p_val})")
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "rb_fig_s2_recovery_event_study.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── ordering check console summary ───────────────────────────────────────────

def print_ordering_check(rows):
    present = sorted(set(r["variant"] for r in rows if r.get("has_innovation") == "1"),
                     key=lambda v: VARIANTS_ORDERED.index(v) if v in VARIANTS_ORDERED else 99)
    print("\n  ── Ethical performance ordering (mean across strategies, +I) ──")
    print(f"  {'Variant':<24} {'WAC':>6} {'SF':>6} {'Comp.':>6} {'CEI':>6}  Pass?")
    for vk in present:
        subset = [r for r in rows
                  if r.get("variant") == vk and r.get("has_innovation") == "1"]
        means  = {}
        for mindset in MINDSETS:
            pts = [_float(r, "final_eth_omega") for r in subset
                   if r["mindset"] == mindset]
            means[mindset] = float(np.mean(pts)) if pts else float("nan")
        order  = [means["wac"], means["sf"], means["compliance"], means["cei"]]
        passed = all(order[i] <= order[i+1] + 1e-6 for i in range(len(order)-1))
        flag   = "PASS" if passed else "** FAIL **"
        print(f"  {VARIANT_TITLE.get(vk, vk):<24} "
              f"{means['wac']:>6.4f} {means['sf']:>6.4f} "
              f"{means['compliance']:>6.4f} {means['cei']:>6.4f}  {flag}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Robustness figures")
    ap.add_argument("--data",   default=_HERE)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    output_dir = args.output or args.data
    os.makedirs(output_dir, exist_ok=True)

    rows    = _load_csv(os.path.join(args.data, "rb_per_rep.csv"))
    hb_rows = _load_csv(os.path.join(args.data, "rb_highbeta_wac.csv"))
    ev_rows = _load_csv(os.path.join(args.data, "rb_event_study.csv"))

    if not rows and not hb_rows:
        print(f"No data found in {args.data}/ — run rb_experiment.py first.")
        return

    print(f"\nLoaded {len(rows)} per-rep rows, {len(hb_rows)} high-β rows, "
          f"{len(ev_rows)} event-study rows.")

    if rows:
        print_ordering_check(rows)
        print("\n  Generating figures...")
        variants_in_data = set(r["variant"] for r in rows)
        for vk in VARIANTS_ORDERED:
            if vk in variants_in_data:
                fig_frontier(rows, vk, output_dir)
        fig_compliance(rows, output_dir)
        fig_eth_ordering(rows, output_dir)
        fig_innov_counts(rows, output_dir)
        # Group-B supplementary figures
        fig_eth_by_p(rows, output_dir)
        fig_compliance_by_p(rows, output_dir)
        fig_biz_by_p(rows, output_dir)

    fig_recovery_event_study(ev_rows, output_dir)

    if hb_rows or rows:
        fig_highbeta_wac(hb_rows, rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
