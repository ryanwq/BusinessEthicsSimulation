"""
Robustness simulation wrapper for Stage 3C.

Wraps run_replication_s3c() from the primary model with four innovation
variants described in the robustness spec:

  Variant A — Stochastic success (innov_pi):
    With probability 1-pi the innovation "fails": g*-contributions are drawn
    from U(0, 1) instead of U(0.5, 1.0).

  Variant B — Symmetric rewiring (innov_symmetric):
    All draws use U(0, 1) — no upward bias anywhere.

  Variant C — Explicit innovation cost (innov_cost):
    Each innovation step reduces the recorded performance by innov_cost.

  Supplementary — Stochastic Group-B assignment (innov_tradeoff_prob):
    With probability p, a non-WAC mindset's innovation is assigned to
    Group B (business-favoring) instead of Group N (neutral), using the
    same partition-landscape logic as model initialization.  The new
    decision's contributions to non-g* ethical goals are drawn from
    U(-alpha, 0) rather than U(0, 1).

Implementation:
  Module-level _RB_FLAGS dict is set by run_replication_rb before each
  replication and read inside _innovate_rb.  A _group_b_log list records
  one entry per innovation event: (is_group_b, g_star_is_ethical).
  run_replication_rb returns (s3c_result_tuple, group_b_log).
"""

import sys
import os
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.join(_HERE, "..", "Extension by stages")
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s3c_simulation as _s3c
from landscape import contribution_index

# ── module-level state ────────────────────────────────────────────────────────
_RB_FLAGS = {
    "innov_pi":            1.0,
    "innov_symmetric":     False,
    "innov_cost":          0.0,
    "innov_cost_periods":  1,     # Variant D: consecutive periods to apply cost
    "innov_hold_max":      0,     # Variant E: max hold steps drawn from U{0,...,max}
    "innov_tradeoff_prob": 0.0,
    "_group_b_log":        [],    # list of (is_group_b, g_star_is_ethical) per event
    "_hold_counter":       0,     # Variant E: steps remaining in current hold
}


# ── patched _innovate ─────────────────────────────────────────────────────────

def _innovate_rb(org, g_star, mindset, alpha, target_b, target_e, target_any, rng):
    """
    Drop-in replacement for s3c_simulation._innovate.

    Applies three draw-site modifications controlled by _RB_FLAGS:
      - Success/failure draw (Variant A / B)
      - Group-B assignment for non-WAC mindsets (Supplementary)

    Also records each event in _RB_FLAGS["_group_b_log"] as
    (is_group_b, g_star_is_ethical) for post-hoc event-study analysis.
    """
    pi        = _RB_FLAGS["innov_pi"]
    symmetric = _RB_FLAGS["innov_symmetric"]
    p_tradeoff = _RB_FLAGS["innov_tradeoff_prob"]

    # ── Variant A / B: success draw ───────────────────────────────────────
    success = True
    if symmetric:
        success = False
    elif pi < 1.0:
        success = (rng.random() < pi)

    # ── Supplementary: stochastic Group-B assignment ──────────────────────
    # WAC is always Group B (unchanged). Non-WAC mindsets are Group N by
    # default; with probability p_tradeoff they become Group B instead.
    if mindset == "wac":
        force_group_b = True
    elif p_tradeoff > 0.0 and rng.random() < p_tradeoff:
        force_group_b = True
    else:
        force_group_b = False

    g_star_is_ethical = bool(org.goal_types[g_star] == 1)
    _RB_FLAGS["_group_b_log"].append((force_group_b, g_star_is_ethical))

    # ── structural innovation (mirrors s3c_simulation._innovate exactly) ──
    G     = org.G
    N_old = org.N
    d_new = N_old

    org.N += 1
    dept_idx = int(rng.integers(0, org.M))
    org.departments[dept_idx].append(d_new)
    org.dept_sets[dept_idx].add(d_new)

    org.decisions    = np.append(org.decisions, np.int8(0))
    # Group assignment: B (1) if WAC or stochastic tradeoff; N (0) otherwise
    new_grp = np.int8(1) if force_group_b else np.int8(0)
    org.group_assign = np.append(org.group_assign, new_grp)

    org.influencers.append([])
    org.influencers_set.append(set())
    org.dependents.append([])

    K_new    = max(1, round(org.R_t / (org.N - 1)))
    target_d = _s3c._select_target(mindset, g_star, org.goal_types,
                                    target_b, target_e, target_any)

    existing = list(range(N_old))
    if target_d is not None:
        pool     = [d for d in existing if d != target_d]
        n_rnd    = min(K_new - 1, len(pool))
        rnd      = rng.choice(pool, size=n_rnd, replace=False).tolist() if n_rnd > 0 else []
        outgoing = [target_d] + rnd
    else:
        n_rnd    = min(K_new, len(existing))
        outgoing = rng.choice(existing, size=n_rnd, replace=False).tolist() if n_rnd > 0 else []

    d_new_grp = int(new_grp)
    table_new = []
    for g in range(G):
        gtype   = int(org.goal_types[g])
        entries = np.empty(2)
        if g == g_star:
            # Variant A/B draw site 1 — target goal
            if success:
                entries[0] = rng.uniform(0.5, 1.0)
                entries[1] = rng.uniform(0.5, 1.0)
            else:
                entries[0] = rng.uniform(0.0, 1.0)
                entries[1] = rng.uniform(0.0, 1.0)
        elif (d_new_grp == 1 and gtype == 1) or (d_new_grp == 2 and gtype == 0):
            # Group-B: harmful to ethical goals; Group-E: harmful to business
            entries[0] = rng.uniform(-alpha, 0.0)
            entries[1] = rng.uniform(-alpha, 0.0)
        else:
            entries[0] = rng.uniform(0.0, 1.0)
            entries[1] = rng.uniform(0.0, 1.0)
        table_new.append(entries)
    org.tables.append(table_new)

    for i in outgoing:
        if len(org.influencers[i]) >= _s3c.K_MAX_INFLUENCERS:
            continue
        org.influencers[i].append(d_new)
        org.influencers_set[i].add(d_new)
        org.dependents[d_new].append(i)

        old_size = len(org.tables[i][0])
        new_size = old_size * 2
        grp_i    = int(org.group_assign[i])
        new_tables_i = []
        for g in range(G):
            old_vals = org.tables[i][g]
            new_t    = np.empty(new_size)
            new_t[0::2] = old_vals
            gtype = int(org.goal_types[g])
            if g == g_star:
                # Variant A/B draw site 2 — target goal branch
                if success:
                    caps = np.maximum(0.0, 1.0 - old_vals)
                    new_t[1::2] = old_vals + rng.uniform(0.0, 1.0, size=old_size) * caps
                else:
                    new_t[1::2] = rng.uniform(0.0, 1.0, size=old_size)
            elif (grp_i == 1 and gtype == 1) or (grp_i == 2 and gtype == 0):
                new_t[1::2] = rng.uniform(-alpha, 1.0, size=old_size)
            else:
                new_t[1::2] = rng.uniform(0.0, 1.0, size=old_size)
            new_tables_i.append(new_t)
        org.tables[i] = new_tables_i

    dept_members = org.departments[dept_idx]
    n_dept       = len(dept_members)

    def _dept_perf_g(dval):
        org.decisions[d_new] = dval
        total = 0.0
        for ii in dept_members:
            cidx  = contribution_index(ii, org.decisions, org.influencers)
            total += org.tables[ii][g_star][cidx]
        return total / n_dept

    p0 = _dept_perf_g(0)
    p1 = _dept_perf_g(1)
    org.decisions[d_new] = np.int8(1) if p1 >= p0 else np.int8(0)

    d_cidx    = int(org.decisions[d_new])
    new_w_row = np.array([[org.tables[d_new][g][d_cidx] for g in range(G)]])
    org.w     = np.vstack([org.w, new_w_row])

    for i in outgoing:
        cidx = contribution_index(i, org.decisions, org.influencers)
        for g in range(G):
            org.w[i, g] = org.tables[i][g][cidx]

    org.R_t += K_new

    # Variant E: draw hold duration after each innovation event
    if _RB_FLAGS["innov_hold_max"] > 0:
        _RB_FLAGS["_hold_counter"] = int(rng.integers(0, _RB_FLAGS["innov_hold_max"] + 1))

    return d_new, K_new


# ── Variant E: hold-aware accept wrappers ─────────────────────────────────────

_original_accept_org  = _s3c._accept_org
_original_accept_dept = _s3c._accept_dept


def _accept_org_hold(*args, **kwargs):
    """Return False (freeze search) while hold counter is active."""
    if _RB_FLAGS["_hold_counter"] > 0:
        _RB_FLAGS["_hold_counter"] -= 1
        return False
    return _original_accept_org(*args, **kwargs)


def _accept_dept_hold(*args, **kwargs):
    """Return False (freeze search) while hold counter is active."""
    if _RB_FLAGS["_hold_counter"] > 0:
        _RB_FLAGS["_hold_counter"] -= 1
        return False
    return _original_accept_dept(*args, **kwargs)


# ── patched _simulate_s3c (applies innov_cost) ────────────────────────────────

_original_simulate = _s3c._simulate_s3c


def _simulate_s3c_rb(org, T, regime, S, mindset, has_innovation, innovation_type,
                      theta_g, A_g, delta_g, alpha, rng):
    hold_max = _RB_FLAGS["innov_hold_max"]

    orig_innovate = _s3c._innovate
    _s3c._innovate = _innovate_rb

    if hold_max > 0:
        _s3c._accept_org  = _accept_org_hold
        _s3c._accept_dept = _accept_dept_hold
        _RB_FLAGS["_hold_counter"] = 0

    try:
        results = _original_simulate(
            org, T, regime, S, mindset, has_innovation, innovation_type,
            theta_g, A_g, delta_g, alpha, rng
        )
    finally:
        _s3c._innovate = orig_innovate
        if hold_max > 0:
            _s3c._accept_org  = _original_accept_org
            _s3c._accept_dept = _original_accept_dept
            _RB_FLAGS["_hold_counter"] = 0

    cost      = _RB_FLAGS["innov_cost"]
    k_periods = _RB_FLAGS["innov_cost_periods"]
    if cost > 0.0 and has_innovation:
        perf_series = results[0]
        innov_steps = results[5]
        for t in innov_steps:
            for k in range(k_periods):
                ts = t + k
                if ts < T:
                    perf_series[ts] = np.maximum(0.0, perf_series[ts] - cost)
    return results


# ── public entry point ────────────────────────────────────────────────────────

def run_replication_rb(N, G, G_b, M, R, T, regime, n_conflict, alpha,
                        mindset, has_innovation, innovation_type, seed,
                        beta=0.10, n_within=90, n_between=30,
                        innov_pi=1.0, innov_symmetric=False, innov_cost=0.0,
                        innov_cost_periods=1, innov_hold_max=0,
                        innov_tradeoff_prob=0.0):
    """
    Run one robustness replication.

    Parameters
    ----------
    innov_pi            : float — innovation success probability (Variant A).
    innov_symmetric     : bool  — fully unbiased draws (Variant B).
    innov_cost          : float — per-innovation performance deduction (Variant C).
    innov_tradeoff_prob : float — probability that a non-WAC innovation is
                          assigned Group B (business-favoring) rather than Group N.

    Returns
    -------
    (s3c_result, group_b_log)

    s3c_result  : the standard 14-element tuple from run_replication_s3c.
    group_b_log : list of (is_group_b: bool, g_star_is_ethical: bool),
                  one entry per innovation event, in order of occurrence.
                  Zip with s3c_result[11] (innov_steps) for step-level detail.
    """
    _RB_FLAGS["innov_pi"]            = float(innov_pi)
    _RB_FLAGS["innov_symmetric"]     = bool(innov_symmetric)
    _RB_FLAGS["innov_cost"]          = float(innov_cost)
    _RB_FLAGS["innov_cost_periods"]  = int(innov_cost_periods)
    _RB_FLAGS["innov_hold_max"]      = int(innov_hold_max)
    _RB_FLAGS["innov_tradeoff_prob"] = float(innov_tradeoff_prob)
    _RB_FLAGS["_group_b_log"]        = []
    _RB_FLAGS["_hold_counter"]       = 0

    orig_sim      = _s3c._simulate_s3c
    _s3c._simulate_s3c = _simulate_s3c_rb
    try:
        result = _s3c.run_replication_s3c(
            N=N, G=G, G_b=G_b, M=M, R=R, T=T,
            regime=regime, n_conflict=n_conflict, alpha=alpha,
            mindset=mindset, has_innovation=has_innovation,
            innovation_type=innovation_type, seed=seed,
            beta=beta, n_within=n_within, n_between=n_between,
        )
    finally:
        _s3c._simulate_s3c = orig_sim
        _RB_FLAGS["innov_pi"]            = 1.0
        _RB_FLAGS["innov_symmetric"]     = False
        _RB_FLAGS["innov_cost"]          = 0.0
        _RB_FLAGS["innov_cost_periods"]  = 1
        _RB_FLAGS["innov_hold_max"]      = 0
        _RB_FLAGS["innov_tradeoff_prob"] = 0.0
        _RB_FLAGS["_hold_counter"]       = 0

    group_b_log = list(_RB_FLAGS["_group_b_log"])
    _RB_FLAGS["_group_b_log"] = []
    return result, group_b_log
