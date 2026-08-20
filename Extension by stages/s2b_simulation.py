"""
Stage 2B simulation — ethical mindsets on the Stage 1B partition landscape.

Four mindsets replace the goal-blind acceptance rule:
  "wac"        Win at All Costs
  "sf"         Survival First
  "compliance" Compliance
  "cei"        Continuous Ethical Improvement

Thresholds: Omega_g(e)_min = p_g(0) * k_g, k_g ~ U(0.9, 1.1), per ethical goal.
Set once at initialization from org-level initial performance; held fixed.

New output: accepted_counts[3] — accepted flips by group type (N, B, E).
"""
import numpy as np
from landscape import contribution_index
from s1b_organization import OrganizationS1B
from regime import get_active_goals, DEFAULT_S


# ── threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(org, rng):
    """
    Compute per-ethical-goal minimum performance thresholds.

    Returns ndarray (G,) where entry g is Omega_g(e)_min for ethical goals
    and 0.0 for business goals (thresholds unused for business goals).
    """
    p0 = org.w.mean(axis=0)          # org-level initial performance per goal
    thresholds = np.zeros(org.G)
    for g in org.ethical_idx:
        k_g = rng.uniform(0.9, 1.1)
        thresholds[g] = p0[g] * k_g
    return thresholds


# ── mindset acceptance rules ──────────────────────────────────────────────────

def _accept_org(mindset, delta, active_b, active_e, cur_perf, new_perf,
                b_arr, e_arr, thresholds):
    """
    Org-level (Baseline strategy) acceptance decision.

    delta    : (G,) change in org-wide mean performance per goal if flip accepted
    cur_perf : (G,) current org-wide mean performance per goal
    new_perf : (G,) = cur_perf + delta
    """
    if mindset == "wac":
        # Pareto restricted to business goals
        db = delta[b_arr]
        return bool(np.any(db > 0) and np.all(db >= 0))

    elif mindset == "sf":
        # Standard Pareto over all goals, OR any business goal strictly improves
        db = delta[b_arr]
        pareto_all = bool(np.any(delta > 0) and np.all(delta >= 0))
        biz_improves = bool(np.any(db > 0) and np.all(db >= 0))
        return pareto_all or biz_improves

    elif mindset == "compliance":
        # (a) No ethical goal at/above threshold may fall below it
        for g in e_arr:
            if cur_perf[g] >= thresholds[g] and new_perf[g] < thresholds[g]:
                return False
        # (b) If any ethical goal below threshold, it must improve or stay same
        for g in e_arr:
            if cur_perf[g] < thresholds[g] and new_perf[g] < cur_perf[g]:
                return False
        # (c) Pareto over business goals
        db = delta[b_arr]
        return bool(np.any(db > 0) and np.all(db >= 0))

    elif mindset == "cei":
        # Strong Pareto over all goals
        return bool(np.any(delta > 0) and np.all(delta >= 0))

    return False


def _accept_dept(mindset, delta, active_b, active_e, active_goals,
                 cur_dept, new_dept, b_arr, e_arr, thresholds):
    """
    Dept-level acceptance decision.

    delta    : (G,) change in dept-mean performance per goal
    cur_dept : (G,) current dept-mean performance per goal
    new_dept : (G,) new dept-mean performance per goal
    active_b : list of active business goal indices
    active_e : list of active ethical goal indices
    """
    if mindset == "wac":
        if not active_b:
            return True          # no active business goals → accept freely
        return float(np.mean(delta[active_b])) > 0

    elif mindset == "sf":
        # (a) active business goal improves
        if active_b and float(np.mean(delta[active_b])) > 0:
            return True
        # (b) active ethical goal improves AND no business goal worsens
        if active_e and float(np.mean(delta[active_e])) > 0:
            if np.all(delta[b_arr] >= 0):
                return True
        return False

    elif mindset == "compliance":
        # (a) no ethical goal at/above threshold may fall below it
        for g in e_arr:
            if cur_dept[g] >= thresholds[g] and new_dept[g] < thresholds[g]:
                return False
        # (b) if active ethical goal below threshold, it must not worsen
        for g in active_e:
            if cur_dept[g] < thresholds[g] and new_dept[g] < cur_dept[g]:
                return False
        # (c) active business goal must not worsen
        if active_b and float(np.mean(delta[active_b])) < 0:
            return False
        return True

    elif mindset == "cei":
        # Improve active business OR active ethical; worsen no active goal
        improves_b = bool(active_b and float(np.mean(delta[active_b])) > 0)
        improves_e = bool(active_e and float(np.mean(delta[active_e])) > 0)
        if not (improves_b or improves_e):
            return False
        if active_goals and np.any(delta[list(active_goals)] < 0):
            return False
        return True

    return False


# ── main simulation loop ──────────────────────────────────────────────────────

def _simulate_s2b(org, T, regime, S, mindset, thresholds, rng):
    """
    Run one Stage 2B replication.

    Returns
    -------
    perf_series     : ndarray (T, G) — per-goal performance at each step
    accepted_counts : ndarray (3,)   — accepted flips by group [N, B, E]
    """
    N         = org.N
    G         = org.G
    M         = org.M
    dept_size = org.dept_size
    b_set     = set(org.business_idx)
    e_set     = set(org.ethical_idx)
    b_arr     = np.array(org.business_idx)
    e_arr     = np.array(org.ethical_idx)

    perf_series     = np.empty((T, G))
    accepted_counts = np.zeros(3, dtype=np.int32)

    for t in range(T):
        dept_idx     = int(rng.integers(0, M))
        dept_dec     = org.departments[dept_idx]
        d            = dept_dec[int(rng.integers(0, dept_size))]
        active_goals = get_active_goals(regime, dept_idx, t, org, S)
        active_b     = [g for g in active_goals if g in b_set]
        active_e     = [g for g in active_goals if g in e_set]

        # ── tentative flip ────────────────────────────────────────────────
        org.decisions[d] ^= 1
        all_affected = [d] + org.dependents[d]

        if regime == "baseline":
            # Org-level: accumulate contribution changes across all affected
            delta         = np.zeros(G)
            new_w_affected = {}
            for i in all_affected:
                cidx = contribution_index(i, org.decisions, org.influencers)
                nw   = np.array([org.tables[i][g][cidx] for g in range(G)])
                delta += nw - org.w[i]
                new_w_affected[i] = nw
            delta   /= N
            cur_perf = org.w.mean(axis=0)
            new_perf = cur_perf + delta

            accepted = _accept_org(mindset, delta, active_b, active_e,
                                   cur_perf, new_perf, b_arr, e_arr, thresholds)
            if accepted:
                for i in all_affected:
                    org.w[i] = new_w_affected[i]
            else:
                org.decisions[d] ^= 1

        else:
            # Dept-level: update contributions for affected decisions in dept
            dept_set         = org.dept_sets[dept_idx]
            affected_in_dept = [d] + [i for i in org.dependents[d] if i in dept_set]
            dept_w           = org.w[dept_dec].copy()
            for i in affected_in_dept:
                local = i - dept_idx * dept_size
                cidx  = contribution_index(i, org.decisions, org.influencers)
                for g in range(G):
                    dept_w[local, g] = org.tables[i][g][cidx]

            cur_dept = org.w[dept_dec].mean(axis=0)
            new_dept = dept_w.mean(axis=0)
            delta    = new_dept - cur_dept

            accepted = _accept_dept(mindset, delta, active_b, active_e,
                                    active_goals, cur_dept, new_dept,
                                    b_arr, e_arr, thresholds)
            if accepted:
                for i in all_affected:
                    cidx = contribution_index(i, org.decisions, org.influencers)
                    for g in range(G):
                        org.w[i, g] = org.tables[i][g][cidx]
            else:
                org.decisions[d] ^= 1

        if accepted:
            accepted_counts[org.group_assign[d]] += 1

        perf_series[t] = org.w.mean(axis=0)

    return perf_series, accepted_counts


def run_replication_s2b(N, G, G_b, M, R, T, regime, n_conflict, alpha,
                         mindset, seed, n_within=90, n_between=30):
    """
    Run one Stage 2B replication.

    Returns
    -------
    perf_series     : (T, G)
    business_idx    : list
    ethical_idx     : list
    thresholds      : (G,)
    accepted_counts : (3,) [N, B, E]
    """
    rng = np.random.default_rng(seed)
    org = OrganizationS1B(N, M, G, G_b, R, n_conflict, alpha, rng, n_within, n_between)

    # WAC only monitors business goals — restrict all regime goal pools accordingly
    # so temporal never cycles to ethical goals and myopic never draws one.
    if mindset == "wac":
        b = org.business_idx                          # e.g. [1, 3, 5, 7]
        org.myopic_goal = int(rng.choice(b))
        tg: list = []
        while len(tg) < 300:
            tg.extend(rng.permutation(b).tolist())
        org.temporal_goals = tg
        org.dept_goals = [[b[k]] for k in range(len(b))]   # one business goal per dept

    pool_size = len(org.business_idx) if mindset == "wac" else G
    S   = DEFAULT_S.get(pool_size, max(1, T // pool_size))
    thresholds = compute_thresholds(org, rng)
    perf, acc  = _simulate_s2b(org, T, regime, S, mindset, thresholds, rng)
    return perf, org.business_idx, org.ethical_idx, thresholds, acc
