"""
Stage 3C simulation — aspiration-level innovation on the partition landscape.

Key changes from Stage 3B:
  1. Innovation trigger: shortfall — ω_g < A_g for N_t consecutive steps.
     Stage 3B used stagnation (no improvement for N_t steps).
  2. A_g = ω_g(0) × (1 + β): internal aspiration. Default β = 0.10.
     Compliance ethical: A_g,E = θ_g,E = ω_g(0) × k_g, k_g ~ U(0.9, 1.1).
  3. θ_g,E drawn for ALL conditions (including no-innovation).
     comp_rate tracked every step in all 36 cells.
  4. g* = argmax(A_g − ω_g) over triggered goals (largest shortfall first).
  5. Post-innovation reset: T_below[g] ← 0 if ω_g(post) ≥ A_g (not if improved).
  6. CEI+I Ratcheting: when T_below[g] resets from >0 to 0, A_g += Δ_g.
  7. Myopic: both a business goal AND an ethical goal active per step (not one).

Acceptance rules identical to Stage 3B:
  WAC baseline = Pareto over all biz;  non-baseline = mean active biz > 0.
  SF  baseline = Pareto or any biz>0;  non-baseline = (a) or (b).
  Compliance   = three conditions (a)(b)(c) with θ_g,E floor.
  CEI baseline = Pareto over all goals; non-baseline = Pareto over active goals.
"""
import numpy as np
from landscape import contribution_index, compute_all_contributions
from s1b_organization import OrganizationS1B
from regime import DEFAULT_S

BETA = 0.10


# ── threshold / aspiration computation ───────────────────────────────────────

def compute_aspirations(org, rng, mindset, has_innovation, beta=BETA):
    """
    Compute θ_g,E (external threshold, all conditions) and A_g (internal
    aspiration, +I conditions only).

    Returns
    -------
    theta_g  : (G,) float  — external compliance threshold for ethical goals
    A_g      : (G,) float  — internal aspiration (0 for untracked goals)
    delta_g  : (G,) float  — ratchet increment (0 for non-ratcheting)
    """
    perf0   = org.w.mean(axis=0)
    theta_g = np.zeros(org.G)
    A_g     = np.zeros(org.G)
    delta_g = np.zeros(org.G)

    # θ_g,E drawn for ALL conditions
    for g in org.ethical_idx:
        k_g = rng.uniform(0.9, 1.1)
        theta_g[g] = perf0[g] * k_g

    if not has_innovation:
        return theta_g, A_g, delta_g

    if mindset in ("wac", "sf"):
        for g in org.business_idx:
            A_g[g] = perf0[g] * (1.0 + beta)
    elif mindset == "compliance":
        for g in org.business_idx:
            A_g[g] = perf0[g] * (1.0 + beta)
        for g in org.ethical_idx:
            A_g[g] = theta_g[g]            # external standard serves dual role
    elif mindset == "cei":
        for g in range(org.G):
            A_g[g]     = perf0[g] * (1.0 + beta)
            delta_g[g] = perf0[g] * beta

    return theta_g, A_g, delta_g


# ── acceptance rules (same logic as Stage 3B) ────────────────────────────────

def _accept_org(mindset, delta, cur_perf, b_arr, e_arr, theta_g):
    """Baseline (org-level) acceptance."""
    if mindset == "wac":
        db = delta[b_arr]
        return bool(np.any(db > 0) and np.all(db >= 0))
    elif mindset == "sf":
        db = delta[b_arr]
        return bool((np.any(delta > 0) and np.all(delta >= 0)) or (np.any(db > 0) and np.all(db >= 0)))
    elif mindset == "compliance":
        new_perf = cur_perf + delta
        for g in e_arr:
            if cur_perf[g] >= theta_g[g] and new_perf[g] < theta_g[g]:
                return False
        for g in e_arr:
            if cur_perf[g] < theta_g[g] and new_perf[g] < cur_perf[g]:
                return False
        db = delta[b_arr]
        return bool(np.any(db > 0) and np.all(db >= 0))
    elif mindset == "cei":
        return bool(np.any(delta > 0) and np.all(delta >= 0))
    return False


def _accept_dept(mindset, delta, active_b, active_e, active_goals,
                 cur_dept, new_dept, b_arr, e_arr, theta_g):
    """Non-baseline (dept-level) acceptance."""
    if mindset == "wac":
        if not active_b:
            return True
        return float(np.mean(delta[active_b])) > 0
    elif mindset == "sf":
        if active_b and float(np.mean(delta[active_b])) > 0:
            return True
        if active_e and float(np.mean(delta[active_e])) > 0:
            if np.all(delta[b_arr] >= 0):
                return True
        return False
    elif mindset == "compliance":
        for g in e_arr:
            if cur_dept[g] >= theta_g[g] and new_dept[g] < theta_g[g]:
                return False
        for g in active_e:
            if cur_dept[g] < theta_g[g] and new_dept[g] < cur_dept[g]:
                return False
        if active_b and float(np.mean(delta[active_b])) < 0:
            return False
        return True
    elif mindset == "cei":
        improves_b = bool(active_b and float(np.mean(delta[active_b])) > 0)
        improves_e = bool(active_e and float(np.mean(delta[active_e])) > 0)
        if not (improves_b or improves_e):
            return False
        if active_goals and np.any(delta[list(active_goals)] < 0):
            return False
        return True
    return False


# ── innovation procedure (identical to Stage 3B) ─────────────────────────────

def _select_target(mindset, g_star, goal_types, target_b, target_e, target_any):
    g_is_ethical = goal_types[g_star] == 1
    if mindset == "cei":
        rec = target_b.get(g_star) if g_is_ethical else target_e.get(g_star)
    else:
        rec = target_any.get(g_star)
    return rec[0] if rec is not None else None


def _innovate(org, g_star, mindset, alpha, target_b, target_e, target_any, rng):
    """Execute the innovation procedure (same as Stage 3B). Modifies org in-place."""
    G     = org.G
    N_old = org.N
    d_new = N_old

    org.N += 1
    dept_idx = int(rng.integers(0, org.M))
    org.departments[dept_idx].append(d_new)
    org.dept_sets[dept_idx].add(d_new)

    org.decisions    = np.append(org.decisions,    np.int8(0))
    org.group_assign = np.append(org.group_assign, np.int8(1) if mindset == "wac" else np.int8(0))
    org.influencers.append([])
    org.influencers_set.append(set())
    org.dependents.append([])

    K_new    = max(1, round(org.R_t / (org.N - 1)))
    target_d = _select_target(mindset, g_star, org.goal_types,
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

    d_new_grp = int(org.group_assign[d_new])
    table_new = []
    for g in range(G):
        gtype   = int(org.goal_types[g])
        entries = np.empty(2)
        if g == g_star:
            entries[0] = rng.uniform(0.5, 1.0)
            entries[1] = rng.uniform(0.5, 1.0)
        elif (d_new_grp == 1 and gtype == 1) or (d_new_grp == 2 and gtype == 0):
            entries[0] = rng.uniform(-alpha, 0.0)
            entries[1] = rng.uniform(-alpha, 0.0)
        else:
            entries[0] = rng.uniform(0.0, 1.0)
            entries[1] = rng.uniform(0.0, 1.0)
        table_new.append(entries)
    org.tables.append(table_new)

    for i in outgoing:
        if len(org.influencers[i]) >= K_MAX_INFLUENCERS:
            continue  # skip: table would exceed 2^K_MAX entries
        org.influencers[i].append(d_new)
        org.influencers_set[i].add(d_new)
        org.dependents[d_new].append(i)

        old_size = len(org.tables[i][0])
        new_size = old_size * 2
        grp_i    = int(org.group_assign[i])
        new_tables_i = []
        for g in range(G):
            old_vals = org.tables[i][g]         # shape (old_size,)
            new_t    = np.empty(new_size)
            new_t[0::2] = old_vals               # d_new=0 branch: unchanged
            gtype = int(org.goal_types[g])
            if g == g_star:
                caps = np.maximum(0.0, 1.0 - old_vals)
                new_t[1::2] = old_vals + rng.uniform(0.0, 1.0, size=old_size) * caps
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
    return d_new, K_new


# ── shortfall helpers ─────────────────────────────────────────────────────────

def _update_target_record(rec_dict, g, d_idx, delta_g):
    current = rec_dict.get(g)
    if current is None or delta_g < current[1]:
        rec_dict[g] = (d_idx, float(delta_g))


def _reset_goal_records(g, target_b, target_e, target_any):
    target_b.pop(g, None)
    target_e.pop(g, None)
    target_any.pop(g, None)


def _check_trigger(mindset, T_below, N_t, active_b, active_e):
    """
    Return list of (gap, g) for all triggering goals (T_below[g] >= N_t).
    Caller selects g* = argmax gap.
    """
    triggered = []
    if mindset in ("wac", "sf"):
        for g in active_b:
            if T_below[g] >= N_t:
                triggered.append(g)
    elif mindset == "compliance":
        for g in active_b + active_e:
            if T_below[g] >= N_t:
                triggered.append(g)
    elif mindset == "cei":
        for g in list(active_b) + list(active_e):
            if T_below[g] >= N_t:
                triggered.append(g)
    return triggered


# ── main simulation loop ──────────────────────────────────────────────────────

MAX_INNOV_PER_REP   = 80  # safety cap against runaway ratcheting
K_MAX_INFLUENCERS   = 15  # max influencers per decision (tables stay ≤ 2^15 = 32768)


def _simulate_s3c(org, T, regime, S, mindset, has_innovation, innovation_type,
                  theta_g, A_g, delta_g, alpha, rng):
    """
    Run one Stage 3C replication.

    innovation_type : None | 'fixed' | 'ratcheting'
    theta_g         : (G,) external compliance thresholds (all conditions)
    A_g             : (G,) internal aspirations (0 for untracked; all conditions pass it)
    delta_g         : (G,) ratchet increments (0 for non-ratcheting)

    Returns
    -------
    perf_series       : (T, G)   org-level performance at each step
    n_t_series        : (T,)     N_t at each step
    accepted_counts   : (3,)     accepted flips by group [N, B, E]
    innov_count       : int
    cascade_count     : int
    innov_steps       : list[int]
    above_all_count   : int  — steps where ALL ethical goals >= θ_g,E
    above_goal_counts : (G,) int — per-goal steps above θ_g,E
    ratchet_count     : int      — times A_g was raised (ratcheting only)
    A_g_final         : (G,) float — final aspiration values
    """
    G     = org.G
    M     = org.M
    b_set = set(org.business_idx)
    e_set = set(org.ethical_idx)
    b_arr = np.array(org.business_idx)
    e_arr = np.array(org.ethical_idx)

    perf_series       = np.empty((T, G))
    n_t_series        = np.empty(T, dtype=np.int32)
    accepted_counts   = np.zeros(3, dtype=np.int32)
    above_all_count   = 0
    above_goal_counts = np.zeros(G, dtype=np.int32)

    # Myopic: draw one business AND one ethical goal (Stage 3C extension)
    myopic_b = None
    myopic_e = None
    if regime == "myopic":
        myopic_b = int(rng.choice(org.business_idx))
        myopic_e = int(rng.choice(org.ethical_idx))

    # Shortfall state (with-innovation only)
    A_g_live  = A_g.copy()       # may change under ratcheting
    T_below   = np.zeros(G, dtype=np.int32)
    T_above   = np.zeros(G, dtype=np.int32)   # durable-attainment counter (Option 2)
    target_b  = {}
    target_e  = {}
    target_any = {}

    innov_steps   = []
    innov_count   = 0
    cascade_count = 0
    ratchet_count = 0
    prev_active   = set()

    for t in range(T):
        dept_idx     = int(rng.integers(0, M))
        dept_dec     = org.departments[dept_idx]
        d            = dept_dec[int(rng.integers(0, len(dept_dec)))]

        # Active goals — Stage 3C Myopic overrides org.myopic_goal
        if regime == "myopic":
            if mindset in ("sf", "compliance", "cei"):
                active_goals = [myopic_b, myopic_e]
            else:
                active_goals = [myopic_b]
        else:
            active_goals = _get_active_goals(regime, dept_idx, t, org, S)

        active_b = [g for g in active_goals if g in b_set]
        active_e = [g for g in active_goals if g in e_set]

        # Temporal CEI: reset counters when active goal rotates
        if has_innovation and mindset == "cei" and regime == "temporal":
            curr_active = set(active_goals)
            for g in prev_active - curr_active:
                T_below[g] = 0
                T_above[g] = 0
                _reset_goal_records(g, target_b, target_e, target_any)
            prev_active = curr_active

        # ── Step 4: innovation trigger ────────────────────────────────────
        innovation_fired = False
        if has_innovation and innov_count < MAX_INNOV_PER_REP:
            N_t      = org.N
            perf_now = org.w.mean(axis=0)
            triggered = _check_trigger(mindset, T_below, N_t, active_b, active_e)

            if triggered:
                # g* = largest shortfall
                gaps  = {g: A_g_live[g] - perf_now[g] for g in triggered}
                g_star = max(gaps, key=lambda g: (gaps[g], -g))

                innovation_fired = True
                innov_count += 1
                if innov_steps and (t - innov_steps[-1] < N_t):
                    cascade_count += 1
                innov_steps.append(t)

                pre_perf = perf_now.copy()
                _innovate(org, g_star, mindset, alpha,
                          target_b, target_e, target_any, rng)
                post_perf = org.w.mean(axis=0)

                # Post-innovation counter reset (Section 7.6)
                for g in range(G):
                    if post_perf[g] >= A_g_live[g]:
                        T_below[g] = 0
                        _reset_goal_records(g, target_b, target_e, target_any)
                        T_above[g] += 1
                        if innovation_type == "ratcheting" and T_above[g] >= org.N:
                            A_g_live[g] += delta_g[g]
                            ratchet_count += 1
                            T_above[g] = 0
                    else:
                        T_above[g] = 0  # innovation fell short; reset durable counter

                # Record on innovation steps (skipped by continue below)
                cur_perf_all = post_perf
                n_t_series[t] = org.N
                perf_series[t] = cur_perf_all
                for g in org.ethical_idx:
                    if cur_perf_all[g] >= theta_g[g]:
                        above_goal_counts[g] += 1
                if all(cur_perf_all[g] >= theta_g[g] for g in org.ethical_idx):
                    above_all_count += 1
                continue

        # ── Steps 5–6: flip evaluation ────────────────────────────────────
        if not innovation_fired:
            org.decisions[d] ^= 1
            all_affected = [d] + org.dependents[d]

            if regime == "baseline":
                delta       = np.zeros(G)
                new_w_cache = {}
                for i in all_affected:
                    cidx = contribution_index(i, org.decisions, org.influencers)
                    nw   = np.array([org.tables[i][g][cidx] for g in range(G)])
                    delta += nw - org.w[i]
                    new_w_cache[i] = nw
                delta   /= org.N
                cur_perf = org.w.mean(axis=0)

                accepted = _accept_org(mindset, delta, cur_perf, b_arr, e_arr, theta_g)
                if accepted:
                    for i in all_affected:
                        org.w[i] = new_w_cache[i]
                else:
                    org.decisions[d] ^= 1

            else:
                dept_set         = org.dept_sets[dept_idx]
                affected_in_dept = [d] + [i for i in org.dependents[d] if i in dept_set]
                dept_dec_list    = org.departments[dept_idx]
                dept_map         = {dec: k for k, dec in enumerate(dept_dec_list)}
                dept_w           = org.w[dept_dec_list].copy()

                for i in affected_in_dept:
                    k    = dept_map[i]
                    cidx = contribution_index(i, org.decisions, org.influencers)
                    for g in range(G):
                        dept_w[k, g] = org.tables[i][g][cidx]

                cur_dept = org.w[dept_dec_list].mean(axis=0)
                new_dept = dept_w.mean(axis=0)
                delta    = new_dept - cur_dept

                accepted = _accept_dept(mindset, delta, active_b, active_e,
                                        active_goals, cur_dept, new_dept,
                                        b_arr, e_arr, theta_g)
                if accepted:
                    for i in all_affected:
                        cidx = contribution_index(i, org.decisions, org.influencers)
                        for g in range(G):
                            org.w[i, g] = org.tables[i][g][cidx]
                else:
                    org.decisions[d] ^= 1

            if accepted:
                accepted_counts[org.group_assign[d]] += 1

            # ── Step 7: shortfall counter update ─────────────────────────
            if has_innovation:
                perf_eval = org.w.mean(axis=0) if regime == "baseline" else (
                    new_dept if accepted else cur_dept
                )
                grp_d = int(org.group_assign[d])

                if mindset in ("wac", "sf"):
                    tracked = active_b
                elif mindset == "compliance":
                    tracked = active_b + active_e
                else:
                    tracked = list(active_goals)

                for g in tracked:
                    if perf_eval[g] < A_g_live[g]:
                        T_below[g] += 1
                        T_above[g] = 0   # any dip below bar resets durable counter
                        # Update targeting records
                        if mindset == "cei":
                            if grp_d == 1:
                                _update_target_record(target_b, g, d, delta[g])
                            elif grp_d == 2:
                                _update_target_record(target_e, g, d, delta[g])
                        else:
                            _update_target_record(target_any, g, d, delta[g])
                    else:
                        T_below[g] = 0
                        _reset_goal_records(g, target_b, target_e, target_any)
                        T_above[g] += 1
                        if innovation_type == "ratcheting" and T_above[g] >= org.N:
                            A_g_live[g] += delta_g[g]
                            ratchet_count += 1
                            T_above[g] = 0   # must re-earn the new bar

        # ── Step 8: compliance tracking (ALL conditions, ALL steps) ──────
        cur_perf_all = org.w.mean(axis=0)
        all_above = True
        for g in org.ethical_idx:
            if cur_perf_all[g] >= theta_g[g]:
                above_goal_counts[g] += 1
            else:
                all_above = False
        if all_above:
            above_all_count += 1

        n_t_series[t]  = org.N
        perf_series[t] = cur_perf_all

    return (perf_series, n_t_series, accepted_counts,
            innov_count, cascade_count, innov_steps,
            above_all_count, above_goal_counts, ratchet_count, A_g_live.copy())


def _get_active_goals(regime, dept_idx, t, org, S):
    """Wrapper around regime module, avoids import collision."""
    G = org.G
    if regime == "baseline":
        return list(range(G))
    elif regime == "myopic":
        return [org.myopic_goal]
    elif regime == "spatial":
        return org.dept_goals[dept_idx]
    elif regime == "temporal":
        if S is None:
            S = DEFAULT_S.get(G, max(1, 500 // G))
        period = t // S
        return [org.temporal_goals[period]]
    return list(range(G))


# ── public entry point ────────────────────────────────────────────────────────

def run_replication_s3c(N, G, G_b, M, R, T, regime, n_conflict, alpha,
                         mindset, has_innovation, innovation_type, seed, beta=BETA,
                         n_within=90, n_between=30):
    """
    Run one Stage 3C replication.

    Parameters
    ----------
    innovation_type : None | 'fixed' | 'ratcheting'
      None     → no innovation (has_innovation must be False)
      'fixed'  → standard aspiration trigger, A_g fixed
      'ratcheting' → CEI only; A_g rises by delta_g each time achieved

    Returns
    -------
    perf_series       : (T, G)
    n_t_series        : (T,)
    business_idx      : list
    ethical_idx       : list
    theta_g           : (G,) external thresholds (all conditions)
    A_g_init          : (G,) initial internal aspirations
    A_g_final         : (G,) final aspirations (rises under ratcheting)
    accepted_counts   : (3,) [N, B, E]
    innov_count       : int
    cascade_count     : int
    innov_steps       : list[int]
    above_all_count   : int — steps where ALL ethical goals >= θ_g,E
    above_goal_counts : (G,) int — per-goal steps above θ_g,E
    ratchet_count     : int
    """
    rng = np.random.default_rng(seed)
    org = OrganizationS1B(N, M, G, G_b, R, n_conflict, alpha, rng, n_within, n_between)
    org.R_t = R

    # WAC only monitors business goals — restrict temporal and spatial goal pools.
    # (Myopic is already overridden below in _simulate_s3c using org.business_idx.)
    if mindset == "wac":
        b = org.business_idx
        tg: list = []
        while len(tg) < 300:
            tg.extend(rng.permutation(b).tolist())
        org.temporal_goals = tg
        org.dept_goals = [[b[k]] for k in range(len(b))]   # one business goal per dept

    pool_size = len(org.business_idx) if mindset == "wac" else G
    S       = DEFAULT_S.get(pool_size, max(1, T // pool_size))

    theta_g, A_g, delta_g = compute_aspirations(org, rng, mindset, has_innovation, beta)
    A_g_init = A_g.copy()

    (perf, n_t, acc, innov_cnt, casc_cnt, innov_steps,
     above_all_cnt, above_goal_cnts, ratchet_cnt, A_g_final) = _simulate_s3c(
        org, T, regime, S, mindset, has_innovation, innovation_type,
        theta_g, A_g, delta_g, alpha, rng
    )

    return (perf, n_t, org.business_idx, org.ethical_idx,
            theta_g, A_g_init, A_g_final,
            acc, innov_cnt, casc_cnt, innov_steps,
            above_all_cnt, above_goal_cnts, ratchet_cnt)
