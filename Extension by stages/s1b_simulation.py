"""
Stage 1B simulation loop — identical acceptance rules to Stage 1.
"""
import numpy as np
from landscape import contribution_index
from s1b_organization import OrganizationS1B
from regime import get_active_goals, DEFAULT_S


def _simulate_s1b(org, T, regime, S, rng):
    """Run one replication; returns perf_series (T, G)."""
    N = org.N
    G = org.G
    M = org.M
    dept_size = org.dept_size

    perf_series = np.empty((T, G))

    for t in range(T):
        dept_idx     = int(rng.integers(0, M))
        dept_dec     = org.departments[dept_idx]
        d            = dept_dec[int(rng.integers(0, dept_size))]
        active_goals = get_active_goals(regime, dept_idx, t, org, S)

        if regime == "baseline":
            org.decisions[d] ^= 1
            all_affected = [d] + org.dependents[d]

            delta = np.zeros(G)
            new_w_affected = {}
            for i in all_affected:
                cidx = contribution_index(i, org.decisions, org.influencers)
                nw = np.array([org.tables[i][g][cidx] for g in range(G)])
                delta += nw - org.w[i]
                new_w_affected[i] = nw

            if bool(np.any(delta > 0) and np.all(delta >= 0)):
                for i in all_affected:
                    org.w[i] = new_w_affected[i]
            else:
                org.decisions[d] ^= 1

        else:
            cur_contrib = org.w[dept_dec][:, active_goals].mean(axis=0)

            org.decisions[d] ^= 1
            all_affected = [d] + org.dependents[d]

            dept_set = org.dept_sets[dept_idx]
            affected_in_dept = [d] + [i for i in org.dependents[d] if i in dept_set]
            dept_w = org.w[dept_dec].copy()
            for i in affected_in_dept:
                local = i - dept_idx * dept_size
                cidx  = contribution_index(i, org.decisions, org.influencers)
                for g in range(G):
                    dept_w[local, g] = org.tables[i][g][cidx]
            new_contrib = dept_w[:, active_goals].mean(axis=0)

            if float((new_contrib - cur_contrib).mean()) > 0:
                for i in all_affected:
                    cidx = contribution_index(i, org.decisions, org.influencers)
                    for g in range(G):
                        org.w[i, g] = org.tables[i][g][cidx]
            else:
                org.decisions[d] ^= 1

        perf_series[t] = org.w.mean(axis=0)

    return perf_series


def run_replication_s1b(N, G, G_b, M, R, T, regime, n_conflict, alpha, seed,
                         n_within=90, n_between=30):
    """Run one Stage 1B replication; return (perf_series, business_idx, ethical_idx)."""
    rng = np.random.default_rng(seed)
    org = OrganizationS1B(N, M, G, G_b, R, n_conflict, alpha, rng, n_within, n_between)
    S   = DEFAULT_S.get(G, max(1, T // G))
    perf = _simulate_s1b(org, T, regime, S, rng)
    return perf, org.business_idx, org.ethical_idx
