"""
Stage 1B contribution table builder — decision-level partition model.

Each decision i is assigned to one of three groups at replication init:
  Group N (neutral)          : omega_ig ~ U(0, 1)  for all goals g
  Group B (business-favoring): omega_ig ~ U(0, 1)  for business goals
                               omega_ig ~ U(-alpha, 0) for ethical goals
  Group E (ethics-favoring)  : omega_ig ~ U(0, 1)  for ethical goals
                               omega_ig ~ U(-alpha, 0) for business goals

All draws are independent — no shared factors, no designed correlations.
Any goal-performance correlations that emerge are purely from dynamics.
"""
import numpy as np


def build_contribution_tables_s1b(N, G, influencers, goal_types,
                                   group_assign, alpha, rng=None):
    """
    Parameters
    ----------
    N           : int — number of decisions
    G           : int — total goals
    influencers : list of lists — influencers[i] = decisions that influence i
    goal_types  : ndarray (G,) — 0 = business, 1 = ethical
    group_assign: ndarray (N,) — 0 = Group N, 1 = Group B, 2 = Group E
    alpha       : float — negative bound magnitude; conflicted draws ~ U(-alpha, 0)
    rng         : numpy Generator

    Returns
    -------
    tables : list of lists — tables[i][g] is ndarray of contributions for
             decision i, goal g, over all 2^(1+k_i) lookup-table rows.
    """
    if rng is None:
        rng = np.random.default_rng()

    tables = []
    for i in range(N):
        k_i  = len(influencers[i])
        size = 1 << (1 + k_i)
        omega = np.empty((size, G))

        grp = group_assign[i]

        for g in range(G):
            gtype = goal_types[g]   # 0 = business, 1 = ethical

            if grp == 0:
                # Group N: positive for all goals
                omega[:, g] = rng.uniform(0.0, 1.0, size=size)
            elif grp == 1:
                # Group B: positive for business, negative for ethical
                if gtype == 0:
                    omega[:, g] = rng.uniform(0.0, 1.0, size=size)
                else:
                    omega[:, g] = rng.uniform(-alpha, 0.0, size=size)
            else:
                # Group E: positive for ethical, negative for business
                if gtype == 1:
                    omega[:, g] = rng.uniform(0.0, 1.0, size=size)
                else:
                    omega[:, g] = rng.uniform(-alpha, 0.0, size=size)

        tables.append([omega[:, g].copy() for g in range(G)])

    return tables
