"""
Builds the NK landscape: dependency graph, contribution lookup tables,
and performance computation for the Ethiraj & Levinthal (2009) model.
"""
import numpy as np


def build_dependency_graph(N, M, R, rng, n_within=90, n_between=30, undirected=False):
    """
    Create dependency edges across N decisions in M departments.

    If undirected=False (default): each sampled pair (i,j) is a one-way directed edge
    — j influences i only. Total influencer relationships = n_within + n_between.

    If undirected=True: each sampled pair (i,j) is treated as bidirectional — j
    influences i AND i influences j. Total influencer relationships up to
    2*(n_within + n_between), roughly doubling landscape complexity.

    Returns:
        influencers:     influencers[i]  = list of decision indices that decision i depends on
        dept_of:         array length N, dept_of[i] = department index of decision i
        dependents:      dependents[j]   = list of decisions i where j is in influencers[i]
        influencers_set: influencers_set[i] = set version of influencers[i] for O(1) lookup
    """
    dept_size = N // M
    dept_of = np.repeat(np.arange(M), dept_size)

    influencers = [[] for _ in range(N)]
    dependents = [[] for _ in range(N)]
    influencers_set = [set() for _ in range(N)]

    if R == 0:
        return influencers, dept_of, dependents, influencers_set

    within_pairs = []
    between_pairs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dept_of[i] == dept_of[j]:
                within_pairs.append((i, j))
            else:
                between_pairs.append((i, j))

    n_w = min(n_within, len(within_pairs))
    n_b = min(n_between, len(between_pairs))

    chosen_w = rng.choice(len(within_pairs), size=n_w, replace=False)
    chosen_b = rng.choice(len(between_pairs), size=n_b, replace=False)

    def _add_edge(i, j):
        if j not in influencers_set[i]:
            influencers[i].append(j)
            influencers_set[i].add(j)
            dependents[j].append(i)

    for k in chosen_w:
        i, j = within_pairs[k]
        _add_edge(i, j)
        if undirected:
            _add_edge(j, i)

    for k in chosen_b:
        i, j = between_pairs[k]
        _add_edge(i, j)
        if undirected:
            _add_edge(j, i)

    return influencers, dept_of, dependents, influencers_set


def build_contribution_tables(N, G, influencers, rng):
    """
    For each decision i and goal g, draw Uniform(0,1) values for every possible
    combination of (d_i, influencers). Draws are independent across i and g.

    Returns tables where tables[i][g] is a numpy array of length 2^(1 + k_i).
    """
    tables = []
    for i in range(N):
        k_i = len(influencers[i])
        size = 1 << (1 + k_i)
        tables.append([rng.random(size) for _ in range(G)])
    return tables


def contribution_index(i, decisions, influencers):
    """
    Compute the lookup-table index for decision i given the current decision vector.
    Index = binary number formed by [decisions[i], decisions[inf_1], decisions[inf_2], ...].
    """
    idx = int(decisions[i])
    for j in influencers[i]:
        idx = (idx << 1) | int(decisions[j])
    return idx


def compute_all_contributions(N, G, decisions, influencers, tables):
    """Return w of shape (N, G) where w[i,g] = contribution of decision i to goal g."""
    w = np.empty((N, G))
    for i in range(N):
        cidx = contribution_index(i, decisions, influencers)
        for g in range(G):
            w[i, g] = tables[i][g][cidx]
    return w


def compute_global_peak(N, G, influencers, tables):
    """
    Find the global performance peak by exhaustive enumeration of all 2^N
    decision vectors.  Only tractable for small N (N ≤ 20).

    Returns the maximum achievable mean firm performance (scalar in [0, 1]).
    """
    best = 0.0
    decisions = np.zeros(N, dtype=np.int8)
    for state_int in range(1 << N):
        for i in range(N):
            decisions[i] = (state_int >> i) & 1
        total = 0.0
        for i in range(N):
            cidx = contribution_index(i, decisions, influencers)
            for g in range(G):
                total += tables[i][g][cidx]
        avg = total / (N * G) if G > 0 else total / N
        if avg > best:
            best = avg
    return best
