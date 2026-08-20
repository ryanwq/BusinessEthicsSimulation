"""
Holds the organization state: decision vector, department assignments,
goal assignments for spatial and goal-myopia regimes.
"""
import numpy as np
from landscape import (
    build_dependency_graph,
    build_contribution_tables,
    compute_all_contributions,
)


class Organization:
    """
    Represents one organization instance for a single simulation replication.

    Attributes:
        decisions:       int8 array of shape (N,), values in {0, 1}
        w:               float64 array of shape (N, G), cached contribution values
        departments:     list of M lists, each containing the decision indices for that dept
        dept_goals:      list of M lists, each containing the goal indices assigned to that dept
                         (used by the spatial-differentiation regime)
        myopic_goal:     int, the single goal index used by the goal-myopia regime
    """

    def __init__(self, N, M, G, R, rng, n_within=90, n_between=30, undirected=False):
        self.N = N
        self.M = M
        self.G = G
        self.dept_size = N // M

        # Build landscape
        self.influencers, self.dept_of, self.dependents, self.influencers_set = (
            build_dependency_graph(N, M, R, rng, n_within, n_between, undirected)
        )
        self.tables = build_contribution_tables(N, G, self.influencers, rng)

        # Department membership (contiguous blocks of decisions)
        self.departments = [
            list(range(k * self.dept_size, (k + 1) * self.dept_size))
            for k in range(M)
        ]
        self.dept_sets = [set(d) for d in self.departments]

        # Spatial differentiation: assign goals evenly across departments
        goals_per_dept = G // M
        self.dept_goals = [
            list(range(k * goals_per_dept, (k + 1) * goals_per_dept))
            for k in range(M)
        ]

        # Goal myopia: pick one goal at random for the entire run
        self.myopic_goal = int(rng.integers(0, G))

        # Temporal differentiation: pre-generate random-permutation goal sequence.
        # At each "round" the organisation cycles through all G goals in a freshly
        # drawn random order (sample without replacement), matching Ethiraj's original
        # implementation.  We generate enough entries to cover any number of periods
        # up to T=250 with the minimum switch period S=1 (worst case 250 entries).
        temporal_goals: list[int] = []
        while len(temporal_goals) < 300:          # 300 > T for any plausible S
            temporal_goals.extend(rng.permutation(G).tolist())
        self.temporal_goals: list[int] = temporal_goals

        # Random initial decision vector
        self.decisions = rng.integers(0, 2, size=N, dtype=np.int8)

        # Initialize contribution cache
        self.w = compute_all_contributions(N, G, self.decisions, self.influencers, self.tables)

    def performance(self):
        """Return Omega_g for all G goals (shape (G,))."""
        return self.w.mean(axis=0)
