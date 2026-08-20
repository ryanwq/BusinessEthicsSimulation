"""
Stage 1B Organization — decision-level partition model.

Differences from OrganizationS1:
  - No factor model, no lambdas, no d_x.
  - Decisions randomly partitioned into Group N / Group B / Group E.
  - n_conflict controls total conflicted decisions (split equally B/E).
  - alpha controls the negative bound U(-alpha, 0) for conflicted pairs.
  - No theoretical_correlations() method — correlations are empirical only.
"""
import numpy as np
from landscape import build_dependency_graph, compute_all_contributions
from s1b_landscape import build_contribution_tables_s1b
from regime import DEFAULT_S


class OrganizationS1B:
    """
    One organization instance for a Stage 1B replication.

    Parameters
    ----------
    N          : total binary decisions (24)
    M          : departments (4)
    G          : total goals (8, split G_b=4 business / G_e=4 ethical)
    G_b        : number of business goals (4)
    R          : directed interdependency edges (120)
    n_conflict : total conflicted decisions — must be even; n_conflict/2 go
                 to Group B and n_conflict/2 to Group E.
                 Remaining N - n_conflict decisions are Group N.
    alpha      : float — magnitude of negative bound for conflicted draws.
    rng        : numpy Generator
    n_within   : within-dept edges (default 90)
    n_between  : between-dept edges (default 30)
    """

    def __init__(self, N, M, G, G_b, R, n_conflict, alpha, rng,
                 n_within=90, n_between=30):
        self.N = N
        self.M = M
        self.G = G
        self.G_b = G_b
        self.G_e = G - G_b
        self.n_conflict = n_conflict
        self.alpha = alpha
        self.dept_size = N // M

        # ── dependency graph ──────────────────────────────────────────────
        self.influencers, self.dept_of, self.dependents, self.influencers_set = (
            build_dependency_graph(N, M, R, rng, n_within, n_between)
        )

        # ── goal type assignment: random half-half split ──────────────────
        perm = rng.permutation(G)
        self.business_idx = sorted(perm[:G_b].tolist())
        self.ethical_idx  = sorted(perm[G_b:].tolist())

        self.goal_types = np.zeros(G, dtype=np.int8)   # 0=business, 1=ethical
        for g in self.ethical_idx:
            self.goal_types[g] = 1

        # ── decision group assignment (random, without replacement) ───────
        # group_assign[i] in {0=N, 1=B, 2=E}
        n_b = n_conflict // 2
        n_e = n_conflict // 2
        n_n = N - n_conflict

        group_labels = np.array([0] * n_n + [1] * n_b + [2] * n_e, dtype=np.int8)
        self.group_assign = rng.permutation(group_labels)

        # ── contribution tables ───────────────────────────────────────────
        self.tables = build_contribution_tables_s1b(
            N, G, self.influencers, self.goal_types,
            self.group_assign, alpha, rng
        )

        # ── department membership ─────────────────────────────────────────
        self.departments = [
            list(range(k * self.dept_size, (k + 1) * self.dept_size))
            for k in range(M)
        ]
        self.dept_sets = [set(d) for d in self.departments]

        # ── regime-specific goal assignments ──────────────────────────────
        goals_per_dept = G // M
        self.dept_goals = [
            list(range(k * goals_per_dept, (k + 1) * goals_per_dept))
            for k in range(M)
        ]
        self.myopic_goal = int(rng.integers(0, G))

        temporal_goals: list = []
        while len(temporal_goals) < 300:
            temporal_goals.extend(rng.permutation(G).tolist())
        self.temporal_goals = temporal_goals

        # ── initial state ─────────────────────────────────────────────────
        self.decisions = rng.integers(0, 2, size=N, dtype=np.int8)
        self.w = compute_all_contributions(
            N, G, self.decisions, self.influencers, self.tables
        )

    def performance(self):
        """Return per-goal performance p_g, shape (G,)."""
        return self.w.mean(axis=0)

    def performance_by_type(self):
        """Return (Omega_b, Omega_e)."""
        p = self.w.mean(axis=0)
        return float(p[self.business_idx].mean()), float(p[self.ethical_idx].mean())
