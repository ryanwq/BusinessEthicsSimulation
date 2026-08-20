"""
Encapsulates the four evaluation regimes from Ethiraj & Levinthal (2009).

The only thing that differs across regimes is which goals are "active"
(i.e., used in the accept/reject decision) at each step.

Regimes
-------
baseline   : all G goals active for every department on every step
myopic     : one randomly chosen goal, fixed for the entire run
             (goal is drawn at random once per replication in Organization)
spatial    : each department has a permanently assigned subset of goals
temporal   : the whole organization rotates through one goal at a time,
             switching every S periods.  The order of goals follows a fresh
             random permutation at each round (Ethiraj's implementation):
             each round picks the G goals in a newly sampled random order
             (without replacement), stored in org.temporal_goals.
"""

# Default goal-switch periods from the paper
DEFAULT_S = {4: 63, 8: 32}


def get_active_goals(regime, dept_idx, t, org, S=None):
    """
    Return the list of goal indices that are active for this evaluation.

    Parameters
    ----------
    regime   : str, one of {'baseline', 'myopic', 'spatial', 'temporal'}
    dept_idx : int, index of the department currently acting
    t        : int, current time step (0-indexed)
    org      : Organization instance
    S        : int or None, goal-switch period for temporal regime;
               defaults to DEFAULT_S[G] if None
    """
    G = org.G

    if regime == "baseline":
        return list(range(G))

    elif regime == "myopic":
        return [org.myopic_goal]

    elif regime == "spatial":
        return org.dept_goals[dept_idx]

    elif regime == "temporal":
        if S is None:
            S = DEFAULT_S.get(G, max(1, 250 // G))
        # Use the pre-generated random-permutation sequence stored in org.
        # Each period p = t // S maps to one goal in the sequence.
        period = t // S
        return [org.temporal_goals[period]]

    elif regime == "longjump":
        # Long-jump uses the same active goals as baseline (all G goals).
        return list(range(G))

    else:
        raise ValueError(f"Unknown regime '{regime}'. "
                         "Choose from: baseline, myopic, spatial, temporal, longjump.")
