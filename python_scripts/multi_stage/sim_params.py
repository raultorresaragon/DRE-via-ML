# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: sim_params.py
# Date: 2026-04-02
# Note: Factory function for auto-sizing simulation parameters given k1, k2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np


def make_sim_params(p1, p2, k1, k2, seed=None):
    """
    Generate correctly-sized simulation parameters for a two-stage DTR.

    All array dimensions are derived from p1, p2, k1, k2 so that changing
    the number of treatment levels never requires manual resizing.

    Parameters
    ----------
    p1 : int   Number of stage 1 covariates
    p2 : int   Number of stage 2 covariates (includes intermediate outcome Y_1)
    k1 : int   Number of stage 1 treatment levels
    k2 : int   Number of stage 2 treatment levels
    seed : int or None   Random seed (None = do not set)

    Returns
    -------
    dict with keys:
        beta_A1    (p1+1,  k1-1)     Stage 1 treatment model coefficients
        beta_A2    (p1+p2+2, k2-1)   Stage 2 treatment model coefficients
        gamma_stay scalar             Stay-probability parameter
        delta1     (k1-1,)            A1 main effects on Y_1
        beta_Y1    (p1+1,)            X1 effects on Y_1
        Delta1     (k1-1,)            A1 × X1_bin interaction on Y_1
        delta2     (k2-1,)            A2 main effects on Y
        beta_Y2    (1+p1+1+p2,)       [1,X1,A1,X2] effects on Y
        Delta2     (k2-1,)            A2 × X1_bin interaction on Y
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    params = {
        # Treatment models
        'beta_A1':    rng.uniform(-0.5,  0.5,  size=(p1 + 1,          k1 - 1)),
        'beta_A2':    rng.uniform(-0.5,  0.5,  size=(p1 + p2 + 2,     k2 - 1)),
        'gamma_stay': 0.5,

        # Intermediate outcome (Y_1) model: Y_1 = g(X1 @ beta_Y1 + delta1*A1 + Delta1*(A1*X_p))
        'delta1':  np.array([0.6, 0.4, 0.75, 0.17])[:k1 - 1],
        'beta_Y1': rng.uniform(-0.3,  0.3,  size=(p1 + 1,)),
        'Delta1':  np.array([-1.2, -1.0, -1.0,  0.8])[:k1 - 1],

        # Final outcome (Y) model: Y = g([X1,A1,Y_1,X2] @ beta_Y2 + delta2*A2 + Delta2*(A2*X_p))
        'delta2':  np.array([0.5, 0.3, 0.50, 0.19])[:k2 - 1],
        'beta_Y2': rng.uniform(-0.5,  0.5,  size=(1 + p1 + 1 + p2,)),
        'Delta2':  np.array([-0.8, -0.7, -0.6,  0.3])[:k2 - 1],
    }

    # A1 main effect on Y is zero by default (index p1+1 in beta_Y2)
    # A1's effect on Y is assumed to be fully mediated through Y_1
    params['beta_Y2'][p1 + 1] = 0.0

    return params


def print_param_shapes(params, p1, p2, k1, k2):
    """Print parameter shapes alongside their expected dimensions."""
    expected = {
        'beta_A1':    f'({p1+1}, {k1-1})',
        'beta_A2':    f'({p1+p2+2}, {k2-1})',
        'gamma_stay': 'scalar',
        'delta1':     f'({k1-1},)',
        'beta_Y1':    f'({p1+1},)',
        'Delta1':     f'({k1-1},)',
        'delta2':     f'({k2-1},)',
        'beta_Y2':    f'({1+p1+1+p2},)',
        'Delta2':     f'({k2-1},)',
    }
    print(f"Parameter shapes  [p1={p1}, p2={p2}, k1={k1}, k2={k2}]")
    print("-" * 52)
    for name, exp in expected.items():
        val = params[name]
        actual = 'scalar' if np.isscalar(val) else str(np.array(val).shape)
        status = '✓' if actual == exp else '✗'
        print(f"  {status}  {name:<12}  actual={actual:<16}  expected={exp}")
