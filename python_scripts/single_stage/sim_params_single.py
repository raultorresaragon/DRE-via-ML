# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sim_params_single.py
# Factory for single-stage DGP parameters.
#
# Returns correctly-sized simulation parameters for a single-stage DTR.
# All array dimensions are derived from p and k so that changing
# the number of treatment levels or covariates never requires manual resizing.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np


def make_sim_params_single(p, k, seed=None, flavor_Y=None):
    """
    Generate correctly-sized simulation parameters for a single-stage DTR.

    Parameters
    ----------
    p        : int          Number of covariates
    k        : int          Number of treatment levels
    seed     : int or None  Random seed (None = do not set)
    flavor_Y : str or None  'expo' uses narrower delta range; else wider

    Returns
    -------
    dict with keys:
        beta_A : (p+1, k-1)   Covariate (+ intercept) effects on treatment propensity
        beta_Y : (p+1,)       Covariate (+ intercept) effects on outcome
        delta  : (k-1,)       Main treatment effects on Y  [positive]
        Delta  : (k-1,)       Treatment × X_bin interaction on Y  [negative]
    """
    rng = np.random.default_rng(seed)

    # Delta range: narrower for expo to avoid extreme Y values under exponential link
    _lo, _hi = (0.3, 1.0) if flavor_Y == 'expo' else (0.5, 2.5)
    scale = 1.0 / np.sqrt(p)

    return {
        'beta_A': rng.uniform(-0.5, 0.5, size=(p + 1, k - 1)),
        'beta_Y': rng.uniform(-scale, scale, size=(p + 1,)),
        'delta':  rng.uniform(_lo, _hi, size=(k - 1,)),    # positive
        'Delta':  -rng.uniform(_lo, _hi, size=(k - 1,)),   # negative
    }


def print_param_shapes_single(params, p, k):
    """Print parameter shapes alongside their expected dimensions."""
    expected = {
        'beta_A': f'({p+1}, {k-1})',
        'beta_Y': f'({p+1},)',
        'delta':  f'({k-1},)',
        'Delta':  f'({k-1},)',
    }
    print(f"Parameter shapes  [p={p}, k={k}]")
    print("-" * 52)
    for name, exp in expected.items():
        val    = params[name]
        actual = 'scalar' if np.isscalar(val) else str(np.array(val).shape)
        status = '✓' if actual == exp else '✗'
        print(f"  {status}  {name:<12}  actual={actual:<16}  expected={exp}")
