# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: sim_params.py
# Date: 2026-04-02
# Note: Factory function for auto-sizing simulation parameters given k1, k2, k3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np


def make_sim_params(p1, p2, k1, k2, k3=None, seed=None, flavor_Y=None):
    """
    Generate correctly-sized simulation parameters for a two- or three-stage DTR.

    All array dimensions are derived from p1, p2, k1, k2 (and k3 if provided)
    so that changing the number of treatment levels never requires manual resizing.

    delta / Delta parameters are drawn randomly from rng.uniform so that each
    replication (seed) has distinct effect sizes, matching the two-stage convention.

    Parameters
    ----------
    p1       : int          Number of stage 1 covariates
    p2       : int          Number of stage 2 covariates (includes intermediate outcome Y_1)
    k1       : int          Number of stage 1 treatment levels
    k2       : int          Number of stage 2 treatment levels
    k3       : int or None  Number of stage 3 treatment levels (None = two-stage only)
    seed     : int or None  Random seed (None = do not set)
    flavor_Y : str or None  DGP flavor — 'expo' uses a narrower delta range to avoid
                            extreme Y values under the exponential link; all others use
                            the wider range.

    Returns
    -------
    dict with keys:
        beta_A1    (p1+1,  k1-1)       Stage 1 treatment model coefficients
        beta_A2    (p1+p2+2, k2-1)     Stage 2 treatment model coefficients
        gamma_stay scalar               Stay-probability parameter
        delta1     (k1-1,)             A1 main effects on Y_1            [positive]
        beta_Y1    (p1+1,)             X1 effects on Y_1                 [scaled by 1/√p1]
        Delta1     (k1-1,)             A1 × X1_bin interaction on Y_1   [negative]
        delta2     (k2-1,)             A2 main effects on Y_2            [positive]
        beta_Y2    (1+p1+1+p2,)        [1,X1,A1,X2] effects on Y        [scaled by 0.5/√p1]
        Delta2     (k2-1,)             A2 × X1_bin interaction on Y_2   [negative]
        --- three-stage only (when k3 is not None) ---
        beta_A3    (p1+1, k3-1)        Stage 3 treatment model coefficients
        delta3     (k3-1,)             A3 main effects on final Y        [positive]
        Delta3     (k3-1,)             A3 × X1_bin interaction on Y     [negative]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Delta range: narrower for expo to avoid extreme Y values under exponential link
    _lo, _hi = (0.3, 1.0) if flavor_Y == 'expo' else (0.5, 2.5)

    params = {
        # Treatment models
        'beta_A1':    rng.uniform(-0.5, 0.5, size=(p1 + 1,      k1 - 1)),
        'beta_A2':    rng.uniform(-0.5, 0.5, size=(p1 + p2 + 2, k2 - 1)),
        'gamma_stay': 0.5,

        # Intermediate outcome (Y_1): Y_1 = g(X1 @ beta_Y1 + delta1*A1 + Delta1*(A1*X_bin))
        'delta1':  rng.uniform(_lo, _hi, size=(k1 - 1,)),              # positive
        'beta_Y1': rng.uniform(-1/np.sqrt(p1), 1/np.sqrt(p1), size=(p1 + 1,)),
        'Delta1':  -rng.uniform(_lo, _hi, size=(k1 - 1,)),             # negative

        # Stage-2 outcome (Y_2 / Y for two-stage)
        'delta2':  rng.uniform(_lo, _hi, size=(k2 - 1,)),              # positive
        'beta_Y2': rng.uniform(-0.5/np.sqrt(p1), 0.5/np.sqrt(p1), size=(1 + p1 + 1 + p2,)),
        'Delta2':  -rng.uniform(_lo, _hi, size=(k2 - 1,)),             # negative
    }

    # A1 main effect on Y is zero by default (index p1+1 in beta_Y2)
    # A1's effect on Y is assumed to be fully mediated through Y_1
    params['beta_Y2'][p1 + 1] = 0.0

    # Three-stage parameters (only when k3 is provided)
    if k3 is not None:
        params['beta_A3'] = rng.uniform(-0.5, 0.5, size=(p1 + 1, k3 - 1))
        params['delta3']  = rng.uniform(_lo, _hi, size=(k3 - 1,))      # positive
        params['Delta3']  = -rng.uniform(_lo, _hi, size=(k3 - 1,))     # negative

    return params


def print_param_shapes(params, p1, p2, k1, k2, k3=None):
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
    if k3 is not None:
        expected['beta_A3'] = f'({p1+1}, {k3-1})'
        expected['delta3']  = f'({k3-1},)'
        expected['Delta3']  = f'({k3-1},)'

    label = f"p1={p1}, p2={p2}, k1={k1}, k2={k2}" + (f", k3={k3}" if k3 else "")
    print(f"Parameter shapes  [{label}]")
    print("-" * 52)
    for name, exp in expected.items():
        val = params[name]
        actual = 'scalar' if np.isscalar(val) else str(np.array(val).shape)
        status = '✓' if actual == exp else '✗'
        print(f"  {status}  {name:<12}  actual={actual:<16}  expected={exp}")
