# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get_true_otr_two_stage_simple.py
# Compute the true OTR for simplified two-stage datasets.
#
# Simple DGP properties that make this fully analytic (no Monte Carlo needed)
# ---------------------------------------------------------------------------
# - X2 = X1  (time-invariant baseline covariates)
# - Y  = f(X1 @ beta_Y1 + delta2[0]*A2 + Delta2[0]*A2*X1_bin) + noise
#   → Y depends only on X1 and A2, NOT on A1 or Y1
#
# Backward induction
# ------------------
# Stage 2:
#   Q2(a, X1_i) = E[Y | X1_i, A2=a]
#               = _mean_outcome(X1_with_int_i @ beta_Y1
#                               + delta2[0]*a + Delta2[0]*a*X1_bin_i,  flavor)
#   d2_star(i)  = argmax_a Q2(a, X1_i)
#
# Stage 1:
#   Q1(a, X1_i) = E[Y1(a) | X1_i]  +  V2*(X1_i)
#   where V2*(X1_i) = max_a2 Q2(a2, X1_i)  is the optimal stage-2 value.
#   Because V2* does not depend on A1, it cancels in the argmax:
#   d1_star(i)  = argmax_a E[Y1(a) | X1_i]
#
# Output columns (same format as get_true_otr_two_stage.py)
#   d1_star, d2_star, Q1_a0, Q1_a1, ..., Q2_a0, Q2_a1, ...
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helper: analytic E[Y | eta, flavor]
# ============================================================

def _mean_outcome(eta, flavor_Y):
    """
    Analytic conditional mean E[Y | eta, flavor_Y], dropping mean-zero noise.

    expo     : E[Y] = exp(eta)
    sigmoid  : E[Y] = 10 / (1 + exp(-eta))
    gamma    : E[Y] = f_gamma(eta) * 10 + 0.1
    lognormal: E[Y] = exp(eta + sigma^2/2),  sigma = 0.5
    """
    if flavor_Y == 'expo':
        return np.exp(eta)
    elif flavor_Y == 'sigmoid':
        return 10.0 / (1.0 + np.exp(-eta))
    elif flavor_Y == 'gamma':
        shape, scale = 2, 3
        return (np.exp(shape * eta) * np.exp(-np.exp(eta) / scale) /
                (math.gamma(shape) * scale**shape)) * 10 + 0.1
    elif flavor_Y == 'lognormal':
        sigma = 0.5
        return np.exp(eta + sigma**2 / 2.0)
    else:
        raise ValueError(f'Unknown flavor_Y: {flavor_Y}')


# ============================================================
# Main function
# ============================================================

def get_otr_simple(filename):
    """
    Compute true OTR for a simplified two-stage dataset via analytic backward induction.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')

    Returns
    -------
    DataFrame with columns: d1_star, d2_star, Q1_a0..., Q2_a0...
    """
    # ------------------------------------------------------------------
    # Load dataset and metadata
    # ------------------------------------------------------------------
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    matches = info[info['filename'] == filename]
    if len(matches) == 0:
        raise ValueError(f"No entry found for '{filename}' in _info_simple.csv")
    row = matches.iloc[0]

    p1, p2   = int(row['p1']), int(row['p2'])
    k1, k2   = int(row['k1']), int(row['k2'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params  = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)
    beta_Y1 = params['beta_Y1']   # (p1+1,)
    delta1  = params['delta1']    # (k1-1,)
    Delta1  = params['Delta1']    # (k1-1,)
    delta2  = params['delta2']    # (k2-1,) — only delta2[0] used
    Delta2  = params['Delta2']    # (k2-1,) — only Delta2[0] used

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1      = dat[X1_cols].values          # (n, p1)
    A1_obs  = dat['A1'].values             # observed stage 1 treatment
    n       = len(dat)

    X1_bin      = X1[:, -1]                           # binary effect modifier
    X1_with_int = np.column_stack([np.ones(n), X1])   # (n, p1+1)

    print(f"\nComputing OTR (simple) for: {filename}")
    print(f"  n={n}, p1={p1}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}")

    # ==================================================================
    # Stage 2 Q-function  (analytic — no Monte Carlo needed)
    # Q2(a2, X1_i, A1_i) = E[Y | X1_i, A1_i, do(A2=a2)]
    #   = _mean_outcome(X1_with_int_i @ beta_Y1
    #                   + delta1[A1_i-1]*0.5 + Delta1[A1_i-1]*0.5*X1_bin_i  (A1>0)
    #                   + delta2[0]*a2 + Delta2[0]*a2*X1_bin_i,  flavor)
    # Y depends on X1, A1 (halved effects), and A2. A1 is observed at stage 2
    # so Q2 is a deterministic function of observed quantities — no MC needed.
    # ==================================================================

    # A1 contribution to Y under observed A1 (halved effects)
    A1_contrib_obs = np.zeros(n)
    for a1 in range(1, k1):
        mask = (A1_obs == a1)
        A1_contrib_obs[mask] = delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin[mask]

    eta_Y2_base = X1_with_int @ beta_Y1 + A1_contrib_obs   # (n,) under A2=0, observed A1

    Q2_all = np.zeros((n, k2))
    Q2_all[:, 0] = _mean_outcome(eta_Y2_base, flavor_Y)
    for a2 in range(1, k2):
        eta_a2       = eta_Y2_base + delta2[0] * a2 + Delta2[0] * a2 * X1_bin
        Q2_all[:, a2] = _mean_outcome(eta_a2, flavor_Y)

    d2_star = np.argmax(Q2_all, axis=1)   # (n,)

    # ==================================================================
    # Stage 1 Q-function
    # Q1(a1, X1_i) = E[Y1(a1) | X1_i]  +  V2*(X1_i, A1=a1)
    # V2*(X1_i, A1=a1) = max_a2 Q2(a2, X1_i, A1=a1)
    # Since Y depends on A1, V2* now varies across candidate arms a1 —
    # we compute it separately for each hypothetical A1=a1.
    # Still fully analytic: no random variables to integrate over.
    # ==================================================================
    eta_Y1_base = X1_with_int @ beta_Y1   # (n,) under A1=0

    Q1_all = np.zeros((n, k1))
    for a1 in range(k1):
        # A1 contribution under hypothetical A1=a1 (halved effects)
        if a1 == 0:
            A1_contrib_hyp = np.zeros(n)
        else:
            A1_contrib_hyp = delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin

        # V2*(X1_i, A1=a1): optimal stage-2 value under hypothetical A1=a1
        eta_Y2_hyp = X1_with_int @ beta_Y1 + A1_contrib_hyp
        Q2_hyp     = np.zeros((n, k2))
        Q2_hyp[:, 0] = _mean_outcome(eta_Y2_hyp, flavor_Y)
        for a2 in range(1, k2):
            eta_a2        = eta_Y2_hyp + delta2[0] * a2 + Delta2[0] * a2 * X1_bin
            Q2_hyp[:, a2] = _mean_outcome(eta_a2, flavor_Y)
        V2_star_hyp = Q2_hyp.max(axis=1)   # (n,)

        # E[Y1(a1) | X1]
        if a1 == 0:
            eta_Y1_a1 = eta_Y1_base
        else:
            eta_Y1_a1 = eta_Y1_base + delta1[a1-1] + Delta1[a1-1] * X1_bin

        Q1_all[:, a1] = _mean_outcome(eta_Y1_a1, flavor_Y) + V2_star_hyp

    d1_star = np.argmax(Q1_all, axis=1)   # (n,)

    print(f"  d1_star distribution: {np.bincount(d1_star)}")
    print(f"  d2_star distribution: {np.bincount(d2_star)}")

    # ==================================================================
    # Assemble and save output
    # ==================================================================
    otr_dat = pd.DataFrame({'d1_star': d1_star, 'd2_star': d2_star})
    for a in range(k1):
        otr_dat[f'Q1_a{a}'] = Q1_all[:, a]
    for a in range(k2):
        otr_dat[f'Q2_a{a}'] = Q2_all[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_OTR.csv')
    otr_dat.to_csv(out_path, index=False)
    print(f"✓ Saved: {filename}_OTR.csv")

    return otr_dat


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    info = pd.read_csv(info_path)
    for _, row in info.iterrows():
        get_otr_simple(row['filename'])
