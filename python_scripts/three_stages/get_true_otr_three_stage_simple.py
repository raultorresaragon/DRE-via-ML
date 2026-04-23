# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get_true_otr_three_stage_simple.py
# Compute the true OTR for simplified three-stage datasets via analytic backward induction.
#
# Simple DGP properties (no MC needed)
# -------------------------------------
# - X2 = X3 = X1  (time-invariant baseline covariates)
# - Y1 = f(X1, A1)
# - Y2 = f(X1, A1, A2)  — depends on A1 (halved) and A2 (full)
# - Y  = f(X1, A1, A2, A3)  — depends on A1 (0.25), A2 (0.5), A3 (full)
# - None of Y1, Y2, Y are structural functions of intermediate random outcomes,
#   so backward induction is fully analytic (no random variables to integrate out).
#
# Backward induction
# ------------------
# Stage 3:
#   Q3(a3, X1_i, A1_i, A2_i) = E[Y | X1_i, A1_i, A2_i, do(A3=a3)]
#   d3_star(i) = argmax_{a3} Q3(a3, X1_i, A1_obs_i, A2_obs_i)
#
# Stage 2:
#   Q2(a2, X1_i, A1_i) = E[Y2 | X1_i, A1_i, do(A2=a2)]
#                       + V3*(X1_i, A1_i, A2=a2)
#   where V3* = max_{a3} Q3(a3, X1_i, A1_i, A2=a2)
#   d2_star(i) = argmax_{a2} Q2(a2, X1_i, A1_obs_i)
#
# Stage 1:
#   Q1(a1, X1_i) = E[Y1(a1) | X1_i]  +  V2*(X1_i, A1=a1)
#   where V2* = max_{a2} Q2(a2, X1_i, A1=a1)
#   d1_star(i) = argmax_{a1} Q1(a1, X1_i)
#
# Output columns:
#   d1_star, d2_star, d3_star, Q1_a0..., Q2_a0..., Q3_a0...
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helper: analytic E[Y | eta, flavor]
# ============================================================

def _mean_outcome(eta, flavor_Y):
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

def get_otr_3stage_simple(filename):
    """
    Compute true OTR for a simplified three-stage dataset via analytic backward induction.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')

    Returns
    -------
    DataFrame with columns: d1_star, d2_star, d3_star, Q1_a*, Q2_a*, Q3_a*
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
    k3       = int(row['k3'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params  = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, k3=k3, seed=seed)
    beta_Y1 = params['beta_Y1']   # (p1+1,)
    delta1  = params['delta1']    # (k1-1,)
    Delta1  = params['Delta1']    # (k1-1,)
    delta2  = params['delta2']    # only delta2[0] used
    Delta2  = params['Delta2']    # only Delta2[0] used
    delta3  = params['delta3']    # only delta3[0] used
    Delta3  = params['Delta3']    # only Delta3[0] used

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1      = dat[X1_cols].values          # (n, p1)
    A1_obs  = dat['A1'].values
    A2_obs  = dat['A2'].values
    n       = len(dat)

    X1_bin      = X1[:, -1]
    X1_with_int = np.column_stack([np.ones(n), X1])

    print(f"\nComputing OTR (3-stage simple) for: {filename}")
    print(f"  n={n}, p1={p1}, k1={k1}, k2={k2}, k3={k3}, flavor_Y={flavor_Y}")

    # ------------------------------------------------------------------
    # Helper: A1 contribution to eta at given attenuation weight
    # ------------------------------------------------------------------
    def _a1_contrib(A1_vals, weight):
        contrib = np.zeros(n)
        for a1 in range(1, k1):
            mask          = (A1_vals == a1)
            contrib[mask] = (delta1[a1-1] * weight
                             + Delta1[a1-1] * weight * X1_bin[mask])
        return contrib

    # ==================================================================
    # Stage 3 Q-function  (uses observed A1, A2)
    # eta_Y_base = X1 @ beta_Y1 + A1_contrib(obs, 0.25) + A2_contrib(obs, 0.5)
    # Q3(a3) = E[Y | eta_Y_base + delta3[0]*a3 + Delta3[0]*a3*X1_bin]
    # ==================================================================
    eta_Y_base = (X1_with_int @ beta_Y1
                  + _a1_contrib(A1_obs, 0.25)
                  + delta2[0] * 0.5 * A2_obs + Delta2[0] * 0.5 * A2_obs * X1_bin)

    Q3_all = np.zeros((n, k3))
    Q3_all[:, 0] = _mean_outcome(eta_Y_base, flavor_Y)
    for a3 in range(1, k3):
        Q3_all[:, a3] = _mean_outcome(
            eta_Y_base + delta3[0] * a3 + Delta3[0] * a3 * X1_bin, flavor_Y)

    d3_star = np.argmax(Q3_all, axis=1)

    # ==================================================================
    # Stage 2 Q-function  (uses observed A1, loops over hypothetical A2)
    # Q2(a2, A1_obs) = E[Y2 | X1, A1_obs, A2=a2]  +  V3*(X1, A1_obs, A2=a2)
    # ==================================================================
    eta_Y2_a1base = X1_with_int @ beta_Y1 + _a1_contrib(A1_obs, 0.5)

    Q2_all = np.zeros((n, k2))
    for a2 in range(k2):
        # E[Y2 | A2=a2, observed A1]
        eta_Y2_a2 = eta_Y2_a1base + delta2[0] * a2 + Delta2[0] * a2 * X1_bin
        EY2_a2    = _mean_outcome(eta_Y2_a2, flavor_Y)

        # V3*(A2=a2, obs A1): max over a3 of Q3 under hypothetical A2=a2
        eta_Y_hyp2 = (X1_with_int @ beta_Y1
                      + _a1_contrib(A1_obs, 0.25)
                      + delta2[0] * 0.5 * a2 + Delta2[0] * 0.5 * a2 * X1_bin)
        Q3_hyp = np.zeros((n, k3))
        Q3_hyp[:, 0] = _mean_outcome(eta_Y_hyp2, flavor_Y)
        for a3 in range(1, k3):
            Q3_hyp[:, a3] = _mean_outcome(
                eta_Y_hyp2 + delta3[0] * a3 + Delta3[0] * a3 * X1_bin, flavor_Y)
        V3_star_a2 = Q3_hyp.max(axis=1)

        Q2_all[:, a2] = EY2_a2 + V3_star_a2

    d2_star = np.argmax(Q2_all, axis=1)

    # ==================================================================
    # Stage 1 Q-function  (loops over hypothetical A1 values)
    # Q1(a1, X1_i) = E[Y1(a1) | X1_i]  +  V2*(X1_i, A1=a1)
    # ==================================================================
    eta_Y1_base = X1_with_int @ beta_Y1   # under A1=0

    Q1_all = np.zeros((n, k1))
    for a1 in range(k1):
        # E[Y1(a1) | X1]  — full A1 effect on Y1
        if a1 == 0:
            eta_Y1_a1  = eta_Y1_base
            A1c_half   = np.zeros(n)
            A1c_qtr    = np.zeros(n)
        else:
            eta_Y1_a1  = eta_Y1_base + delta1[a1-1] + Delta1[a1-1] * X1_bin
            A1c_half   = delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin
            A1c_qtr    = delta1[a1-1] * 0.25 + Delta1[a1-1] * 0.25 * X1_bin
        EY1_a1 = _mean_outcome(eta_Y1_a1, flavor_Y)

        # V2*(X1, A1=a1): max over a2 of Q2 under hypothetical A1=a1
        Q2_hyp = np.zeros((n, k2))
        for a2 in range(k2):
            # E[Y2 | X1, A1=a1, A2=a2]
            eta_Y2_hyp = (X1_with_int @ beta_Y1 + A1c_half
                          + delta2[0] * a2 + Delta2[0] * a2 * X1_bin)
            EY2_hyp    = _mean_outcome(eta_Y2_hyp, flavor_Y)

            # V3*(X1, A1=a1, A2=a2)
            eta_Y_hyp = (X1_with_int @ beta_Y1 + A1c_qtr
                         + delta2[0] * 0.5 * a2 + Delta2[0] * 0.5 * a2 * X1_bin)
            Q3_hyp2 = np.zeros((n, k3))
            Q3_hyp2[:, 0] = _mean_outcome(eta_Y_hyp, flavor_Y)
            for a3 in range(1, k3):
                Q3_hyp2[:, a3] = _mean_outcome(
                    eta_Y_hyp + delta3[0] * a3 + Delta3[0] * a3 * X1_bin, flavor_Y)
            V3_star_hyp = Q3_hyp2.max(axis=1)

            Q2_hyp[:, a2] = EY2_hyp + V3_star_hyp

        V2_star_a1    = Q2_hyp.max(axis=1)
        Q1_all[:, a1] = EY1_a1 + V2_star_a1

    d1_star = np.argmax(Q1_all, axis=1)

    print(f"  d1_star distribution: {np.bincount(d1_star)}")
    print(f"  d2_star distribution: {np.bincount(d2_star)}")
    print(f"  d3_star distribution: {np.bincount(d3_star)}")

    # ==================================================================
    # Assemble and save output
    # ==================================================================
    otr_dat = pd.DataFrame({
        'd1_star': d1_star,
        'd2_star': d2_star,
        'd3_star': d3_star,
    })
    for a in range(k1):
        otr_dat[f'Q1_a{a}'] = Q1_all[:, a]
    for a in range(k2):
        otr_dat[f'Q2_a{a}'] = Q2_all[:, a]
    for a in range(k3):
        otr_dat[f'Q3_a{a}'] = Q3_all[:, a]

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
        get_otr_3stage_simple(row['filename'])
