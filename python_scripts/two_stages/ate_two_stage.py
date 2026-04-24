# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ate_two_stage.py
# Two standalone functions:
#
#   true_ate(filename)      — true ATE from known DGP parameters
#   estimated_ate(filename) — AIPW-based ATE from DRE modified outcomes
#
# Definition
# ----------
#   ATE_k(a) = E[ Y_k(a) - Y_k(0) ]   (arm a vs baseline arm 0)
#
# True ATE
# --------
# Uses the analytic mean E[Y | eta, flavor] — noise terms are mean-zero and drop out.
# Stage 2 conditions on observed (A1, Y_1, X1, X2).
#
# Estimated ATE
# -------------
# Uses AIPW-style modified outcomes mu_hat from estimate_dre_two_stage:
#   ATE_k(a) = mean( mu_hat_k_a  -  mu_hat_k_a0 )
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
info_path    = os.path.join(datasets_dir, '_info.csv')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/2stages/tables')


# ============================================================
# Helper: analytic mean outcome for each flavor
# ============================================================

def _mean_outcome(eta, flavor):
    """
    E[Y | eta] for each flavor, ignoring mean-zero noise terms.

    expo     : Y = exp(eta) + N(0, 0.5)        → E[Y] = exp(eta)
    sigmoid  : Y = 10/(1+exp(-eta)) + N(0,0.5) → E[Y] = 10/(1+exp(-eta))
    gamma    : Y = f_gamma(eta)*10 + N(0,0.5) + 0.1
    lognormal: Y = exp(eta + N(0, 0.5))        → E[Y] = exp(eta + 0.125)
    """
    if flavor == 'expo':
        return np.exp(eta)
    elif flavor == 'sigmoid':
        return 10.0 / (1.0 + np.exp(-eta))
    elif flavor == 'gamma':
        shape, scale = 2, 3
        return (np.exp(shape * eta) * np.exp(-np.exp(eta) / scale) /
                (math.gamma(shape) * scale**shape)) * 10 + 0.1
    elif flavor == 'lognormal':
        sigma = 0.5
        return np.exp(eta + sigma**2 / 2.0)
    else:
        raise ValueError(f'Unknown flavor: {flavor}')


# ============================================================
# Function 1: true_ate
# ============================================================

def true_ate(filename, verbose=True):
    """
    Compute the true ATE using known DGP parameters.

    ATE_1(a) = E[ Y_1(a) - Y_1(0) ]  for a in {1, ..., k1-1}
    ATE_2(a) = E[ Y(a)   - Y(0)   ]  for a in {1, ..., k2-1}
               conditioning on observed (A1, Y_1, X1, X2)

    Parameters
    ----------
    filename : str   Base filename without extension
    verbose  : bool

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2' — each a dict {arm: float}
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    p1, p2   = int(row['p1']), int(row['p2'])
    k1, k2   = int(row['k1']), int(row['k2'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params  = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)
    beta_Y1 = params['beta_Y1']    # (p1+1,)
    delta1  = params['delta1']     # (k1-1,)
    Delta1  = params['Delta1']     # (k1-1,)
    beta_Y2 = params['beta_Y2']    # (1+p1+1+p2,)
    delta2  = params['delta2']     # (k2-1,)
    Delta2  = params['Delta2']     # (k2-1,)

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X2_cols = [c for c in dat.columns if c.startswith('X2_')]   # excludes Y_1
    X1      = dat[X1_cols].values   # (n, p1)
    X2      = dat[X2_cols].values   # (n, p2-1)
    A1      = dat['A1'].values
    Y1      = dat['Y_1'].values
    n       = len(dat)

    X1_bin      = X1[:, -1]                              # binary modifier (last col of X1)
    X1_with_int = np.column_stack([np.ones(n), X1])      # (n, p1+1)

    # ------------------------------------------------------------------
    # Stage 1 ATE
    # ------------------------------------------------------------------
    eta_Y1_base = X1_with_int @ beta_Y1   # (n,)  under A1=0

    ATE_1 = {}
    for a in range(1, k1):
        eta_Y1_a = eta_Y1_base + delta1[a-1] + Delta1[a-1] * X1_bin
        ATE_1[a] = float(np.mean(
            _mean_outcome(eta_Y1_a,    flavor_Y) -
            _mean_outcome(eta_Y1_base, flavor_Y)
        ))

    # ------------------------------------------------------------------
    # Stage 2 ATE  (conditional on observed A1, Y_1, X1, X2)
    # Feature order: [1, X1, A1, Y_1, X2]  — matches beta_Y2 structure
    # ------------------------------------------------------------------
    X_combined = np.column_stack([np.ones(n), X1, A1, Y1, X2])   # (n, 1+p1+1+p2)
    eta_Y_base = X_combined @ beta_Y2                              # (n,)  under A2=0

    Y1_bin = (Y1 > np.median(Y1)).astype(float)   # binary modifier for Delta2

    ATE_2 = {}
    for a in range(1, k2):
        eta_Y_a = eta_Y_base + delta2[a-1] + Delta2[a-1] * Y1_bin
        ATE_2[a] = float(np.mean(
            _mean_outcome(eta_Y_a,    flavor_Y) -
            _mean_outcome(eta_Y_base, flavor_Y)
        ))

    if verbose:
        print(f"\n{'='*55}")
        print(f"True ATE: {filename}")
        print(f"{'='*55}")
        print(f"\n  Stage 1  (A1 vs A1=0)")
        for a, ate in ATE_1.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")
        print(f"\n  Stage 2  (A2 vs A2=0, conditional on observed history)")
        for a, ate in ATE_2.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


# ============================================================
# Function 2: estimated_ate
# ============================================================

def estimated_ate(filename, verbose=True):
    """
    Estimate ATE using AIPW-style modified outcomes from estimate_dre_two_stage.

    ATE_1(a) = mean( mu_hat_1_a  -  mu_hat_1_a0 )
    ATE_2(a) = mean( mu_hat_2_a  -  mu_hat_2_a0 )

    Parameters
    ----------
    filename : str   Base filename without extension
    verbose  : bool

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2' — each a dict {arm: float}
    """
    info    = pd.read_csv(info_path)
    row     = info[info['filename'] == filename].iloc[0]
    k1, k2  = int(row['k1']), int(row['k2'])

    dre = pd.read_csv(os.path.join(datasets_dir, f'{filename}_DRE.csv'))

    ATE_1 = {}
    for a in range(1, k1):
        ATE_1[a] = float(np.mean(dre[f'mu_hat_1_a{a}'] - dre['mu_hat_1_a0']))

    ATE_2 = {}
    for a in range(1, k2):
        ATE_2[a] = float(np.mean(dre[f'mu_hat_2_a{a}'] - dre['mu_hat_2_a0']))

    if verbose:
        print(f"\n{'='*55}")
        print(f"Estimated ATE (DRE): {filename}")
        print(f"{'='*55}")
        print(f"\n  Stage 1  (A1 vs A1=0)")
        for a, ate in ATE_1.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")
        print(f"\n  Stage 2  (A2 vs A2=0)")
        for a, ate in ATE_2.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


# ============================================================
# Run over all datasets in _info.csv
# ============================================================
if __name__ == '__main__':
    info    = pd.read_csv(info_path)
    results = []

    for _, row in info.iterrows():
        fname = row['filename']
        try:
            t = true_ate(fname,      verbose=False)
            e = estimated_ate(fname, verbose=False)

            for stage, arms_t, arms_e in [
                (1, t['ATE_1'], e['ATE_1']),
                (2, t['ATE_2'], e['ATE_2']),
            ]:
                for a in arms_t:
                    results.append({
                        'filename': fname,
                        'k':        row['k1'],
                        'flavor_Y': row['flavor_Y'],
                        'stage':    stage,
                        'arm':      a,
                        'ATE_true': arms_t[a],
                        'ATE_est':  arms_e.get(a, np.nan),
                    })
        except FileNotFoundError as exc:
            print(f"Skipping {fname}: {exc}")

    summary  = pd.DataFrame(results)
    out_path = os.path.join(tables_dir, '_ate_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ ATE summary saved to _ate_summary.csv")
    print(summary.groupby(['k', 'flavor_Y', 'stage', 'arm'])[
        ['ATE_true', 'ATE_est']
    ].mean().round(4))
