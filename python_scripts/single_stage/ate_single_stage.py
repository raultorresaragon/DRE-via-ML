# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ate_single_stage.py
# True and estimated ATEs for the single-stage DGP.
#
# Reads metadata from _info_single.csv.
# Saves summary to tables/_ate_single_summary.csv.
#
# ATE definition
# --------------
#   ATE(a) = E[ Y(a) - Y(0) ]
#
# Functions
# ---------
#   true_ate_single     : analytic true ATE via known DGP params
#   naive_ate_single    : observed mean difference, unadjusted for confounding
#   estimated_ate_single: estimated ATE from DRE AIPW modified outcomes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from sim_params_single import make_sim_params_single

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/tables')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# Helper: analytic mean outcome for each flavor
# ============================================================

def _mean_outcome(eta, flavor):
    """
    E[Y | eta] for each flavor, ignoring mean-zero noise terms.

    expo     : Y = exp(eta) + N(0, 0.5)             -> E[Y] = exp(eta)
    sigmoid  : Y = 10/(1+exp(-eta)) + N(0, 0.5)     -> E[Y] = 10/(1+exp(-eta))
    gamma    : Y = f_gamma(eta)*10 + N(0,0.5) + 0.1  (shape=2, scale=3)
    lognormal: Y = exp(eta + N(0, 0.5))              -> E[Y] = exp(eta + 0.5^2/2)
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
# Function 1: true_ate_single
# ============================================================

def true_ate_single(filename, verbose=True):
    """
    Compute the true ATE using known DGP parameters for the single-stage DGP.

      ATE(a) = E[ Y(a) - Y(0) ]
             = mean_i( E[Y(a)|X_i] - E[Y(0)|X_i] )

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_expo_0')
    verbose  : bool

    Returns
    -------
    dict with key 'ATE' — dict {arm: float}
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    p, k     = int(row['p']), int(row['k'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params = make_sim_params_single(p=p, k=k, seed=seed, flavor_Y=flavor_Y)
    beta_Y = params['beta_Y']   # (p+1,)
    delta  = params['delta']    # (k-1,)  positive
    Delta  = params['Delta']    # (k-1,)  negative

    X_cols     = [c for c in dat.columns if c.startswith('X')]
    X          = dat[X_cols].values          # (n, p)
    n          = len(dat)

    X_bin      = X[:, -1]                          # binary modifier (last col of X)
    X_with_int = np.column_stack([np.ones(n), X])  # (n, p+1)

    eta_base = X_with_int @ beta_Y   # (n,) under A=0

    ATE = {}
    for a in range(1, k):
        eta_a  = eta_base + delta[a-1] + Delta[a-1] * X_bin
        ATE[a] = float(np.mean(
            _mean_outcome(eta_a,    flavor_Y) -
            _mean_outcome(eta_base, flavor_Y)
        ))

    if verbose:
        print(f"\n{'='*55}")
        print(f"True ATE (single-stage DGP): {filename}")
        print(f"{'='*55}")
        for a, ate in ATE.items():
            print(f"  ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE': ATE}


# ============================================================
# Function 2: naive_ate_single
# ============================================================

def naive_ate_single(filename, verbose=True):
    """
    Naive ATE: observed mean difference, unadjusted for confounding.

      ATE_naive(a) = mean(Y | A=a) - mean(Y | A=0)

    This is biased whenever X confounds the treatment-outcome relationship.
    Serves as a benchmark to contrast against the DRE-adjusted estimate.

    Parameters
    ----------
    filename : str
    verbose  : bool

    Returns
    -------
    dict with key 'ATE' — dict {arm: float}
    """
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]
    k    = int(row['k'])

    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    A    = dat['A'].values
    Y    = dat['Y'].values

    base = Y[A == 0].mean()
    ATE  = {a: float(Y[A == a].mean() - base) for a in range(1, k)}

    if verbose:
        print(f"\n{'='*55}")
        print(f"Naive ATE (unadjusted): {filename}")
        print(f"{'='*55}")
        for a, ate in ATE.items():
            print(f"  ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE': ATE}


# ============================================================
# Function 3: estimated_ate_single
# ============================================================

def estimated_ate_single(filename, verbose=True):
    """
    Estimate ATE using AIPW-style modified outcomes from estimate_dre_single_stage.

      ATE(a) = mean( mu_hat_a - mu_hat_a0 )

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_expo_0')
    verbose  : bool

    Returns
    -------
    dict with key 'ATE' — dict {arm: float}
    """
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]
    k    = int(row['k'])

    dre = pd.read_csv(os.path.join(datasets_dir, f'{filename}_DRE.csv'))

    ATE = {a: float(np.mean(dre[f'mu_hat_a{a}'] - dre['mu_hat_a0']))
           for a in range(1, k)}

    if verbose:
        print(f"\n{'='*55}")
        print(f"Estimated ATE (DRE, single-stage): {filename}")
        print(f"{'='*55}")
        for a, ate in ATE.items():
            print(f"  ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE': ATE}


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    os.makedirs(tables_dir, exist_ok=True)

    info    = pd.read_csv(info_path)
    results = []

    for _, row in info.iterrows():
        fname = row['filename']
        try:
            t = true_ate_single(fname,      verbose=False)
            n = naive_ate_single(fname,     verbose=False)
            e = estimated_ate_single(fname, verbose=False)

            for a in t['ATE']:
                ate_true  = t['ATE'][a]
                ate_naive = n['ATE'].get(a, float('nan'))
                ate_est   = e['ATE'].get(a, float('nan'))
                results.append({
                    'filename':  fname,
                    'k':         row['k'],
                    'flavor_Y':  row['flavor_Y'],
                    'arm':       a,
                    'ATE_true':  ate_true,
                    'ATE_naive': ate_naive,
                    'ATE_est':   ate_est,
                    'bias_naive': ate_naive - ate_true,
                    'bias_est':   ate_est   - ate_true,
                })
        except FileNotFoundError as exc:
            print(f"Skipping {fname}: {exc}")

    summary  = pd.DataFrame(results)
    out_path = os.path.join(tables_dir, '_ate_single_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ ATE summary saved to tables/_ate_single_summary.csv")
    print(summary.groupby(['k', 'flavor_Y', 'arm'])[
        ['ATE_true', 'ATE_naive', 'ATE_est', 'bias_naive', 'bias_est']
    ].mean().round(4))
