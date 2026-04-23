# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ate_three_stage_simple.py
# True and estimated ATEs for the simplified three-stage DGP.
#
# Reads metadata from _info_simple.csv.
# Saves summary to tables/_ate_simple_summary.csv.
#
# ATE definitions
# ---------------
#   ATE_1(a) = E[ Y1(a) - Y1(0) ]
#   ATE_2(a) = E[ Y2(a) - Y2(0) ]   (marginal over observed X1, A1)
#   ATE_3(a) = E[ Y(a)  - Y(0)  ]   (marginal over observed X1, A1, A2)
#
# True ATEs are computed analytically using DGP parameters.
# Naive ATEs are unadjusted observed mean differences.
# Estimated ATEs use mu_hat columns from the DRE output file.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/3stages/tables')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helper: analytic mean outcome for each flavor
# ============================================================

def _mean_outcome(eta, flavor):
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
# Function 1: true_ate_simple
# ============================================================

def true_ate_simple(filename, verbose=True):
    """
    Compute the true ATE using known DGP parameters for the simple three-stage DGP.

    Stage 1: ATE_1(a) = E[ Y1(a) - Y1(0) ]  (marginal over X1)
    Stage 2: ATE_2(a) = E[ Y2(a) - Y2(0) ]  (marginal over X1, observed A1)
    Stage 3: ATE_3(a) = E[ Y(a)  - Y(0)  ]  (marginal over X1, observed A1, A2)

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2', 'ATE_3' -- each a dict {arm: float}
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    p1, p2   = int(row['p1']), int(row['p2'])
    k1, k2   = int(row['k1']), int(row['k2'])
    k3       = int(row['k3'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params  = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, k3=k3, seed=seed)
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']
    delta2  = params['delta2']
    Delta2  = params['Delta2']
    delta3  = params['delta3']
    Delta3  = params['Delta3']

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1      = dat[X1_cols].values
    A1_obs  = dat['A1'].values
    A2_obs  = dat['A2'].values
    n       = len(dat)

    X1_bin      = X1[:, -1]
    X1_with_int = np.column_stack([np.ones(n), X1])

    # ------------------------------------------------------------------
    # Stage 1 ATE
    # ------------------------------------------------------------------
    eta_Y1_base = X1_with_int @ beta_Y1

    ATE_1 = {}
    for a in range(1, k1):
        eta_a = eta_Y1_base + delta1[a-1] + Delta1[a-1] * X1_bin
        ATE_1[a] = float(np.mean(
            _mean_outcome(eta_a,        flavor_Y) -
            _mean_outcome(eta_Y1_base,  flavor_Y)
        ))

    # ------------------------------------------------------------------
    # Stage 2 ATE  (conditioning on observed A1; A1 halved in Y2)
    # ------------------------------------------------------------------
    A1_contrib = np.zeros(n)
    for a1 in range(1, k1):
        mask             = (A1_obs == a1)
        A1_contrib[mask] = delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin[mask]

    eta_Y2_base = X1_with_int @ beta_Y1 + A1_contrib   # under A2=0

    ATE_2 = {}
    for a in range(1, k2):
        eta_a = eta_Y2_base + delta2[0] * a + Delta2[0] * a * X1_bin
        ATE_2[a] = float(np.mean(
            _mean_outcome(eta_a,        flavor_Y) -
            _mean_outcome(eta_Y2_base,  flavor_Y)
        ))

    # ------------------------------------------------------------------
    # Stage 3 ATE  (conditioning on observed A1, A2; A1 at 0.25, A2 at 0.5)
    # ------------------------------------------------------------------
    A1_contrib_q = np.zeros(n)
    for a1 in range(1, k1):
        mask               = (A1_obs == a1)
        A1_contrib_q[mask] = delta1[a1-1] * 0.25 + Delta1[a1-1] * 0.25 * X1_bin[mask]

    eta_Y_base = (X1_with_int @ beta_Y1 + A1_contrib_q
                  + delta2[0] * 0.5 * A2_obs + Delta2[0] * 0.5 * A2_obs * X1_bin)

    ATE_3 = {}
    for a in range(1, k3):
        eta_a = eta_Y_base + delta3[0] * a + Delta3[0] * a * X1_bin
        ATE_3[a] = float(np.mean(
            _mean_outcome(eta_a,       flavor_Y) -
            _mean_outcome(eta_Y_base,  flavor_Y)
        ))

    if verbose:
        print(f"\n{'='*55}")
        print(f"True ATE (3-stage simple): {filename}")
        print(f"{'='*55}")
        for stage, ates in [(1, ATE_1), (2, ATE_2), (3, ATE_3)]:
            print(f"\n  Stage {stage}  (A{stage} vs A{stage}=0)")
            for a, ate in ates.items():
                print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2, 'ATE_3': ATE_3}


# ============================================================
# Function 2: naive_ate_simple
# ============================================================

def naive_ate_simple(filename, verbose=True):
    """
    Naive ATE: unadjusted observed mean differences at each stage.

      ATE_1_naive(a) = mean(Y1 | A1=a) - mean(Y1 | A1=0)
      ATE_2_naive(a) = mean(Y2 | A2=a) - mean(Y2 | A2=0)
      ATE_3_naive(a) = mean(Y  | A3=a) - mean(Y  | A3=0)

    Biased due to confounding by X1.  Serves as a baseline benchmark.

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2', 'ATE_3' -- each a dict {arm: float}
    """
    info   = pd.read_csv(info_path)
    row    = info[info['filename'] == filename].iloc[0]
    k1, k2 = int(row['k1']), int(row['k2'])
    k3     = int(row['k3'])

    dat = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    A1  = dat['A1'].values
    A2  = dat['A2'].values
    A3  = dat['A3'].values
    Y1  = dat['Y_1'].values
    Y2  = dat['Y_2'].values
    Y   = dat['Y'].values

    ATE_1 = {}
    base1 = Y1[A1 == 0].mean()
    for a in range(1, k1):
        ATE_1[a] = float(Y1[A1 == a].mean() - base1)

    ATE_2 = {}
    base2 = Y2[A2 == 0].mean()
    for a in range(1, k2):
        ATE_2[a] = float(Y2[A2 == a].mean() - base2)

    ATE_3 = {}
    base3 = Y[A3 == 0].mean()
    for a in range(1, k3):
        ATE_3[a] = float(Y[A3 == a].mean() - base3)

    if verbose:
        print(f"\n{'='*55}")
        print(f"Naive ATE (unadjusted): {filename}")
        print(f"{'='*55}")
        for stage, ates in [(1, ATE_1), (2, ATE_2), (3, ATE_3)]:
            print(f"\n  Stage {stage}  (A{stage} vs A{stage}=0)")
            for a, ate in ates.items():
                print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2, 'ATE_3': ATE_3}


# ============================================================
# Function 3: estimated_ate_simple
# ============================================================

def estimated_ate_simple(filename, verbose=True):
    """
    Estimate ATE using AIPW-style modified outcomes from estimate_dre_three_stage.

      ATE_t(a) = mean( mu_hat_t_a{a} - mu_hat_t_a0 )   for t in {1, 2, 3}

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2', 'ATE_3' -- each a dict {arm: float}
    """
    info   = pd.read_csv(info_path)
    row    = info[info['filename'] == filename].iloc[0]
    k1, k2 = int(row['k1']), int(row['k2'])
    k3     = int(row['k3'])

    dre = pd.read_csv(os.path.join(datasets_dir, f'{filename}_DRE.csv'))

    ATE_1 = {}
    for a in range(1, k1):
        ATE_1[a] = float(np.mean(dre[f'mu_hat_1_a{a}'] - dre['mu_hat_1_a0']))

    ATE_2 = {}
    for a in range(1, k2):
        ATE_2[a] = float(np.mean(dre[f'mu_hat_2_a{a}'] - dre['mu_hat_2_a0']))

    ATE_3 = {}
    for a in range(1, k3):
        ATE_3[a] = float(np.mean(dre[f'mu_hat_3_a{a}'] - dre['mu_hat_3_a0']))

    if verbose:
        print(f"\n{'='*55}")
        print(f"Estimated ATE (DRE, 3-stage simple): {filename}")
        print(f"{'='*55}")
        for stage, ates in [(1, ATE_1), (2, ATE_2), (3, ATE_3)]:
            print(f"\n  Stage {stage}  (A{stage} vs A{stage}=0)")
            for a, ate in ates.items():
                print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2, 'ATE_3': ATE_3}


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    info    = pd.read_csv(info_path)
    results = []

    for _, row in info.iterrows():
        fname = row['filename']
        try:
            t = true_ate_simple(fname,      verbose=False)
            n = naive_ate_simple(fname,     verbose=False)
            e = estimated_ate_simple(fname, verbose=False)

            for stage, arms_t, arms_n, arms_e in [
                (1, t['ATE_1'], n['ATE_1'], e['ATE_1']),
                (2, t['ATE_2'], n['ATE_2'], e['ATE_2']),
                (3, t['ATE_3'], n['ATE_3'], e['ATE_3']),
            ]:
                for a in arms_t:
                    ate_true  = arms_t[a]
                    ate_naive = arms_n.get(a, np.nan)
                    ate_est   = arms_e.get(a, np.nan)
                    results.append({
                        'filename':   fname,
                        'k':          row['k1'],
                        'flavor_Y':   row['flavor_Y'],
                        'stage':      stage,
                        'arm':        a,
                        'ATE_true':   ate_true,
                        'ATE_naive':  ate_naive,
                        'ATE_est':    ate_est,
                        'bias_naive': ate_naive - ate_true,
                        'bias_est':   ate_est   - ate_true,
                    })
        except FileNotFoundError as exc:
            print(f"Skipping {fname}: {exc}")

    summary  = pd.DataFrame(results)
    out_path = os.path.join(tables_dir, '_ate_simple_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ ATE summary saved to tables/_ate_simple_summary.csv")
    print(summary.groupby(['k', 'flavor_Y', 'stage', 'arm'])[
        ['ATE_true', 'ATE_naive', 'ATE_est', 'bias_naive', 'bias_est']
    ].mean().round(4))
