# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ate_two_stage_simple.py
# True and estimated ATEs for the simplified two-stage DGP.
#
# Reads metadata from _info_simple.csv.
# Saves summary to tables/_ate_simple_summary.csv.
#
# Differences from ate_two_stage.py
# ----------------------------------
# Stage 1  : unchanged — same DGP as original.
# Stage 2  : simplified DGP; covariates are time-invariant (X2 = X1) and:
#              eta(a) = X1_with_int @ beta_Y1
#                         + delta2[0] * a
#                         + Delta2[0] * a * X1_bin
#            Effect modifier is X1_bin (last col of X1), NOT Y1_bin.
#            A2 enters as a scalar multiplier (dose interpretation for k>2).
#
# ATE definitions
# ---------------
#   ATE_1(a) = E[ Y1(a) - Y1(0) ]
#   ATE_2(a) = E[ Y(a)  - Y(0)  ]   (marginal over observed X1, A1, Y1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/2stages/tables')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helper: analytic mean outcome for each flavor
# ============================================================

def _mean_outcome(eta, flavor):
    """
    E[Y | eta] for each flavor, ignoring mean-zero noise terms.

    expo     : Y = exp(eta) + N(0, 0.5)         -> E[Y] = exp(eta)
    sigmoid  : Y = 10/(1+exp(-eta)) + N(0, 0.5) -> E[Y] = 10/(1+exp(-eta))
    gamma    : Y = f_gamma(eta)*10 + N(0,0.5) + 0.1
    lognormal: Y = exp(eta + N(0, 0.5))          -> E[Y] = exp(eta + 0.125)
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
# Function 1: true_ate_simple
# ============================================================

def true_ate_simple(filename, verbose=True):
    """
    Compute the true ATE using known DGP parameters for the simple two-stage DGP.

    Stage 1 (unchanged from original DGP):
      ATE_1(a) = E[ Y1(a) - Y1(0) ]

    Stage 2 (simplified DGP — X2 = X1, no Y1 or A1 in outcome model):
      eta(a) = X1_with_int @ beta_Y1 + delta2[0]*a + Delta2[0]*a*X1_bin
      ATE_2(a) = mean( E[Y(a)|X1] - E[Y(0)|X1] )

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    verbose  : bool

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2' -- each a dict {arm: float}
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
    delta2  = params['delta2']     # (k2-1,) — only delta2[0] used (scalar multiplier)
    Delta2  = params['Delta2']     # (k2-1,) — only Delta2[0] used

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1      = dat[X1_cols].values          # (n, p1)
    n       = len(dat)

    X1_bin      = X1[:, -1]                          # binary modifier (last col of X1)
    X1_with_int = np.column_stack([np.ones(n), X1])  # (n, p1+1)

    # ------------------------------------------------------------------
    # Stage 1 ATE  (identical to ate_two_stage.py)
    # ------------------------------------------------------------------
    eta_Y1_base = X1_with_int @ beta_Y1   # (n,) under A1=0

    ATE_1 = {}
    for a in range(1, k1):
        eta_Y1_a = eta_Y1_base + delta1[a-1] + Delta1[a-1] * X1_bin
        ATE_1[a] = float(np.mean(
            _mean_outcome(eta_Y1_a,    flavor_Y) -
            _mean_outcome(eta_Y1_base, flavor_Y)
        ))

    # ------------------------------------------------------------------
    # Stage 2 ATE  (simple DGP)
    # eta(a) = X1_with_int @ beta_Y1 + delta2[0]*a + Delta2[0]*a*X1_bin
    # Effect modifier is X1_bin (NOT Y1_bin as in the original DGP).
    # A2 enters as a scalar multiplier (dose for k>2).
    # ------------------------------------------------------------------
    eta_Y2_base = X1_with_int @ beta_Y1   # (n,) under A2=0

    ATE_2 = {}
    for a in range(1, k2):
        eta_Y2_a = eta_Y2_base + delta2[0] * a + Delta2[0] * a * X1_bin
        ATE_2[a] = float(np.mean(
            _mean_outcome(eta_Y2_a,    flavor_Y) -
            _mean_outcome(eta_Y2_base, flavor_Y)
        ))

    if verbose:
        print(f"\n{'='*55}")
        print(f"True ATE (simple DGP): {filename}")
        print(f"{'='*55}")
        print(f"\n  Stage 1  (A1 vs A1=0)")
        for a, ate in ATE_1.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")
        print(f"\n  Stage 2  (A2 vs A2=0)")
        for a, ate in ATE_2.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


# ============================================================
# Function 2: estimated_ate_simple
# ============================================================

def estimated_ate_simple(filename, verbose=True):
    """
    Estimate ATE using AIPW-style modified outcomes from estimate_dre_two_stage.

    Formula is identical to the original estimated_ate — the DRE output format
    is the same regardless of which DGP generated the data:
      ATE_1(a) = mean( mu_hat_1_a - mu_hat_1_a0 )
      ATE_2(a) = mean( mu_hat_2_a - mu_hat_2_a0 )

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    verbose  : bool

    Returns
    -------
    dict with keys 'ATE_1', 'ATE_2' -- each a dict {arm: float}
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
        print(f"Estimated ATE (DRE, simple DGP): {filename}")
        print(f"{'='*55}")
        print(f"\n  Stage 1  (A1 vs A1=0)")
        for a, ate in ATE_1.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")
        print(f"\n  Stage 2  (A2 vs A2=0)")
        for a, ate in ATE_2.items():
            print(f"    ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


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
            e = estimated_ate_simple(fname, verbose=False)

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
    out_path = os.path.join(tables_dir, '_ate_simple_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ ATE summary saved to tables/_ate_simple_summary.csv")
    print(summary.groupby(['k', 'flavor_Y', 'stage', 'arm'])[
        ['ATE_true', 'ATE_est']
    ].mean().round(4))
