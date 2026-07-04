# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ate_single_stage_simple.py
# True and estimated ATEs for the single-stage simple DGP.
#
# Mirrors ate_two_stage_simple.py for the single-stage case.
# ATE definition:
#   ATE(a) = E[ Y(a) - Y(0) ]   (marginal over X1 in the observed sample)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/1stages/tables')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)


def _make_params(p1, k, seed):
    """Reproduce DGP parameters from gen_single_stage_data._make_params."""
    rng = np.random.default_rng(seed)
    return {
        'beta_A1': rng.uniform(-0.5, 0.5, size=(p1 + 1, k - 1)),
        'beta_Y1': rng.uniform(-1.0, 1.0, size=(p1 + 1,)),
        'delta1':  np.array([0.6, 0.4, 0.75, 0.17])[:k - 1],
        'Delta1':  np.array([-1.2, -1.0, -1.0,  0.8])[:k - 1],
    }


def _mean_outcome(eta, flavor):
    """
    E[Y | eta] for each flavor (ignoring mean-zero noise).

    expo     : Y = exp(eta) + N(0, 0.5)         -> E[Y] = exp(eta)
    sigmoid  : Y = 10/(1+exp(-eta)) + N(0, 0.5) -> E[Y] = 10/(1+exp(-eta))
    gamma    : Y = f_gamma(eta)*10 + N(0,0.5) + 0.1
    lognormal: Y = exp(eta + N(0, 0.5))          -> E[Y] = exp(eta + sigma^2/2)
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
        return np.exp(eta + 0.5**2 / 2.0)
    else:
        raise ValueError(f'Unknown flavor: {flavor}')


def true_ate_simple(filename, verbose=True):
    """
    Compute the true ATE using known DGP parameters for the single-stage DGP.

      eta(a) = X1_with_int @ beta_Y1 + delta1[a-1] + Delta1[a-1]*X1_bin
      ATE(a) = mean( E[Y(a)|X1] - E[Y(0)|X1] )

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')
    verbose  : bool

    Returns
    -------
    dict with key 'ATE_1' -- a dict {arm: float}
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    p1       = int(row['p1'])
    k        = int(row['k1'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])

    params  = _make_params(p1=p1, k=k, seed=seed)
    beta_Y1 = params['beta_Y1']   # (p1+1,)
    delta1  = params['delta1']    # (k-1,)
    Delta1  = params['Delta1']    # (k-1,)

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1      = dat[X1_cols].values          # (n, p1)
    n       = len(dat)

    X1_bin      = X1[:, -1]                          # last col is binary
    X1_with_int = np.column_stack([np.ones(n), X1])  # (n, p1+1)

    eta_base = X1_with_int @ beta_Y1   # (n,) under A1=0

    ATE_1 = {}
    for a in range(1, k):
        eta_a    = eta_base + delta1[a-1] + Delta1[a-1] * X1_bin
        ATE_1[a] = float(np.mean(
            _mean_outcome(eta_a,    flavor_Y) -
            _mean_outcome(eta_base, flavor_Y)
        ))

    if verbose:
        print(f"\n{'='*55}")
        print(f"True ATE (single-stage simple DGP): {filename}")
        print(f"{'='*55}")
        for a, ate in ATE_1.items():
            print(f"  ATE(a={a} vs 0) = {ate:.4f}")

    return {'ATE_1': ATE_1}


if __name__ == '__main__':
    os.makedirs(tables_dir, exist_ok=True)
    info    = pd.read_csv(info_path)
    results = []

    for _, row in info.iterrows():
        fname = row['filename']
        try:
            t = true_ate_simple(fname, verbose=False)
            for a, ate in t['ATE_1'].items():
                results.append({
                    'filename': fname,
                    'k':        row['k1'],
                    'flavor_Y': row['flavor_Y'],
                    'arm':      a,
                    'ATE_true': ate,
                })
        except FileNotFoundError as exc:
            print(f"Skipping {fname}: {exc}")

    summary  = pd.DataFrame(results)
    out_path = os.path.join(tables_dir, '_ate_simple_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ ATE summary saved: _ate_simple_summary.csv")
    print(summary.groupby(['k', 'flavor_Y', 'arm'])[['ATE_true']].mean().round(4))
