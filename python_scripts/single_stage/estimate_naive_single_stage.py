# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_naive_single_stage.py
# Naive baseline estimator for single-stage DTR.
#
# Decisions: d_star drawn uniformly at random for each individual.
# mu_hat   : unadjusted observed arm means (constant per arm, broadcast to all n).
#
# Output format mirrors estimate_dre_single_stage.py so that downstream scripts
# (boxplots, ate) can consume either file unchanged.
# Saved as: {filename}_NAIVE.csv
# Columns:  d_star, mu_hat_a0, mu_hat_a1, ..., mu_hat_a{k-1}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


def estimate_naive(filename, dgp='single'):
    """
    Naive single-stage estimator: random decisions, unadjusted mu_hat.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_expo_0')
    dgp      : str   Reserved for compatibility; single-stage always reads _info_single.csv

    Returns
    -------
    DataFrame with columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
    """
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k   = int(row['k'])
    dat = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    n   = len(dat)
    A   = dat['A'].values
    Y   = dat['Y'].values

    # Random decisions — uniform over arms
    d_star = np.random.randint(0, k, size=n)

    # Naive mu_hat: observed arm mean broadcast to all n individuals (no confounder adjustment)
    mu_hat = np.zeros((n, k))
    for a in range(k):
        mask = (A == a)
        mu_hat[:, a] = Y[mask].mean() if mask.sum() > 0 else 0.0

    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'mu_hat_a{a}'] = mu_hat[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_NAIVE.csv')
    out.to_csv(out_path, index=False)
    print(f'✓ Saved: {filename}_NAIVE.csv')
    return out


if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5 to run only that k; None = run all
    FLAVOR_FILTER = None   # set to 'expo', 'gamma', etc.; None = all

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        estimate_naive(row['filename'])
