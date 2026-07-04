# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_naive_single_stage.py
# Naive baseline estimator for single-stage DTR.
#
# Decisions: d_star_1 drawn uniformly at random for each individual.
# mu_hat   : unadjusted observed arm means (constant per arm, broadcast to all n).
#
# Output format mirrors estimate_dre_single_stage.py.
# Saved as: {filename}_NAIVE.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


def estimate_naive(filename):
    """
    Naive single-stage estimator: random decisions, unadjusted mu_hat.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')

    Returns
    -------
    DataFrame with columns: d_star_1, mu_hat_1_a0, mu_hat_1_a1, ...
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k  = int(row['k1'])
    A1 = dat['A1'].values
    Y  = dat['Y'].values
    n  = len(dat)

    # Random decisions — uniform over arms
    d_star_1 = np.random.choice(k, size=n)

    # Naive mu_hat: observed arm mean broadcast to all n individuals
    mu_hat_1 = np.zeros((n, k))
    for a in range(k):
        mask = (A1 == a)
        mu_hat_1[:, a] = Y[mask].mean() if mask.sum() > 0 else 0.0

    out = pd.DataFrame({'d_star_1': d_star_1})
    for a in range(k):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_NAIVE.csv')
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {filename}_NAIVE.csv")
    return out


if __name__ == '__main__':
    K_FILTER = None   # set to 2, 3, or 5 to run only that k; None = run all
    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    for _, row in info.iterrows():
        estimate_naive(row['filename'])
