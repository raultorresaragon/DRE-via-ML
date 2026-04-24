# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_naive_three_stage.py
# Naive baseline estimator for three-stage DTR.
#
# Decisions: d_star_1, d_star_2, d_star_3 drawn uniformly at random for each individual.
# mu_hat   : unadjusted observed arm means (constant per arm, broadcast to all n).
#
# Output format mirrors estimate_dre_three_stage.py so that OTR_assess.py and
# ate_three_stage_simple.py can consume either file unchanged.
# Saved as: {filename}_NAIVE.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
info_path_simple = os.path.join(datasets_dir, '_info_simple.csv')


def estimate_naive(filename, dgp='simple'):
    """
    Naive three-stage estimator: random decisions, unadjusted mu_hat.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')
    dgp      : str   'simple' reads _info_simple.csv; 'standard' reads _info.csv

    Returns
    -------
    DataFrame with columns: d_star_1..3, mu_hat_1_a0..., mu_hat_2_a0..., mu_hat_3_a0...
    """
    info_fname = '_info_simple.csv' if dgp == 'simple' else '_info.csv'
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(os.path.join(datasets_dir, info_fname))
    row  = info[info['filename'] == filename].iloc[0]

    k1, k2 = int(row['k1']), int(row['k2'])
    k3     = int(row['k3'])
    A1 = dat['A1'].values
    A2 = dat['A2'].values
    A3 = dat['A3'].values
    Y1 = dat['Y_1'].values
    Y2 = dat['Y_2'].values
    Y  = dat['Y'].values
    n  = len(dat)

    # Random decisions — uniform over arms
    d_star_1 = np.random.choice(k1, size=n)
    d_star_2 = np.random.choice(k2, size=n)
    d_star_3 = np.random.choice(k3, size=n)

    # Naive mu_hat: observed arm mean broadcast to all n individuals (no confounder adjustment)
    mu_hat_1 = np.zeros((n, k1))
    for a in range(k1):
        mask = (A1 == a)
        mu_hat_1[:, a] = Y1[mask].mean() if mask.sum() > 0 else 0.0

    mu_hat_2 = np.zeros((n, k2))
    for a in range(k2):
        mask = (A2 == a)
        mu_hat_2[:, a] = Y2[mask].mean() if mask.sum() > 0 else 0.0

    mu_hat_3 = np.zeros((n, k3))
    for a in range(k3):
        mask = (A3 == a)
        mu_hat_3[:, a] = Y[mask].mean() if mask.sum() > 0 else 0.0

    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2, 'd_star_3': d_star_3})
    for a in range(k1):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]
    for a in range(k2):
        out[f'mu_hat_2_a{a}'] = mu_hat_2[:, a]
    for a in range(k3):
        out[f'mu_hat_3_a{a}'] = mu_hat_3[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_NAIVE.csv')
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {filename}_NAIVE.csv")

    return out


if __name__ == '__main__':
    info = pd.read_csv(info_path_simple)
    for _, row in info.iterrows():
        estimate_naive(row['filename'], dgp='simple')
