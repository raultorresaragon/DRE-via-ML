# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_naive_new_i0_two_stage.py
# Naive estimator fitted directly on the new_i0 dataset (two-stage).
#
# Decisions : d_star_1, d_star_2 drawn uniformly at random.
# Y_hat     : observed arm mean for Y1 (stage 1) and Y (stage 2), broadcast to all n.
#
# Output: datasets/new_i0/s2_k{k}_simple_{flavor}_new_i0_NAIVE.csv
#   columns: d_star_1, d_star_2,
#            Y_hat_1_a0..., Y_hat_1_a{k1-1},
#            Y_hat_2_a0..., Y_hat_2_a{k2-1}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


def estimate_naive_new_i0(k, flavor_Y):
    """Naive estimator on new_i0 dataset."""
    fname_new = f"s2_k{k}_simple_{flavor_Y}_new_i0"
    out_path  = os.path.join(new_i0_dir, f'{fname_new}_NAIVE.csv')

    print(f"\n{'='*60}")
    print(f"Naive new_i0 (2-stage): k={k}, flavor={flavor_Y}")
    print(f"{'='*60}")

    dat = pd.read_csv(os.path.join(new_i0_dir, f'{fname_new}.csv'))
    k1 = k2 = k
    A1 = dat['A1'].values
    A2 = dat['A2'].values
    Y1 = dat['Y_1'].values
    Y  = dat['Y'].values
    n  = len(dat)

    d_star_1 = np.random.choice(k1, size=n)
    d_star_2 = np.random.choice(k2, size=n)

    # Y_hat = observed arm mean broadcast to all n
    Y1_hat_all = np.zeros((n, k1))
    for a in range(k1):
        mask = (A1 == a)
        Y1_hat_all[:, a] = Y1[mask].mean() if mask.sum() > 0 else 0.0

    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        mask = (A2 == a)
        Y2_hat_all[:, a] = Y[mask].mean() if mask.sum() > 0 else 0.0

    V = float(np.mean(Y2_hat_all[np.arange(n), d_star_2]))
    print(f"  V(Naive) = {V:.4f}")

    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_all[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_all[:, a]
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {fname_new}_NAIVE.csv")
    return out


if __name__ == '__main__':
    K_FILTER = None   # set to 2, 3, or 5 to run only that k; None = run all

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == 0].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        estimate_naive_new_i0(k=int(row['k1']), flavor_Y=row['flavor_Y'])

    print('\nDone.')
