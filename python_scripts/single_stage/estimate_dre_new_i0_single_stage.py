# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_new_i0_single_stage.py
# Fit DRE-ML outcome models on the original i=0 dataset, save them, then apply
# to the new_i0 dataset to compute predicted Y under the learned policy.
#
# Output per (k, flavor):
#   models saved : datasets/new_i0/models/s1_k{k}_simple_{flavor}_0_DRE_models.pkl
#   predictions  : datasets/new_i0/s1_k{k}_simple_{flavor}_new_i0_DRE.csv
#     columns: d_star_1, Y_hat_1_a0, ..., Y_hat_1_a{k-1}
#
# V(DRE-ML) = mean(Y_hat_1[i, d_star_1[i]])  — evaluated by vplot_new_i0_stage1.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import random
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
models_dir   = os.path.join(new_i0_dir,   'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from Y_nn_tuning import Y_model_nn

DEFAULT_HIDUNITS = [random.randint(10, 115) for _ in range(30)]
DEFAULT_EPS      = [random.randint(40, 150) for _ in range(30)]
DEFAULT_PENALS   = [0.001, 0.005, 0.01]


def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6)


def estimate_dre_new_i0(k, flavor_Y,
                        hidunits=DEFAULT_HIDUNITS,
                        eps=DEFAULT_EPS,
                        penals=DEFAULT_PENALS,
                        i_source=0):
    """Fit DRE-ML on old i=i_source (single-stage); apply outcome models to new_i0."""
    fname_i0    = f"s1_k{k}_simple_{flavor_Y}_{i_source}"
    fname_new   = f"s1_k{k}_simple_{flavor_Y}_new_i0"
    models_path = os.path.join(models_dir, f'{fname_i0}_DRE_models.pkl')
    out_path    = os.path.join(new_i0_dir,  f'{fname_new}_DRE.csv')

    print(f"\n{'='*60}")
    print(f"DRE-ML new_i0 (single-stage): k={k}, flavor={flavor_Y}")
    print(f"  Fitting on : {fname_i0}.csv")
    print(f"  Applying to: {fname_new}.csv")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # Load i=0 data
    # ----------------------------------------------------------------
    dat_i0  = pd.read_csv(os.path.join(datasets_dir, f'{fname_i0}.csv'))

    X1_cols = [c for c in dat_i0.columns if c.startswith('X1_')]
    X1_i0   = dat_i0[X1_cols].reset_index(drop=True)
    A1_i0   = dat_i0['A1'].values
    Y_i0    = dat_i0['Y'].values
    n_i0    = len(dat_i0)

    # ----------------------------------------------------------------
    # Stage 1 — fit outcome models on i=0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — outcome models on i=0]")
    models_Y1     = {}
    Y1_hat_all_i0 = np.zeros((n_i0, k))
    for a in range(k):
        mask    = (A1_i0 == a)
        model_a = _fit_outcome_nn(X1_i0[mask].reset_index(drop=True),
                                  Y_i0[mask], hidunits, eps, penals, tag=f'Y a={a} ')
        models_Y1[a]        = model_a
        Y1_hat_all_i0[:, a] = model_a.predict(X1_i0)

    # ----------------------------------------------------------------
    # Save models
    # ----------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    with open(models_path, 'wb') as f:
        pickle.dump({'models_Y1': models_Y1, 'k': k, 'X1_cols': X1_cols}, f)
    print(f"\n✓ Models saved: {os.path.basename(models_path)}")

    # ----------------------------------------------------------------
    # Load new_i0 data
    # ----------------------------------------------------------------
    dat_new = pd.read_csv(os.path.join(new_i0_dir, f'{fname_new}.csv'))
    X1_new  = dat_new[X1_cols].reset_index(drop=True)
    n_new   = len(dat_new)

    # ----------------------------------------------------------------
    # Apply models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — applying models to new_i0]")
    Y1_hat_all_new = np.zeros((n_new, k))
    for a in range(k):
        Y1_hat_all_new[:, a] = models_Y1[a].predict(X1_new)
    d_star_1 = np.argmax(Y1_hat_all_new, axis=1)
    print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")

    V = float(np.mean(Y1_hat_all_new[np.arange(n_new), d_star_1]))
    print(f"\n  V(DRE-ML) = {V:.4f}")

    # ----------------------------------------------------------------
    # Save output
    # ----------------------------------------------------------------
    out = pd.DataFrame({'d_star_1': d_star_1})
    for a in range(k):
        out[f'Y_hat_1_a{a}'] = Y1_hat_all_new[:, a]
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {fname_new}_DRE.csv")
    return out


if __name__ == '__main__':
    K_FILTER = None   # set to 2, 3, or 5 to run only that k; None = run all
    I_SOURCE = 0      # which dataset index to fit models on

    os.makedirs(new_i0_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == I_SOURCE].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        estimate_dre_new_i0(k=int(row['k1']), flavor_Y=row['flavor_Y'],
                            i_source=I_SOURCE)

    print('\nDone.')
