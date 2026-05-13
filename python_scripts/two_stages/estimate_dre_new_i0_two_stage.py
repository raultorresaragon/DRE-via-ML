# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_new_i0_two_stage.py
# Fit DRE-ML outcome models on the original i=0 dataset, save them, then apply
# to the new_i0 dataset to compute predicted Y under the learned policy.
#
# Output per (k, flavor):
#   models saved : datasets/new_i0/models/s2_k{k}_simple_{flavor}_0_DRE_models.pkl
#   predictions  : datasets/new_i0/s2_k{k}_simple_{flavor}_new_i0_DRE.csv
#     columns: d_star_1, d_star_2,
#              Y_hat_1_a0..., Y_hat_1_a{k1-1},
#              Y_hat_2_a0..., Y_hat_2_a{k2-1}
#
# V(DRE-ML) = mean(Y_hat_2[i, d_star_2[i]])  — evaluated by vplot_new_i0_stage2.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
models_dir   = os.path.join(new_i0_dir,   'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn

DEFAULT_HIDUNITS = [5, 20]
DEFAULT_EPS      = [100, 250]
DEFAULT_PENALS   = [0.001, 0.01]


def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6,
                      early_stopping=not small)


def estimate_dre_new_i0(k, flavor_Y,
                        hidunits=DEFAULT_HIDUNITS,
                        eps=DEFAULT_EPS,
                        penals=DEFAULT_PENALS):
    """
    Fit DRE-ML on old i=0 dataset; apply outcome models to new_i0 dataset.

    Parameters
    ----------
    k        : int   Number of treatment levels
    flavor_Y : str   Y distribution flavor
    """
    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    fname_i0     = f"s2_k{k}_simple_{flavor_Y}_0"
    fname_new    = f"s2_k{k}_simple_{flavor_Y}_new_i0"
    models_path  = os.path.join(models_dir, f'{fname_i0}_DRE_models.pkl')
    out_path     = os.path.join(new_i0_dir,  f'{fname_new}_DRE.csv')

    print(f"\n{'='*60}")
    print(f"DRE-ML new_i0 (2-stage): k={k}, flavor={flavor_Y}")
    print(f"  Fitting on : {fname_i0}.csv")
    print(f"  Applying to: {fname_new}.csv")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # Load i=0 data
    # ----------------------------------------------------------------
    dat_i0 = pd.read_csv(os.path.join(datasets_dir, f'{fname_i0}.csv'))
    k1 = k2 = k

    X1_cols = [c for c in dat_i0.columns if c.startswith('X1_')]
    X2_cols = [c for c in dat_i0.columns if c.startswith('X2_')]

    X1_i0 = dat_i0[X1_cols].reset_index(drop=True)
    X2_i0 = dat_i0[X2_cols].reset_index(drop=True)
    A1_i0 = dat_i0['A1'].values
    A2_i0 = dat_i0['A2'].values
    Y1_i0 = dat_i0['Y_1'].values
    Y_i0  = dat_i0['Y'].values
    n_i0  = len(dat_i0)

    # ----------------------------------------------------------------
    # Stage 1 — fit outcome models on i=0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — outcome models on i=0]")
    models_Y1     = {}
    Y1_hat_all_i0 = np.zeros((n_i0, k1))
    for a in range(k1):
        mask      = (A1_i0 == a)
        model_a   = _fit_outcome_nn(X1_i0[mask].reset_index(drop=True),
                                    Y1_i0[mask], hidunits, eps, penals,
                                    tag=f'Y1 a={a} ')
        models_Y1[a]         = model_a
        Y1_hat_all_i0[:, a]  = model_a.predict(X1_i0)

    # ----------------------------------------------------------------
    # Stage 2 — fit outcome models on i=0
    # ----------------------------------------------------------------
    print("\n[Stage 2 — outcome models on i=0]")
    models_Y2     = {}
    Y2_hat_all_i0 = np.zeros((n_i0, k2))
    for a in range(k2):
        resid_a   = Y1_i0 - Y1_hat_all_i0[:, a]
        feat_2_a  = pd.concat([X1_i0, X2_i0,
                                pd.Series(resid_a, name='Y1_resid')], axis=1)
        mask      = (A2_i0 == a)
        model_a   = _fit_outcome_nn(feat_2_a[mask].reset_index(drop=True),
                                    Y_i0[mask], hidunits, eps, penals,
                                    tag=f'Y2 a={a} ')
        models_Y2[a]         = model_a
        Y2_hat_all_i0[:, a]  = model_a.predict(feat_2_a)

    # ----------------------------------------------------------------
    # Save models
    # ----------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    with open(models_path, 'wb') as f:
        pickle.dump({'models_Y1': models_Y1, 'models_Y2': models_Y2,
                     'k1': k1, 'k2': k2, 'X1_cols': X1_cols, 'X2_cols': X2_cols}, f)
    print(f"\n✓ Models saved: {os.path.basename(models_path)}")

    # ----------------------------------------------------------------
    # Load new_i0 data
    # ----------------------------------------------------------------
    dat_new = pd.read_csv(os.path.join(new_i0_dir, f'{fname_new}.csv'))
    X1_new  = dat_new[X1_cols].reset_index(drop=True)
    X2_new  = dat_new[X2_cols].reset_index(drop=True)
    Y1_new  = dat_new['Y_1'].values
    n_new   = len(dat_new)

    # ----------------------------------------------------------------
    # Apply stage 1 models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — applying models to new_i0]")
    Y1_hat_all_new = np.zeros((n_new, k1))
    for a in range(k1):
        Y1_hat_all_new[:, a] = models_Y1[a].predict(X1_new)
    d_star_1 = np.argmax(Y1_hat_all_new, axis=1)
    print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ----------------------------------------------------------------
    # Apply stage 2 models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 2 — applying models to new_i0]")
    Y2_hat_all_new = np.zeros((n_new, k2))
    for a in range(k2):
        resid_a  = Y1_new - Y1_hat_all_new[:, a]
        feat_2_a = pd.concat([X1_new, X2_new,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        Y2_hat_all_new[:, a] = models_Y2[a].predict(feat_2_a)
    d_star_2 = np.argmax(Y2_hat_all_new, axis=1)
    print(f"  d_star_2 distribution: {np.bincount(d_star_2)}")

    V = float(np.mean(Y2_hat_all_new[np.arange(n_new), d_star_2]))
    print(f"\n  V(DRE-ML) = {V:.4f}")

    # ----------------------------------------------------------------
    # Save output
    # ----------------------------------------------------------------
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_all_new[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_all_new[:, a]
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {fname_new}_DRE.csv")
    return out


if __name__ == '__main__':
    K_FILTER = None   # set to 2, 3, or 5 to run only that k; None = run all

    os.makedirs(new_i0_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == 0].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        estimate_dre_new_i0(k=int(row['k1']), flavor_Y=row['flavor_Y'])

    print('\nDone.')
