# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_drep_new_i0_three_stage.py
# Fit DRE-Param (sklearn OLS) models on the original i=0 dataset, save them, then
# apply to the new_i0 dataset to compute predicted Y under the learned policy.
#
# Output per (k, flavor):
#   models saved : datasets/new_i0/models/s3_k{k}_simple_{flavor}_0_DREp_models.pkl
#   predictions  : datasets/new_i0/s3_k{k}_simple_{flavor}_new_i0_DREp.csv
#     columns: d_star_1, d_star_2, d_star_3,
#              Y_hat_1_a0..., Y_hat_2_a0..., Y_hat_3_a0...
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.linear_model import LinearRegression

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
models_dir   = os.path.join(new_i0_dir,   'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


def estimate_drep_new_i0(k, flavor_Y):
    """Fit DRE-Param on old i=0 (3-stage); apply outcome models to new_i0."""
    fname_i0    = f"s3_k{k}_simple_{flavor_Y}_0"
    fname_new   = f"s3_k{k}_simple_{flavor_Y}_new_i0"
    models_path = os.path.join(models_dir, f'{fname_i0}_DREp_models.pkl')
    out_path    = os.path.join(new_i0_dir,  f'{fname_new}_DREp.csv')

    print(f"\n{'='*60}")
    print(f"DRE-Param new_i0 (3-stage): k={k}, flavor={flavor_Y}")
    print(f"  Fitting on : {fname_i0}.csv")
    print(f"  Applying to: {fname_new}.csv")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # Load i=0 data
    # ----------------------------------------------------------------
    dat_i0  = pd.read_csv(os.path.join(datasets_dir, f'{fname_i0}.csv'))
    k1 = k2 = k3 = k

    X1_cols = [c for c in dat_i0.columns if c.startswith('X1_')]
    X1_i0   = dat_i0[X1_cols].values
    A1_i0   = dat_i0['A1'].values
    Y1_i0   = dat_i0['Y_1'].values
    A2_i0   = dat_i0['A2'].values
    Y2_i0   = dat_i0['Y_2'].values
    A3_i0   = dat_i0['A3'].values
    Y_i0    = dat_i0['Y'].values
    n_i0    = len(dat_i0)

    # ----------------------------------------------------------------
    # Stage 1 — fit OLS on i=0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — OLS outcome models on i=0]")
    models_Y1     = {}
    Y1_hat_all_i0 = np.zeros((n_i0, k1))
    for a in range(k1):
        mask    = (A1_i0 == a)
        model_a = LinearRegression().fit(X1_i0[mask], Y1_i0[mask])
        models_Y1[a]        = model_a
        Y1_hat_all_i0[:, a] = model_a.predict(X1_i0)
        print(f"    OLS Y1 a={a}  (n={mask.sum()})")

    # ----------------------------------------------------------------
    # Stage 2 — fit OLS on i=0  (features: [X1, Y1_resid])
    # ----------------------------------------------------------------
    print("\n[Stage 2 — OLS outcome models on i=0]")
    models_Y2     = {}
    Y2_hat_all_i0 = np.zeros((n_i0, k2))
    for a in range(k2):
        resid_a  = (Y1_i0 - Y1_hat_all_i0[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1_i0, resid_a])
        mask     = (A2_i0 == a)
        model_a  = LinearRegression().fit(feat_2_a[mask], Y2_i0[mask])
        models_Y2[a]        = model_a
        Y2_hat_all_i0[:, a] = model_a.predict(feat_2_a)
        print(f"    OLS Y2 a={a}  (n={mask.sum()})")

    # ----------------------------------------------------------------
    # Stage 3 — fit OLS on i=0  (features: [X1, Y1_resid, Y2_resid])
    # ----------------------------------------------------------------
    print("\n[Stage 3 — OLS outcome models on i=0]")
    models_Y3     = {}
    Y3_hat_all_i0 = np.zeros((n_i0, k3))
    for a in range(k3):
        a1_idx   = min(a, k1 - 1)
        a2_idx   = min(a, k2 - 1)
        resid1_a = (Y1_i0 - Y1_hat_all_i0[:, a1_idx]).reshape(-1, 1)
        resid2_a = (Y2_i0 - Y2_hat_all_i0[:, a2_idx]).reshape(-1, 1)
        feat_3_a = np.hstack([X1_i0, resid1_a, resid2_a])
        mask     = (A3_i0 == a)
        model_a  = LinearRegression().fit(feat_3_a[mask], Y_i0[mask])
        models_Y3[a]        = model_a
        Y3_hat_all_i0[:, a] = model_a.predict(feat_3_a)
        print(f"    OLS Y3 a={a}  (n={mask.sum()})")

    # ----------------------------------------------------------------
    # Save models
    # ----------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    with open(models_path, 'wb') as f:
        pickle.dump({'models_Y1': models_Y1, 'models_Y2': models_Y2,
                     'models_Y3': models_Y3,
                     'k1': k1, 'k2': k2, 'k3': k3,
                     'X1_cols': X1_cols}, f)
    print(f"\n✓ Models saved: {os.path.basename(models_path)}")

    # ----------------------------------------------------------------
    # Load new_i0 data
    # ----------------------------------------------------------------
    dat_new = pd.read_csv(os.path.join(new_i0_dir, f'{fname_new}.csv'))
    X1_new  = dat_new[X1_cols].values
    Y1_new  = dat_new['Y_1'].values
    Y2_new  = dat_new['Y_2'].values
    n_new   = len(dat_new)

    # ----------------------------------------------------------------
    # Apply stage 1 models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 1 — applying to new_i0]")
    Y1_hat_all_new = np.zeros((n_new, k1))
    for a in range(k1):
        Y1_hat_all_new[:, a] = models_Y1[a].predict(X1_new)
    d_star_1 = np.argmax(Y1_hat_all_new, axis=1)
    print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ----------------------------------------------------------------
    # Apply stage 2 models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 2 — applying to new_i0]")
    Y2_hat_all_new = np.zeros((n_new, k2))
    for a in range(k2):
        resid_a  = (Y1_new - Y1_hat_all_new[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1_new, resid_a])
        Y2_hat_all_new[:, a] = models_Y2[a].predict(feat_2_a)
    d_star_2 = np.argmax(Y2_hat_all_new, axis=1)
    print(f"  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ----------------------------------------------------------------
    # Apply stage 3 models to new_i0
    # ----------------------------------------------------------------
    print("\n[Stage 3 — applying to new_i0]")
    Y3_hat_all_new = np.zeros((n_new, k3))
    for a in range(k3):
        a1_idx   = min(a, k1 - 1)
        a2_idx   = min(a, k2 - 1)
        resid1_a = (Y1_new - Y1_hat_all_new[:, a1_idx]).reshape(-1, 1)
        resid2_a = (Y2_new - Y2_hat_all_new[:, a2_idx]).reshape(-1, 1)
        feat_3_a = np.hstack([X1_new, resid1_a, resid2_a])
        Y3_hat_all_new[:, a] = models_Y3[a].predict(feat_3_a)
    d_star_3 = np.argmax(Y3_hat_all_new, axis=1)
    print(f"  d_star_3 distribution: {np.bincount(d_star_3)}")

    V = float(np.mean(Y3_hat_all_new[np.arange(n_new), d_star_3]))
    print(f"\n  V(DRE-Param) = {V:.4f}")

    # ----------------------------------------------------------------
    # Save output
    # ----------------------------------------------------------------
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2, 'd_star_3': d_star_3})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_all_new[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_all_new[:, a]
    for a in range(k3):
        out[f'Y_hat_3_a{a}'] = Y3_hat_all_new[:, a]
    out.to_csv(out_path, index=False)
    print(f"✓ Saved: {fname_new}_DREp.csv")
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
        estimate_drep_new_i0(k=int(row['k1']), flavor_Y=row['flavor_Y'])

    print('\nDone.')
