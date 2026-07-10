# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_train_dre_two_stage.py
# Load DRE-ML pkl, apply AIPW to training data, save _DRE.csv for boxplots_stage2.py
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_dre_two_stage.py and re-applies the stored models + Y1_hat_all to
# the training data to reconstruct the AIPW modified outcomes.
#
# Input:  datasets/models/{filename}_DRE_models.pkl
#         datasets/{filename}.csv  (training data)
# Output: datasets/{filename}_DRE.csv
#   columns: d_star_1, d_star_2, mu_hat_1_a0..., mu_hat_2_a0...
#
# Optional filters in __main__:
#   K_FILTER      : int or None
#   FLAVOR_FILTER : str or None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# AIPW helper (same formula as estimate_dre_two_stage.py)
# ============================================================

def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    Self-normalized AIPW (Hajek-style) modified outcome.

    mu_hat_a = Yhat_a + I(A=a) * (Y - Yhat_a) / pi_a / mean(I(A=a) / pi_a)
    """
    n      = len(A_obs)
    mu_hat = np.zeros((n, k))
    for a in range(k):
        I_a  = (A_obs == a).astype(float)
        pi_a = np.clip(pi_hat_all[:, a], 1e-6, 1 - 1e-6)
        w_a  = np.mean(I_a / pi_a)
        mu_hat[:, a] = Y_hat_all[:, a] + I_a * (Y_obs - Y_hat_all[:, a]) / pi_a / w_a
    return mu_hat


# ============================================================
# Per-dataset runner
# ============================================================

def run_train_dre(filename):
    """
    Load the DRE pkl for `filename`, apply to training data, save _DRE.csv.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    """
    pkl_path = os.path.join(models_dir, f'{filename}_DRE_models.pkl')
    csv_path = os.path.join(datasets_dir, f'{filename}.csv')
    out_path = os.path.join(datasets_dir, f'{filename}_DRE.csv')

    if not os.path.exists(pkl_path):
        print(f"  Skipping {filename}: pkl not found ({pkl_path})")
        return
    if not os.path.exists(csv_path):
        print(f"  Skipping {filename}: training CSV not found ({csv_path})")
        return

    print(f"\n{'='*60}")
    print(f"run_train_dre: {filename}")
    print(f"{'='*60}")

    # ---- Load pkl ----
    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    models_Y1  = pkg['models_Y1']   # dict: arm → fitted NN
    models_Y2  = pkg['models_Y2']   # dict: arm → fitted NN
    pscore_A1  = pkg['pscore_A1']   # fitted NN pscore model
    pscore_A2  = pkg['pscore_A2']   # fitted NN pscore model
    Y1_hat_all = pkg['Y1_hat_all']  # (n_train, k1) — pre-computed on training data
    X1_cols    = pkg['X1_cols']
    X2_cols    = pkg['X2_cols']
    k1         = pkg['k1']
    k2         = pkg['k2']

    # ---- Load training data ----
    dat = pd.read_csv(csv_path)
    n   = len(dat)

    # DataFrames with correct column names (DRE NNs were fitted on DataFrames)
    X1_df = dat[X1_cols].reset_index(drop=True)
    X2_df = dat[X2_cols].reset_index(drop=True)
    A1    = dat['A1'].values
    A2    = dat['A2'].values
    Y1    = dat['Y_1'].values
    Y     = dat['Y'].values

    print(f"  n={n}, k1={k1}, k2={k2}")

    # ---- Stage 1: use stored Y1_hat_all (already predicted on training set) ----
    # No need to re-predict; the pkl stores Y1_hat_all computed at fit time.
    print("\n  [Stage 1 — propensity score]")
    pi1_hat = pscore_A1.predict_proba(X1_df)   # (n, k1)

    mu_hat_1 = compute_mu_hat(A1, Y1, Y1_hat_all, pi1_hat, k1)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ---- Stage 2: reconstruct per-arm features and predict ----
    # The stage-2 outcome NN for arm a was fitted on features [X1, X2, Y1_resid_a],
    # where Y1_resid_a = Y1 - Y1_hat_all[:, a] evaluated at training time.
    # We reconstruct those same residuals from the stored Y1_hat_all.
    print("\n  [Stage 2 — outcome model predictions]")
    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        resid_a  = Y1 - Y1_hat_all[:, a]
        # Build DataFrame matching the column order used at fit time
        feat_2_a = pd.concat([X1_df,
                               X2_df,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        Y2_hat_all[:, a] = models_Y2[a].predict(feat_2_a)

    # ---- Stage 2: propensity ----
    # pscore_A2 was fitted on DataFrame with columns X1_cols + ['A1','Y_1'] + X2_cols
    print("\n  [Stage 2 — propensity score]")
    feat_ps2 = pd.concat([X1_df,
                           pd.Series(A1, name='A1'),
                           pd.Series(Y1, name='Y_1'),
                           X2_df], axis=1)
    pi2_hat = pscore_A2.predict_proba(feat_ps2)   # (n, k2)

    mu_hat_2 = compute_mu_hat(A2, Y, Y2_hat_all, pi2_hat, k2)
    d_star_2 = np.argmax(mu_hat_2, axis=1)
    print(f"  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ---- Save output CSV ----
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]
    for a in range(k2):
        out[f'mu_hat_2_a{a}'] = mu_hat_2[:, a]

    out.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {filename}_DRE.csv")


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        run_train_dre(row['filename'])

    print('\nDone.')
