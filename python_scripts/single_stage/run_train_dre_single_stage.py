# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_train_dre_single_stage.py
# Load DRE-ML pkl, apply AIPW to training data, save _DRE.csv for downstream analysis.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_dre_single_stage.py and re-applies the stored models + Y_hat_all to
# the training data to reconstruct the AIPW modified outcomes.
#
# pkl keys expected: models_Y, pscore_A, Y_hat_all, X_cols, k
#
# Input:  datasets/models/{filename}_DRE_models.pkl
#         datasets/{filename}.csv  (training data)
# Output: datasets/{filename}_DRE.csv
#   columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
#
# Optional filters in __main__:
#   K_FILTER      : int or None
#   FLAVOR_FILTER : str or None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# AIPW helper (same formula as estimate_dre_single_stage.py)
# ============================================================

def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    Self-normalized AIPW (Hajek-style) modified outcome.

    mu_hat_a = Y_hat_a + I(A=a) * (Y - Y_hat_a) / pi_a / mean(I(A=a) / pi_a)
    """
    n      = len(A_obs)
    mu_hat = np.zeros((n, k))
    for a in range(k):
        I_a  = (A_obs == a).astype(float)
        pi_a = np.clip(pi_hat_all[:, a], 1e-6, 1 - 1e-6)
        w_a  = np.mean(I_a / pi_a)
        mu_hat[:, a] = Y_hat_all[:, a] + I_a * (Y_obs - Y_hat_all[:, a]) / pi_a / w_a
    return mu_hat


def _predict_proba_ordered(model, X_df, k):
    """Return (n, k) probability matrix with columns in arm order 0..k-1."""
    raw = model.predict_proba(X_df)
    out = np.zeros((len(X_df), k))
    for col_idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col_idx]
    return out


# ============================================================
# Per-dataset runner
# ============================================================

def run_train_dre(filename):
    """
    Load the DRE pkl for `filename`, apply to training data, save _DRE.csv.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_expo_0')
    """
    pkl_path = os.path.join(models_dir,   f'{filename}_DRE_models.pkl')
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

    models_Y  = pkg['models_Y']   # dict: arm → fitted NN outcome model
    pscore_A  = pkg['pscore_A']   # fitted NN propensity model
    Y_hat_all = pkg['Y_hat_all']  # (n_train, k) — pre-computed on training data
    X_cols    = pkg['X_cols']
    k         = pkg['k']

    # ---- Load training data ----
    dat = pd.read_csv(csv_path)
    n   = len(dat)

    # DRE NNs were fitted on DataFrames — use DataFrame for predict
    X_df = dat[X_cols].reset_index(drop=True)
    A    = dat['A'].values
    Y    = dat['Y'].values

    print(f"  n={n}, k={k}")

    # ---- Use stored Y_hat_all (already predicted on training set at fit time) ----
    # No need to re-predict outcomes; the pkl stores Y_hat_all computed at fit time.
    print("\n  [Propensity score]")
    pi_hat = _predict_proba_ordered(pscore_A, X_df, k)   # (n, k)

    mu_hat = compute_mu_hat(A, Y, Y_hat_all, pi_hat, k)
    d_star = np.argmax(mu_hat, axis=1)
    print(f"  d_star distribution: {np.bincount(d_star)}")

    # ---- Save output CSV ----
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'mu_hat_a{a}'] = mu_hat[:, a]

    out.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {filename}_DRE.csv")


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'gamma', etc.; None = all

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        run_train_dre(row['filename'])

    print('\nDone.')
