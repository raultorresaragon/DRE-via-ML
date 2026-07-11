# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_train_drep_single_stage.py
# Load DRE-Param pkl, apply AIPW to training data, save CSV for downstream analysis.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_drep_single_stage.py and re-applies the stored models + Y_hat_all to
# the training data to reconstruct the AIPW modified outcomes.
#
# pkl keys expected: models_Y, ps, Y_hat_all, X_cols, k, outcome_model
#
# Input:  datasets/models/{filename}_DREp_ols_models.pkl   (OLS)
#         datasets/models/{filename}_DREp_expo_models.pkl  (EXPO)
#         datasets/{filename}.csv  (training data)
# Output:
#   OLS  → datasets/{filename}_DREp_ols.csv
#   EXPO → datasets/{filename}_DREp_expo.csv
#   columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
#
# Optional filters in __main__:
#   K_FILTER      : int or None
#   FLAVOR_FILTER : str or None
#   DREP_MODEL    : 'OLS', 'EXPO', or 'BOTH'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle
import warnings
import statsmodels.api as sm

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# EXPO prediction helper (mirrors estimate_drep_single_stage.py)
# ============================================================

def _predict_expo(result, X):
    """Predict from a fitted GLM-expo (statsmodels) result; adds constant internally."""
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)


# ============================================================
# AIPW helper (same formula as estimate_drep_single_stage.py)
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


def _predict_proba_ordered(model, X, k):
    """Return (n, k) probability matrix with columns in arm order 0..k-1."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), k))
    for col_idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col_idx]
    return out


# ============================================================
# Per-dataset runner
# ============================================================

def run_train_drep(filename, outcome_model='OLS'):
    """
    Load the DREp pkl for `filename`, apply to training data, save CSV.

    Parameters
    ----------
    filename      : str   Base filename without extension (e.g. 's1_k2_expo_0')
    outcome_model : str   'OLS' or 'EXPO'
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('OLS', 'EXPO'):
        raise ValueError(f"outcome_model must be 'OLS' or 'EXPO', got '{outcome_model}'")

    pkl_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'
    csv_suffix = '_DREp_expo' if outcome_model == 'EXPO' else '_DREp_ols'

    pkl_path = os.path.join(models_dir,   f'{filename}_{pkl_suffix}_models.pkl')
    csv_path = os.path.join(datasets_dir, f'{filename}.csv')
    out_path = os.path.join(datasets_dir, f'{filename}{csv_suffix}.csv')

    if not os.path.exists(pkl_path):
        print(f"  Skipping {filename} ({outcome_model}): pkl not found ({pkl_path})")
        return
    if not os.path.exists(csv_path):
        print(f"  Skipping {filename}: training CSV not found ({csv_path})")
        return

    model_tag = 'GLM-expo' if outcome_model == 'EXPO' else 'OLS'
    print(f"\n{'='*60}")
    print(f"run_train_drep ({model_tag}): {filename}")
    print(f"{'='*60}")

    # ---- Load pkl ----
    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    models_Y  = pkg['models_Y']   # dict: arm → fitted parametric outcome model
    ps        = pkg['ps']         # LogisticRegression propensity model
    Y_hat_all = pkg['Y_hat_all']  # (n_train, k) — pre-computed on training data
    X_cols    = pkg['X_cols']
    k         = pkg['k']

    # ---- Load training data ----
    dat = pd.read_csv(csv_path)
    n   = len(dat)

    # DREp models were fitted on numpy arrays
    X = dat[X_cols].values
    A = dat['A'].values
    Y = dat['Y'].values

    print(f"  n={n}, k={k}, outcome_model={outcome_model}")

    # ---- Use stored Y_hat_all (already predicted on training set at fit time) ----
    # ps was fitted on numpy array X
    print("\n  [Propensity score]")
    pi_hat = _predict_proba_ordered(ps, X, k)

    mu_hat = compute_mu_hat(A, Y, Y_hat_all, pi_hat, k)
    d_star = np.argmax(mu_hat, axis=1)
    print(f"  d_star distribution: {np.bincount(d_star)}")

    # ---- Save output CSV ----
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'mu_hat_a{a}'] = mu_hat[:, a]

    out.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {filename}{csv_suffix}.csv")


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'gamma', etc.; None = all
    DREP_MODEL    = 'BOTH'  # 'OLS', 'EXPO', or 'BOTH'

    models_to_run = ['OLS', 'EXPO'] if DREP_MODEL == 'BOTH' else [DREP_MODEL.upper()]

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        for model in models_to_run:
            run_train_drep(row['filename'], outcome_model=model)

    print('\nDone.')
