# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_train_drep_two_stage.py
# Load DRE-Param pkl, apply AIPW to training data, save CSV for boxplots_stage2.py
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_drep_two_stage.py and re-applies the stored models + Y1_hat_all to
# the training data to reconstruct the AIPW modified outcomes.
#
# Input:  datasets/models/{filename}_DREp_ols_models.pkl   (OLS)
#         datasets/models/{filename}_DREp_expo_models.pkl  (EXPO)
#         datasets/{filename}.csv  (training data)
# Output:
#   OLS  → datasets/{filename}_DREp.csv       (backward compat with boxplots_stage2.py)
#   EXPO → datasets/{filename}_DREp_expo.csv
#
# Optional filters in __main__:
#   K_FILTER      : int or None
#   FLAVOR_FILTER : str or None
#   DREP_MODEL    : 'OLS', 'EXPO', or 'BOTH'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import pickle
import warnings
import statsmodels.api as sm

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# EXPO prediction helper (mirrors estimate_drep_two_stage.py)
# ============================================================

def _predict_expo(result, X):
    """Predict from a fitted GLM-expo (statsmodels) result; adds constant internally."""
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)


# ============================================================
# AIPW helper (same formula as estimate_drep_two_stage.py)
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
    filename      : str   Base filename without extension
    outcome_model : str   'OLS' or 'EXPO'
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('OLS', 'EXPO'):
        raise ValueError(f"outcome_model must be 'OLS' or 'EXPO', got '{outcome_model}'")

    pkl_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'
    csv_suffix = '_DREp_expo' if outcome_model == 'EXPO' else '_DREp'

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

    models_Y1  = pkg['models_Y1']   # dict: arm → fitted model
    models_Y2  = pkg['models_Y2']   # dict: arm → fitted model
    ps1        = pkg['ps1']         # LogisticRegression for stage 1
    ps2        = pkg['ps2']         # LogisticRegression for stage 2
    Y1_hat_all = pkg['Y1_hat_all']  # (n_train, k1) — pre-computed on training data
    X1_cols    = pkg['X1_cols']
    X2_cols    = pkg['X2_cols']
    k1         = pkg['k1']
    k2         = pkg['k2']

    # ---- Load training data ----
    dat = pd.read_csv(csv_path)
    n   = len(dat)

    # DREp models were fitted on numpy arrays
    X1 = dat[X1_cols].values
    X2 = dat[X2_cols].values if X2_cols else np.empty((n, 0))
    A1 = dat['A1'].values
    A2 = dat['A2'].values
    Y1 = dat['Y_1'].values
    Y  = dat['Y'].values

    print(f"  n={n}, k1={k1}, k2={k2}, outcome_model={outcome_model}")

    # ---- Stage 1: use stored Y1_hat_all (already predicted on training set) ----
    # ps1 was fitted on numpy array X1
    print("\n  [Stage 1 — propensity score]")
    pi1_hat = _predict_proba_ordered(ps1, X1, k1)

    mu_hat_1 = compute_mu_hat(A1, Y1, Y1_hat_all, pi1_hat, k1)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ---- Stage 2: reconstruct per-arm features and predict ----
    # Stage-2 models were fitted on numpy arrays [X1, X2, Y1_resid_a].
    # Reconstruct residuals from the stored Y1_hat_all.
    print("\n  [Stage 2 — outcome model predictions]")
    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        resid_a  = (Y1 - Y1_hat_all[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1, X2, resid_a])
        if outcome_model == 'EXPO':
            Y2_hat_all[:, a] = _predict_expo(models_Y2[a], feat_2_a)
        else:
            Y2_hat_all[:, a] = models_Y2[a].predict(feat_2_a)

    # ---- Stage 2: propensity ----
    # ps2 was fitted on numpy array [X1, A1, Y1, X2]
    print("\n  [Stage 2 — propensity score]")
    feat_ps2 = np.hstack([X1, A1.reshape(-1, 1), Y1.reshape(-1, 1), X2])
    pi2_hat  = _predict_proba_ordered(ps2, feat_ps2, k2)

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
    print(f"\n  ✓ Saved: {filename}{csv_suffix}.csv")


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all
    DREP_MODEL    = 'BOTH'  # 'OLS', 'EXPO', or 'BOTH'

    models_to_run = ['OLS', 'EXPO'] if DREP_MODEL == 'BOTH' else [DREP_MODEL.upper()]

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        for model in models_to_run:
            run_train_drep(row['filename'], outcome_model=model)

    print('\nDone.')
