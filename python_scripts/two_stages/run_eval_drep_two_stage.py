# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_drep_two_stage.py
# Load DRE-Param pkl, apply outcome models to eval data, save per-replication predictions.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_drep_two_stage.py and applies the stored outcome models to the
# evaluation dataset to compute d_star and V(DRE-Param).
#
# No AIPW is applied — only outcome model predictions are used for decision-making.
#
# Two outcome model options, controlled by DREP_MODEL in __main__:
#   'EXPO' : GLM with Gaussian family + log link  → eval_per_i/{filename}_eval_DREp_expo.csv
#   'OLS'  : sklearn LinearRegression             → eval_per_i/{filename}_eval_DREp_ols.csv
#   'BOTH' : run EXPO then OLS, saving both files
#
# Input:  datasets/models/{filename}_DREp_ols_models.pkl   (OLS)
#         datasets/models/{filename}_DREp_expo_models.pkl  (EXPO)
#         eval_per_i/{filename}_eval.csv
# Output: eval_per_i/{filename}_eval_DREp_expo.csv  (EXPO)
#         eval_per_i/{filename}_eval_DREp_ols.csv   (OLS)
#   columns: d_star_1, d_star_2,
#            Y_hat_1_a0..Y_hat_1_a{k1-1},
#            Y_hat_2_a0..Y_hat_2_a{k2-1}
#
# Optional filters in __main__:
#   K_FILTER      : int or None  — only run for k = K_FILTER
#   FLAVOR_FILTER : str or None  — only run for a specific flavor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import warnings
import numpy as np
import pandas as pd
import os
import pickle
import statsmodels.api as sm

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
eval_dir     = os.path.join(datasets_dir, 'eval_per_i')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# EXPO prediction helper (mirrors estimate_drep_two_stage.py)
# ============================================================

def _predict_expo(result, X):
    """Predict from a fitted GLM-expo (statsmodels) result; adds constant internally."""
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)


# ============================================================
# Per-dataset runner
# ============================================================

def run_eval_drep(filename, outcome_model='EXPO'):
    """
    Load the DREp pkl for `filename`, apply to eval data, save _eval_DREp_{suffix}.csv.

    Parameters
    ----------
    filename      : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    outcome_model : str   'OLS' or 'EXPO'
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('OLS', 'EXPO'):
        raise ValueError(f"outcome_model must be 'OLS' or 'EXPO', got '{outcome_model}'")

    pkl_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'
    out_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'

    pkl_path  = os.path.join(models_dir, f'{filename}_{pkl_suffix}_models.pkl')
    eval_path = os.path.join(eval_dir,   f'{filename}_eval.csv')
    out_path  = os.path.join(eval_dir,   f'{filename}_eval_{out_suffix}.csv')

    if not os.path.exists(pkl_path):
        print(f"  Skipping {filename} ({outcome_model}): pkl not found ({pkl_path})")
        return None
    if not os.path.exists(eval_path):
        print(f"  Skipping {filename}: eval CSV not found ({eval_path})")
        return None

    model_tag = 'GLM-expo' if outcome_model == 'EXPO' else 'OLS'
    print(f"\n{'='*60}")
    print(f"run_eval_drep ({model_tag}): {filename}")
    print(f"{'='*60}")

    # ---- Load pkl ----
    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    models_Y1 = pkg['models_Y1']   # dict: arm → fitted parametric outcome model
    models_Y2 = pkg['models_Y2']   # dict: arm → fitted parametric outcome model
    X1_cols   = pkg['X1_cols']
    X2_cols   = pkg['X2_cols']
    k1        = pkg['k1']
    k2        = pkg['k2']

    # ---- Load eval data ----
    dat_ev = pd.read_csv(eval_path)
    n_ev   = len(dat_ev)

    # DREp models were fitted on numpy arrays
    X1_ev = dat_ev[X1_cols].values
    X2_ev = dat_ev[X2_cols].values if X2_cols else np.empty((n_ev, 0))
    Y1_ev = dat_ev['Y_1'].values

    print(f"  n_ev={n_ev}, k1={k1}, k2={k2}, outcome_model={outcome_model}")

    # ---- Stage 1: apply outcome models to eval X1 ----
    print("\n  [Stage 1 — applying outcome models to eval data]")
    Y1_hat_ev = np.zeros((n_ev, k1))
    for a in range(k1):
        if outcome_model == 'EXPO':
            Y1_hat_ev[:, a] = _predict_expo(models_Y1[a], X1_ev)
        else:
            Y1_hat_ev[:, a] = models_Y1[a].predict(X1_ev)
    d_star_1 = np.argmax(Y1_hat_ev, axis=1)
    print(f"  d_star_1: {np.bincount(d_star_1)}")

    # ---- Stage 2: apply outcome models to eval data ----
    # Residuals are computed from eval stage-1 predictions (not training Y1_hat_all).
    # feat_2_a: numpy array [X1_ev, X2_ev, Y1_ev - Y1_hat_ev[:, a]]
    print("\n  [Stage 2 — applying outcome models to eval data]")
    Y2_hat_ev = np.zeros((n_ev, k2))
    for a in range(k2):
        resid_a  = (Y1_ev - Y1_hat_ev[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1_ev, X2_ev, resid_a])
        if outcome_model == 'EXPO':
            Y2_hat_ev[:, a] = _predict_expo(models_Y2[a], feat_2_a)
        else:
            Y2_hat_ev[:, a] = models_Y2[a].predict(feat_2_a)
    d_star_2 = np.argmax(Y2_hat_ev, axis=1)
    print(f"  d_star_2: {np.bincount(d_star_2)}")

    V = float(np.mean(Y2_hat_ev[np.arange(n_ev), d_star_2]))
    print(f"\n  V(DRE-Param {model_tag}) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_ev[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_ev[:, a]
    out.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval_{out_suffix}.csv")
    return V


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all
    DREP_MODEL    = 'BOTH'  # 'EXPO', 'OLS', or 'BOTH'

    os.makedirs(eval_dir, exist_ok=True)

    models_to_run = (['EXPO', 'OLS'] if DREP_MODEL == 'BOTH'
                     else [DREP_MODEL.upper()])

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k        = int(row['k1'])
        flavor   = row['flavor_Y']
        filename = row['filename']
        print(f'\n{"="*60}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*60}')
        for model in models_to_run:
            run_eval_drep(filename, outcome_model=model)

    print('\nDone.')
