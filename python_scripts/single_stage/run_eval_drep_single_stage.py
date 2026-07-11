# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_drep_single_stage.py
# Load DRE-Param pkl, apply outcome models to eval data, save per-replication predictions.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_drep_single_stage.py and applies the stored outcome models to the
# evaluation dataset to compute d_star and V(DRE-Param).
#
# No AIPW is applied — only outcome model predictions are used for decision-making.
# V(d*) = mean(Y_hat[i, d_star[i]])   (single-stage)
#
# Two outcome model options, controlled by DREP_MODEL in __main__:
#   'EXPO' : GLM with Gaussian family + log link  → eval_sets/{filename}_eval_DREp_expo.csv
#   'OLS'  : sklearn LinearRegression             → eval_sets/{filename}_eval_DREp_ols.csv
#   'BOTH' : run EXPO then OLS, saving both files
#
# Input:  datasets/models/{filename}_DREp_ols_models.pkl   (OLS)
#         datasets/models/{filename}_DREp_expo_models.pkl  (EXPO)
#         eval_sets/{filename}_eval.csv
# Output: eval_sets/{filename}_eval_DREp_expo.csv  (EXPO)
#         eval_sets/{filename}_eval_DREp_ols.csv   (OLS)
#   columns: d_star, Y_hat_a0, ..., Y_hat_a{k-1}
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
import warnings
import statsmodels.api as sm

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# EXPO prediction helper (mirrors estimate_drep_single_stage.py)
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
    filename      : str   Base filename without extension (e.g. 's1_k2_expo_0')
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

    models_Y = pkg['models_Y']   # dict: arm → fitted parametric outcome model
    X_cols   = pkg['X_cols']
    k        = pkg['k']

    # ---- Load eval data ----
    dat_ev = pd.read_csv(eval_path)
    n_ev   = len(dat_ev)

    # DREp models were fitted on numpy arrays
    X_ev = dat_ev[X_cols].values

    print(f"  n_ev={n_ev}, k={k}, outcome_model={outcome_model}")

    # ---- Apply outcome models to eval data ----
    print("\n  [Outcome models — predicting on eval data]")
    Y_hat_ev = np.zeros((n_ev, k))
    for a in range(k):
        if outcome_model == 'EXPO':
            Y_hat_ev[:, a] = _predict_expo(models_Y[a], X_ev)
        else:
            Y_hat_ev[:, a] = models_Y[a].predict(X_ev)
    d_star = np.argmax(Y_hat_ev, axis=1)
    print(f"  d_star: {np.bincount(d_star)}")

    V = float(np.mean(Y_hat_ev[np.arange(n_ev), d_star]))
    print(f"\n  V(DRE-Param {model_tag}) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'Y_hat_a{a}'] = Y_hat_ev[:, a]
    out.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval_{out_suffix}.csv")
    return V


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'gamma', etc.; None = all
    DREP_MODEL    = 'BOTH'  # 'EXPO', 'OLS', or 'BOTH'

    os.makedirs(eval_dir, exist_ok=True)

    models_to_run = (['EXPO', 'OLS'] if DREP_MODEL == 'BOTH'
                     else [DREP_MODEL.upper()])

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k        = int(row['k'])
        flavor   = row['flavor_Y']
        filename = row['filename']
        print(f'\n{"="*60}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*60}')
        for model in models_to_run:
            run_eval_drep(filename, outcome_model=model)

    print('\nDone.')
