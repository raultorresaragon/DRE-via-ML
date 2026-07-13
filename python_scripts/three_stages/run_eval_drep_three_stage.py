# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_drep_three_stage.py
# Load DRE-Param pkl, apply outcome models to eval data, save predictions.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_drep_three_stage.py and applies the stored outcome models to the
# evaluation dataset to compute d_star and V(DRE-Param).
#
# No AIPW is applied — only outcome model predictions are used for decision-making.
#
# Input:  datasets/models/{filename}_DREp_ols_models.pkl   (OLS)
#         datasets/models/{filename}_DREp_expo_models.pkl  (EXPO)
#         eval_sets/{filename}_eval.csv
# Output:
#   OLS  → eval_sets/{filename}_eval_DREp_ols.csv
#   EXPO → eval_sets/{filename}_eval_DREp_expo.csv
#   columns: d_star_1, d_star_2, d_star_3,
#            Y_hat_1_a0..Y_hat_1_a{k1-1},
#            Y_hat_2_a0..Y_hat_2_a{k2-1},
#            Y_hat_3_a0..Y_hat_3_a{k3-1}
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
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


def _predict_expo(result, X):
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)


def run_eval_drep(filename, outcome_model='OLS'):
    """
    Load the DREp pkl for `filename`, apply to eval data, save _eval_DREp_{suffix}.csv.

    Parameters
    ----------
    filename      : str   Base filename without extension (e.g. 's3_k2_simple_expo_0')
    outcome_model : str   'OLS' or 'EXPO'
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('OLS', 'EXPO'):
        raise ValueError(f"outcome_model must be 'OLS' or 'EXPO', got '{outcome_model}'")

    pkl_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'
    csv_suffix = '_DREp_expo' if outcome_model == 'EXPO' else '_DREp_ols'

    pkl_path  = os.path.join(models_dir, f'{filename}_{pkl_suffix}_models.pkl')
    eval_path = os.path.join(eval_dir,   f'{filename}_eval.csv')
    out_path  = os.path.join(eval_dir,   f'{filename}_eval{csv_suffix}.csv')

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

    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    models_Y1 = pkg['models_Y1']
    models_Y2 = pkg['models_Y2']
    models_Y3 = pkg['models_Y3']
    X1_cols   = pkg['X1_cols']
    k1        = pkg['k1']
    k2        = pkg['k2']
    k3        = pkg['k3']

    dat_ev = pd.read_csv(eval_path)
    n_ev   = len(dat_ev)

    X1_ev = dat_ev[X1_cols].values
    Y1_ev = dat_ev['Y_1'].values
    Y2_ev = dat_ev['Y_2'].values

    print(f"  n_ev={n_ev}, k1={k1}, k2={k2}, k3={k3}, outcome_model={outcome_model}")

    def _pred(model, X):
        if outcome_model == 'EXPO':
            return _predict_expo(model, X)
        return model.predict(X)

    # Stage 1
    print("\n  [Stage 1 — applying outcome models to eval data]")
    Y1_hat_ev = np.zeros((n_ev, k1))
    for a in range(k1):
        Y1_hat_ev[:, a] = _pred(models_Y1[a], X1_ev)
    d_star_1 = np.argmax(Y1_hat_ev, axis=1)
    print(f"  d_star_1: {np.bincount(d_star_1)}")

    # Stage 2
    print("\n  [Stage 2 — applying outcome models to eval data]")
    Y2_hat_ev = np.zeros((n_ev, k2))
    for a in range(k2):
        resid_a         = (Y1_ev - Y1_hat_ev[:, a]).reshape(-1, 1)
        feat_2_a        = np.hstack([X1_ev, resid_a])
        Y2_hat_ev[:, a] = _pred(models_Y2[a], feat_2_a)
    d_star_2 = np.argmax(Y2_hat_ev, axis=1)
    print(f"  d_star_2: {np.bincount(d_star_2)}")

    # Stage 3
    print("\n  [Stage 3 — applying outcome models to eval data]")
    Y3_hat_ev = np.zeros((n_ev, k3))
    for a in range(k3):
        a1_idx          = min(a, k1 - 1)
        a2_idx          = min(a, k2 - 1)
        resid1_a        = (Y1_ev - Y1_hat_ev[:, a1_idx]).reshape(-1, 1)
        resid2_a        = (Y2_ev - Y2_hat_ev[:, a2_idx]).reshape(-1, 1)
        feat_3_a        = np.hstack([X1_ev, resid1_a, resid2_a])
        Y3_hat_ev[:, a] = _pred(models_Y3[a], feat_3_a)
    d_star_3 = np.argmax(Y3_hat_ev, axis=1)
    print(f"  d_star_3: {np.bincount(d_star_3)}")

    V = float(np.mean(Y3_hat_ev[np.arange(n_ev), d_star_3]))
    print(f"\n  V(DRE-Param {model_tag}) = {V:.4f}")

    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2, 'd_star_3': d_star_3})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_ev[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_ev[:, a]
    for a in range(k3):
        out[f'Y_hat_3_a{a}'] = Y3_hat_ev[:, a]
    out.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval{csv_suffix}.csv")
    return V


if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all
    DREP_MODEL    = 'BOTH'  # 'OLS', 'EXPO', or 'BOTH'

    models_to_run = ['OLS', 'EXPO'] if DREP_MODEL == 'BOTH' else [DREP_MODEL.upper()]

    os.makedirs(eval_dir, exist_ok=True)

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
