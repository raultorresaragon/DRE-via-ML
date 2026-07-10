# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_two_stage.py
# Fit DRE-ML and DRE-Param models on each training dataset i, apply to the
# corresponding evaluation dataset, and save per-replication predictions.
#
# For each row i in _info_simple.csv:
#   Training data  : _1trt_effect/2stages/datasets/{filename}.csv
#   Evaluation data: _1trt_effect/2stages/datasets/eval_per_i/{filename}_eval.csv
#   Models (DRE-ML): eval_per_i/models/{filename}_DRE_models.pkl
#   Models (DREp)  : eval_per_i/models/{filename}_DREp_models.pkl
#   Predictions    : eval_per_i/{filename}_eval_DRE.csv
#                    eval_per_i/{filename}_eval_DREp.csv
#     columns: d_star_1, d_star_2,
#              Y_hat_1_a0..Y_hat_1_a{k1-1},
#              Y_hat_2_a0..Y_hat_2_a{k2-1}
#
# V(DRE-ML)   = mean(Y_hat_2[i, d_star_2[i]])  — summarised by vplot_eval_two_stage.py
# V(DRE-Param)= mean(Y_hat_2[i, d_star_2[i]])
#
# Optional filters / settings (set in __main__):
#   K_FILTER      : int or None  — only run for k = K_FILTER
#   FLAVOR_FILTER : str or None  — only run for a specific flavor
#   RUN_DRE       : bool         — fit/apply DRE-ML        (default True)
#   RUN_DREP      : bool         — fit/apply DRE-Param      (default True)
#   DREP_MODEL    : str          — 'EXPO' (GLM log-link, default) or 'OLS'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import warnings
import numpy as np
import pandas as pd
import os
import sys
import random
import pickle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_per_i')
models_dir   = os.path.join(eval_dir,     'models')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn   # noqa: F401 — available for propensity if needed

DEFAULT_HIDUNITS = [random.randint(10, 115) for _ in range(12)]
DEFAULT_EPS      = [random.randint(40, 150) for _ in range(12)]
DEFAULT_PENALS   = [0.001, 0.005, 0.01]


# ============================================================
# Shared helpers
# ============================================================

def _fit_outcome_expo(X, y, tag=''):
    """GLM with Gaussian family and log link: E[Y|X] = exp(X @ beta)."""
    print(f"    fitting outcome GLM-expo {tag}(n={len(y)})...")
    X_sm = sm.add_constant(X, has_constant='add')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sm.GLM(y, X_sm,
                        family=sm.families.Gaussian(
                            link=sm.families.links.Log())).fit(
                                maxiter=200, method='irls')
    return result


def _predict_expo(result, X):
    """Predict from a fitted statsmodels GLM result (no intercept column in X)."""
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)

def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6,
                      early_stopping=not small)


def _load_train(datasets_dir, filename, k):
    """Load training dataset; return split components."""
    dat     = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X2_cols = [c for c in dat.columns if c.startswith('X2_')]
    return dat, X1_cols, X2_cols


def _load_eval(eval_dir, filename):
    """Load evaluation dataset."""
    return pd.read_csv(os.path.join(eval_dir, f'{filename}_eval.csv'))


def _save_predictions(out_df, eval_dir, filename, suffix):
    out_path = os.path.join(eval_dir, f'{filename}_eval_{suffix}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval_{suffix}.csv")


# ============================================================
# DRE-ML estimation (per replication i)
# ============================================================

def fit_dre_ml(filename, k, eval_dir, models_dir, datasets_dir,
               hidunits, eps, penals):
    """Fit DRE-ML on training dataset i; apply to eval dataset i."""
    k1 = k2 = k
    models_path = os.path.join(models_dir, f'{filename}_DRE_models.pkl')

    print(f"\n[DRE-ML] {filename}")

    # ---- Load training data ----
    dat_tr, X1_cols, X2_cols = _load_train(datasets_dir, filename, k)
    X1_tr = dat_tr[X1_cols].reset_index(drop=True)
    X2_tr = dat_tr[X2_cols].reset_index(drop=True)
    A1_tr = dat_tr['A1'].values
    A2_tr = dat_tr['A2'].values
    Y1_tr = dat_tr['Y_1'].values
    Y_tr  = dat_tr['Y'].values
    n_tr  = len(dat_tr)

    # ---- Stage 1: fit outcome models ----
    print("  [Stage 1 — outcome models on training data]")
    models_Y1     = {}
    Y1_hat_all_tr = np.zeros((n_tr, k1))
    for a in range(k1):
        mask = (A1_tr == a)
        m    = _fit_outcome_nn(X1_tr[mask].reset_index(drop=True),
                               Y1_tr[mask], hidunits, eps, penals,
                               tag=f'Y1 a={a} ')
        models_Y1[a]        = m
        Y1_hat_all_tr[:, a] = m.predict(X1_tr)

    # ---- Stage 2: fit outcome models ----
    print("  [Stage 2 — outcome models on training data]")
    models_Y2     = {}
    Y2_hat_all_tr = np.zeros((n_tr, k2))
    for a in range(k2):
        resid_a  = Y1_tr - Y1_hat_all_tr[:, a]
        feat_2_a = pd.concat([X1_tr, X2_tr,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        mask = (A2_tr == a)
        m    = _fit_outcome_nn(feat_2_a[mask].reset_index(drop=True),
                               Y_tr[mask], hidunits, eps, penals,
                               tag=f'Y2 a={a} ')
        models_Y2[a]        = m
        Y2_hat_all_tr[:, a] = m.predict(feat_2_a)

    # ---- Save models ----
    with open(models_path, 'wb') as f:
        pickle.dump({'models_Y1': models_Y1, 'models_Y2': models_Y2,
                     'k1': k1, 'k2': k2,
                     'X1_cols': X1_cols, 'X2_cols': X2_cols}, f)
    print(f"  ✓ Models saved: {os.path.basename(models_path)}")

    # ---- Load eval data ----
    dat_ev = _load_eval(eval_dir, filename)
    X1_ev  = dat_ev[X1_cols].reset_index(drop=True)
    X2_ev  = dat_ev[X2_cols].reset_index(drop=True)
    Y1_ev  = dat_ev['Y_1'].values
    n_ev   = len(dat_ev)

    # ---- Apply stage 1 to eval ----
    print("  [Stage 1 — applying to eval data]")
    Y1_hat_ev = np.zeros((n_ev, k1))
    for a in range(k1):
        Y1_hat_ev[:, a] = models_Y1[a].predict(X1_ev)
    d_star_1 = np.argmax(Y1_hat_ev, axis=1)

    # ---- Apply stage 2 to eval ----
    print("  [Stage 2 — applying to eval data]")
    Y2_hat_ev = np.zeros((n_ev, k2))
    for a in range(k2):
        resid_a  = Y1_ev - Y1_hat_ev[:, a]
        feat_2_a = pd.concat([X1_ev, X2_ev,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        Y2_hat_ev[:, a] = models_Y2[a].predict(feat_2_a)
    d_star_2 = np.argmax(Y2_hat_ev, axis=1)

    V = float(np.mean(Y2_hat_ev[np.arange(n_ev), d_star_2]))
    print(f"  V(DRE-ML) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_ev[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_ev[:, a]
    _save_predictions(out, eval_dir, filename, 'DRE')
    return V


# ============================================================
# DRE-Param estimation (per replication i)
# ============================================================

def fit_drep(filename, k, eval_dir, models_dir, datasets_dir, outcome_model='EXPO'):
    """
    Fit DRE-Param on training dataset i; apply to eval dataset i.

    Parameters
    ----------
    outcome_model : str  'EXPO' — GLM with Gaussian family + log link (default)
                         'OLS'  — sklearn LinearRegression
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('EXPO', 'OLS'):
        raise ValueError(f"outcome_model must be 'EXPO' or 'OLS', got '{outcome_model}'")

    k1 = k2 = k
    models_path = os.path.join(models_dir, f'{filename}_DREp_models.pkl')
    model_tag   = 'GLM-expo' if outcome_model == 'EXPO' else 'OLS'

    print(f"\n[DRE-Param ({model_tag})] {filename}")

    # ---- Load training data ----
    dat_tr, X1_cols, X2_cols = _load_train(datasets_dir, filename, k)
    X1_tr = dat_tr[X1_cols].values
    X2_tr = dat_tr[X2_cols].values
    A1_tr = dat_tr['A1'].values
    A2_tr = dat_tr['A2'].values
    Y1_tr = dat_tr['Y_1'].values
    Y_tr  = dat_tr['Y'].values
    n_tr  = len(dat_tr)

    # ---- Stage 1: outcome models ----
    print(f"  [Stage 1 — {model_tag} outcome models on training data]")
    models_Y1     = {}
    Y1_hat_all_tr = np.zeros((n_tr, k1))
    for a in range(k1):
        mask = (A1_tr == a)
        if outcome_model == 'EXPO':
            m                    = _fit_outcome_expo(X1_tr[mask], Y1_tr[mask], tag=f'Y1 a={a} ')
            Y1_hat_all_tr[:, a]  = _predict_expo(m, X1_tr)
        else:
            m                    = LinearRegression().fit(X1_tr[mask], Y1_tr[mask])
            Y1_hat_all_tr[:, a]  = m.predict(X1_tr)
            print(f"    OLS Y1 a={a}  (n={mask.sum()})")
        models_Y1[a] = m

    # ---- Stage 2: outcome models ----
    print(f"  [Stage 2 — {model_tag} outcome models on training data]")
    models_Y2     = {}
    Y2_hat_all_tr = np.zeros((n_tr, k2))
    for a in range(k2):
        resid_a  = (Y1_tr - Y1_hat_all_tr[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1_tr, X2_tr, resid_a])
        mask     = (A2_tr == a)
        if outcome_model == 'EXPO':
            m                    = _fit_outcome_expo(feat_2_a[mask], Y_tr[mask], tag=f'Y2 a={a} ')
            Y2_hat_all_tr[:, a]  = _predict_expo(m, feat_2_a)
        else:
            m                    = LinearRegression().fit(feat_2_a[mask], Y_tr[mask])
            Y2_hat_all_tr[:, a]  = m.predict(feat_2_a)
            print(f"    OLS Y2 a={a}  (n={mask.sum()})")
        models_Y2[a] = m

    # ---- Save models ----
    with open(models_path, 'wb') as f:
        pickle.dump({'models_Y1': models_Y1, 'models_Y2': models_Y2,
                     'k1': k1, 'k2': k2, 'outcome_model': outcome_model,
                     'X1_cols': X1_cols, 'X2_cols': X2_cols}, f)
    print(f"  ✓ Models saved: {os.path.basename(models_path)}")

    # ---- Load eval data ----
    dat_ev = _load_eval(eval_dir, filename)
    X1_ev  = dat_ev[X1_cols].values
    X2_ev  = dat_ev[X2_cols].values
    Y1_ev  = dat_ev['Y_1'].values
    n_ev   = len(dat_ev)

    # ---- Apply stage 1 to eval ----
    print("  [Stage 1 — applying to eval data]")
    Y1_hat_ev = np.zeros((n_ev, k1))
    for a in range(k1):
        Y1_hat_ev[:, a] = (_predict_expo(models_Y1[a], X1_ev)
                           if outcome_model == 'EXPO'
                           else models_Y1[a].predict(X1_ev))
    d_star_1 = np.argmax(Y1_hat_ev, axis=1)

    # ---- Apply stage 2 to eval ----
    print("  [Stage 2 — applying to eval data]")
    Y2_hat_ev = np.zeros((n_ev, k2))
    for a in range(k2):
        resid_a  = (Y1_ev - Y1_hat_ev[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1_ev, X2_ev, resid_a])
        Y2_hat_ev[:, a] = (_predict_expo(models_Y2[a], feat_2_a)
                           if outcome_model == 'EXPO'
                           else models_Y2[a].predict(feat_2_a))
    d_star_2 = np.argmax(Y2_hat_ev, axis=1)

    V = float(np.mean(Y2_hat_ev[np.arange(n_ev), d_star_2]))
    print(f"  V(DRE-Param) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_ev[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_ev[:, a]
    _save_predictions(out, eval_dir, filename, 'DREp')
    return V


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    K_FILTER      = None    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all
    RUN_DRE       = True    # fit and apply DRE-ML
    RUN_DREP      = True    # fit and apply DRE-Param
    DREP_MODEL    = 'EXPO'  # 'EXPO' = GLM with log link (default); 'OLS' = linear regression

    os.makedirs(eval_dir,   exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k        = int(row['k1'])
        flavor   = row['flavor_Y']
        filename = row['filename']

        # Check that the eval dataset exists
        eval_path = os.path.join(eval_dir, f'{filename}_eval.csv')
        if not os.path.exists(eval_path):
            print(f'\nSkipping {filename}: eval dataset not found '
                  f'(run gen_eval_datasets_two_stage.py first).')
            continue

        print(f'\n{"="*60}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*60}')

        if RUN_DRE:
            fit_dre_ml(filename, k, eval_dir, models_dir, datasets_dir,
                       DEFAULT_HIDUNITS, DEFAULT_EPS, DEFAULT_PENALS)

        if RUN_DREP:
            fit_drep(filename, k, eval_dir, models_dir, datasets_dir,
                     outcome_model=DREP_MODEL)

    print('\nDone.')
