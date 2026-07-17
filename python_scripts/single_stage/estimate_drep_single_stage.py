# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_drep_single_stage.py
# Single-stage DRE estimator — PARAMETRIC version.
#
# Identical algorithm to estimate_dre_single_stage.py but replaces neural-network
# outcome and propensity models with:
#   Outcome  : sklearn LinearRegression (OLS) or GLM Gaussian+log link (EXPO)
#   Propensity: sklearn LogisticRegression
#
# The self-normalized AIPW (Hajek) formula for mu_hat is unchanged.
# Output saved as:
#   OLS  → {filename}_DREp_ols.csv
#   EXPO → {filename}_DREp_expo.csv
#
# Models saved as:
#   OLS  → datasets/models/{filename}_DREp_ols_models.pkl
#   EXPO → datasets/models/{filename}_DREp_expo_models.pkl
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle
import warnings
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# Helpers
# ============================================================

def _fit_outcome(X, y, tag=''):
    """Fit LinearRegression (OLS); return fitted model."""
    print(f"    fitting outcome OLS {tag}(n={len(y)})...")
    model = LinearRegression()
    model.fit(X, y)
    return model


def _fit_outcome_expo(X, y, tag=''):
    """GLM with Gaussian family and log link: E[Y|X] = exp(X @ beta)."""
    print(f"    fitting outcome GLM-expo {tag}(n={len(y)})...")
    X_sm = sm.add_constant(X, has_constant='add')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sm.GLM(y, X_sm,
                        family=sm.families.Gaussian(
                            link=sm.families.links.Log())).fit(maxiter=200, method='irls')
        #result = sm.GLM(y, X_sm, 
        #                family=sm.families.Gamma(
        #                    link=sm.families.links.Log())).fit(maxiter=200, method='irls')

    return result


def _predict_expo(result, X):
    """Predict from a fitted GLM-expo (statsmodels) result; adds constant internally."""
    X_sm = sm.add_constant(X, has_constant='add')
    return result.predict(X_sm)


def _fit_pscore(X, a, k, tag=''):
    """Fit LogisticRegression; return fitted model."""
    print(f"    fitting pscore logistic {tag}...")
    model = LogisticRegression(max_iter=1000, penalty=None)
    model.fit(X, a.astype(int))
    return model


def _predict_proba_ordered(model, X, k):
    """Return (n, k) probability matrix with columns in arm order 0..k-1."""
    raw = model.predict_proba(X)          # (n, len(classes_))
    out = np.zeros((len(X), k))
    for col_idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col_idx]
    return out


def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """Self-normalized AIPW (Hajek-style) modified outcome."""
    n      = len(A_obs)
    mu_hat = np.zeros((n, k))
    for a in range(k):
        I_a  = (A_obs == a).astype(float)
        pi_a = np.clip(pi_hat_all[:, a], 1e-6, 1 - 1e-6)          # CORRECT
        #pi_a = np.clip(1 - pi_hat_all[:, a], 1e-6, 1 - 1e-6)      # BUG: uses P(A≠a|X)
        w_a  = np.mean(I_a / pi_a)
        mu_hat[:, a] = Y_hat_all[:, a] + I_a * (Y_obs - Y_hat_all[:, a]) / pi_a / w_a
    return mu_hat


# ============================================================
# Main estimator
# ============================================================

def estimate_drep(filename, dgp='single', outcome_model='OLS'):
    """
    Single-stage parametric DRE estimator.

    Parameters
    ----------
    filename      : str   Base filename without extension (e.g. 's1_k2_expo_0')
    dgp           : str   Reserved for compatibility; always reads _info_single.csv
    outcome_model : str   'OLS' (LinearRegression) or 'EXPO' (GLM Gaussian+log link)

    Returns
    -------
    DataFrame with columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
    """
    outcome_model = outcome_model.upper()
    if outcome_model not in ('OLS', 'EXPO'):
        raise ValueError(f"outcome_model must be 'OLS' or 'EXPO', got '{outcome_model}'")

    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k        = int(row['k'])
    flavor_Y = row['flavor_Y']

    X_cols = [c for c in dat.columns if c.startswith('X')]

    # DREp uses numpy arrays (not DataFrames) for parametric models
    X = dat[X_cols].values
    A = dat['A'].values
    Y = dat['Y'].values
    n = len(dat)

    model_tag = 'GLM-expo' if outcome_model == 'EXPO' else 'OLS'
    print(f"\n{'='*60}")
    print(f"DRE-Param ({model_tag}) estimation (single-stage): {filename}")
    print(f"  n={n}, k={k}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # Outcome models (T-learner: one model per arm)
    # ==================================================================
    print(f"\n[{model_tag} outcome models]")
    models_Y  = {}
    Y_hat_all = np.zeros((n, k))
    for a in range(k):
        mask = (A == a)
        if outcome_model == 'EXPO':
            model_a          = _fit_outcome_expo(X[mask], Y[mask], tag=f'a={a} ')
            Y_hat_all[:, a]  = _predict_expo(model_a, X)
        else:
            model_a          = _fit_outcome(X[mask], Y[mask], tag=f'a={a} ')
            Y_hat_all[:, a]  = model_a.predict(X)
        models_Y[a] = model_a

    # ==================================================================
    # Extract delta_1_hat and Delta_1_hat from outcome model coefficients
    # ==================================================================
    print(f"\n[Extracting delta_1 / Delta_1 from outcome models]")
    delta_rows = []
    for a in range(1, k):
        m_a = models_Y[a]
        m_0 = models_Y[0]
        if outcome_model == 'EXPO':
            # params: [const, X1, ..., X_last]; const is prepended by sm.add_constant
            delta_1_hat = float(m_a.params[0] - m_0.params[0])
            Delta_1_hat = float(m_a.params[-1] - m_0.params[-1])
        else:  # OLS
            delta_1_hat = float(m_a.intercept_ - m_0.intercept_)
            Delta_1_hat = float(m_a.coef_[-1]  - m_0.coef_[-1])
        print(f"    a={a}: delta_1_hat={delta_1_hat:.4f}, Delta_1_hat={Delta_1_hat:.4f}")
        delta_rows.append({'arm': a, 'delta_1_hat': delta_1_hat, 'Delta_1_hat': Delta_1_hat})

    delta_suffix = '_deltas_DREp_expo' if outcome_model == 'EXPO' else '_deltas_DREp_ols'
    delta_df     = pd.DataFrame(delta_rows)
    delta_path   = os.path.join(datasets_dir, f'{filename}{delta_suffix}.csv')
    delta_df.to_csv(delta_path, index=False)
    print(f"  ✓ Deltas saved: {filename}{delta_suffix}.csv")

    # ==================================================================
    # Propensity score model
    # ==================================================================
    print(f"\n[Propensity score]")
    ps     = _fit_pscore(X, A, k, tag='A ')
    pi_hat = _predict_proba_ordered(ps, X, k)

    # ==================================================================
    # AIPW modified outcomes and decisions
    # ==================================================================
    mu_hat = compute_mu_hat(A, Y, Y_hat_all, pi_hat, k)
    d_star = np.argmax(mu_hat, axis=1)
    print(f"\n  d_star distribution: {np.bincount(d_star)}")

    # ==================================================================
    # Save models (pkl)
    # ==================================================================
    pkl_suffix = 'DREp_expo' if outcome_model == 'EXPO' else 'DREp_ols'
    os.makedirs(models_dir, exist_ok=True)
    models_path = os.path.join(models_dir, f'{filename}_{pkl_suffix}_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models_Y':      models_Y,
            'ps':            ps,
            'Y_hat_all':     Y_hat_all,
            'X_cols':        X_cols,
            'k':             k,
            'outcome_model': outcome_model,
        }, f)
    print(f"  ✓ Models saved: {filename}_{pkl_suffix}_models.pkl")

    # ==================================================================
    # Save output CSV
    # ==================================================================
    csv_suffix = '_DREp_expo' if outcome_model == 'EXPO' else '_DREp_ols'
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'mu_hat_a{a}'] = mu_hat[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}{csv_suffix}.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}{csv_suffix}.csv")
    return out


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
            estimate_drep(row['filename'], dgp='single', outcome_model=model)
