# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_drep_two_stage.py
# Two-stage forward DRE estimator — PARAMETRIC version.
#
# Identical algorithm to estimate_dre_two_stage.py but replaces neural-network
# outcome and propensity models with:
#   Outcome  : sklearn LinearRegression   (one model per arm, T-learner)
#   Propensity: sklearn LogisticRegression
#
# The self-normalized AIPW (Hajek) formula for mu_hat is unchanged.
# Output saved as {filename}_DREp.csv — same column layout as _DRE.csv.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, LogisticRegression

script_dir       = os.path.dirname(os.path.abspath(__file__))
datasets_dir     = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
info_path_simple = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helpers
# ============================================================

def _fit_outcome(X, y, tag=''):
    """Fit LinearRegression; return fitted model."""
    print(f"    fitting outcome OLS {tag}(n={len(y)})...")
    model = LinearRegression()
    model.fit(X, y)
    return model


def _fit_pscore(X, a, k, tag=''):
    """Fit LogisticRegression; return fitted model."""
    print(f"    fitting pscore logistic {tag}...")
    model = LogisticRegression(max_iter=1000)
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
        pi_a = np.clip(pi_hat_all[:, a], 1e-6, 1 - 1e-6)
        w_a  = np.mean(I_a / pi_a)
        mu_hat[:, a] = Y_hat_all[:, a] + I_a * (Y_obs - Y_hat_all[:, a]) / pi_a / w_a
    return mu_hat


# ============================================================
# Main estimator
# ============================================================

def estimate_drep(filename, dgp='simple'):
    """
    Forward two-stage parametric DRE estimator.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    dgp      : str   'simple' reads _info_simple.csv; 'standard' reads _info.csv

    Returns
    -------
    DataFrame with columns: d_star_1, d_star_2, mu_hat_1_a0..., mu_hat_2_a0...
    """
    info_fname = '_info_simple.csv' if dgp == 'simple' else '_info.csv'
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(os.path.join(datasets_dir, info_fname))
    row  = info[info['filename'] == filename].iloc[0]

    k1, k2   = int(row['k1']), int(row['k2'])
    flavor_Y = row['flavor_Y']

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X2_cols = [c for c in dat.columns if c.startswith('X2_')]

    X1 = dat[X1_cols].values
    X2 = dat[X2_cols].values if X2_cols else np.empty((len(dat), 0))
    A1 = dat['A1'].values
    A2 = dat['A2'].values
    Y1 = dat['Y_1'].values
    Y  = dat['Y'].values
    n  = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE-Param estimation (2-stage): {filename}")
    print(f"  n={n}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # STAGE 1 — outcome models (X1 features, T-learner)
    # ==================================================================
    print("\n[Stage 1 — outcome models]")
    Y1_hat_all = np.zeros((n, k1))
    for a in range(k1):
        mask    = (A1 == a)
        model_a = _fit_outcome(X1[mask], Y1[mask], tag=f'Y1 a={a} ')
        Y1_hat_all[:, a] = model_a.predict(X1)

    print("\n[Stage 1 — propensity score]")
    ps1     = _fit_pscore(X1, A1, k1, tag='A1 ')
    pi1_hat = _predict_proba_ordered(ps1, X1, k1)

    mu_hat_1 = compute_mu_hat(A1, Y1, Y1_hat_all, pi1_hat, k1)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"\n  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ==================================================================
    # STAGE 2 — outcome models ([X1, X2, Y1_resid] features)
    # ==================================================================
    print("\n[Stage 2 — outcome models]")
    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        resid_a  = (Y1 - Y1_hat_all[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1, X2, resid_a])
        mask     = (A2 == a)
        model_a  = _fit_outcome(feat_2_a[mask], Y[mask], tag=f'Y2 a={a} ')
        Y2_hat_all[:, a] = model_a.predict(feat_2_a)

    print("\n[Stage 2 — propensity score]")
    feat_ps2 = np.hstack([X1, A1.reshape(-1, 1), Y1.reshape(-1, 1), X2])
    ps2      = _fit_pscore(feat_ps2, A2, k2, tag='A2 ')
    pi2_hat  = _predict_proba_ordered(ps2, feat_ps2, k2)

    mu_hat_2 = compute_mu_hat(A2, Y, Y2_hat_all, pi2_hat, k2)
    d_star_2 = np.argmax(mu_hat_2, axis=1)
    print(f"\n  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ==================================================================
    # Save output
    # ==================================================================
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]
    for a in range(k2):
        out[f'mu_hat_2_a{a}'] = mu_hat_2[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_DREp.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}_DREp.csv")
    return out


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    info = pd.read_csv(info_path_simple)
    for _, row in info.iterrows():
        estimate_drep(row['filename'], dgp='simple')
