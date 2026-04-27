# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_drep_three_stage.py
# Three-stage forward DRE estimator — PARAMETRIC version.
#
# Identical algorithm to estimate_dre_three_stage.py but replaces neural-network
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
datasets_dir     = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
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
    raw = model.predict_proba(X)
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
    Forward three-stage parametric DRE estimator.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's3_k2_simple_expo_0')
    dgp      : str   'simple' reads _info_simple.csv; 'standard' reads _info.csv

    Returns
    -------
    DataFrame with columns: d_star_1..3, mu_hat_1_a0..., mu_hat_2_a0..., mu_hat_3_a0...
    """
    info_fname = '_info_simple.csv' if dgp == 'simple' else '_info.csv'
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(os.path.join(datasets_dir, info_fname))
    row  = info[info['filename'] == filename].iloc[0]

    k1, k2, k3 = int(row['k1']), int(row['k2']), int(row['k3'])
    flavor_Y   = row['flavor_Y']

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1 = dat[X1_cols].values
    A1 = dat['A1'].values
    Y1 = dat['Y_1'].values
    A2 = dat['A2'].values
    Y2 = dat['Y_2'].values
    A3 = dat['A3'].values
    Y  = dat['Y'].values
    n  = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE-Param estimation (3-stage): {filename}")
    print(f"  n={n}, k1={k1}, k2={k2}, k3={k3}, flavor_Y={flavor_Y}")
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
    # STAGE 2 — outcome models ([X1, Y1_resid] features)
    # ==================================================================
    print("\n[Stage 2 — outcome models]")
    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        resid_a  = (Y1 - Y1_hat_all[:, a]).reshape(-1, 1)
        feat_2_a = np.hstack([X1, resid_a])
        mask     = (A2 == a)
        model_a  = _fit_outcome(feat_2_a[mask], Y2[mask], tag=f'Y2 a={a} ')
        Y2_hat_all[:, a] = model_a.predict(feat_2_a)

    print("\n[Stage 2 — propensity score]")
    feat_ps2 = np.hstack([X1, A1.reshape(-1, 1), Y1.reshape(-1, 1)])
    ps2      = _fit_pscore(feat_ps2, A2, k2, tag='A2 ')
    pi2_hat  = _predict_proba_ordered(ps2, feat_ps2, k2)

    mu_hat_2 = compute_mu_hat(A2, Y2, Y2_hat_all, pi2_hat, k2)
    d_star_2 = np.argmax(mu_hat_2, axis=1)
    print(f"\n  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ==================================================================
    # STAGE 3 — outcome models ([X1, Y1_resid, Y2_resid] features)
    # ==================================================================
    print("\n[Stage 3 — outcome models]")
    Y3_hat_all = np.zeros((n, k3))
    for a in range(k3):
        a1_idx   = min(a, k1 - 1)
        a2_idx   = min(a, k2 - 1)
        resid1_a = (Y1 - Y1_hat_all[:, a1_idx]).reshape(-1, 1)
        resid2_a = (Y2 - Y2_hat_all[:, a2_idx]).reshape(-1, 1)
        feat_3_a = np.hstack([X1, resid1_a, resid2_a])
        mask     = (A3 == a)
        model_a  = _fit_outcome(feat_3_a[mask], Y[mask], tag=f'Y3 a={a} ')
        Y3_hat_all[:, a] = model_a.predict(feat_3_a)

    print("\n[Stage 3 — propensity score]")
    feat_ps3 = np.hstack([X1,
                           A1.reshape(-1, 1), Y1.reshape(-1, 1),
                           A2.reshape(-1, 1), Y2.reshape(-1, 1)])
    ps3      = _fit_pscore(feat_ps3, A3, k3, tag='A3 ')
    pi3_hat  = _predict_proba_ordered(ps3, feat_ps3, k3)

    mu_hat_3 = compute_mu_hat(A3, Y, Y3_hat_all, pi3_hat, k3)
    d_star_3 = np.argmax(mu_hat_3, axis=1)
    print(f"\n  d_star_3 distribution: {np.bincount(d_star_3)}")

    # ==================================================================
    # Save output
    # ==================================================================
    out = pd.DataFrame({
        'd_star_1': d_star_1,
        'd_star_2': d_star_2,
        'd_star_3': d_star_3,
    })
    for a in range(k1):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]
    for a in range(k2):
        out[f'mu_hat_2_a{a}'] = mu_hat_2[:, a]
    for a in range(k3):
        out[f'mu_hat_3_a{a}'] = mu_hat_3[:, a]

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
