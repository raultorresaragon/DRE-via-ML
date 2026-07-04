# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_drep_single_stage.py
# Single-stage forward DRE estimator — PARAMETRIC version.
#
# Identical algorithm to estimate_dre_single_stage.py but replaces NN models with:
#   Outcome   : sklearn LinearRegression  (one model per arm, T-learner)
#   Propensity: sklearn LogisticRegression
#
# The self-normalized AIPW (Hajek) formula for mu_hat is unchanged.
# Output saved as {filename}_DREp.csv — same column layout as _DRE.csv.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, LogisticRegression

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Helpers
# ============================================================

def _fit_outcome(X, y, tag=''):
    print(f"    fitting outcome OLS {tag}(n={len(y)})...")
    return LinearRegression().fit(X, y)


def _fit_pscore(X, a, tag=''):
    print(f"    fitting pscore logistic {tag}...")
    return LogisticRegression(max_iter=1000).fit(X, a.astype(int))


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

def estimate_drep(filename):
    """
    Single-stage parametric DRE estimator.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')

    Returns
    -------
    DataFrame with columns: d_star_1, mu_hat_1_a0, mu_hat_1_a1, ...
    """
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k        = int(row['k1'])
    flavor_Y = row['flavor_Y']

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1 = dat[X1_cols].values
    A1 = dat['A1'].values
    Y  = dat['Y'].values
    n  = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE-Param estimation (single-stage): {filename}")
    print(f"  n={n}, k={k}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # STAGE 1 — outcome models (X1 features, T-learner)
    # ==================================================================
    print("\n[Stage 1 — outcome models]")
    Y1_hat_all = np.zeros((n, k))
    for a in range(k):
        mask    = (A1 == a)
        model_a = _fit_outcome(X1[mask], Y[mask], tag=f'Y a={a} ')
        Y1_hat_all[:, a] = model_a.predict(X1)

    # ------------------------------------------------------------------
    # STAGE 1 — propensity scores
    # ------------------------------------------------------------------
    print("\n[Stage 1 — propensity score]")
    ps1     = _fit_pscore(X1, A1, tag='A1 ')
    pi1_hat = _predict_proba_ordered(ps1, X1, k)

    mu_hat_1 = compute_mu_hat(A1, Y, Y1_hat_all, pi1_hat, k)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"\n  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ==================================================================
    # Save output
    # ==================================================================
    out = pd.DataFrame({'d_star_1': d_star_1})
    for a in range(k):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_DREp.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}_DREp.csv")
    return out


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER = None   # set to 2, 3, or 5 to run only that k; None = run all
    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    for _, row in info.iterrows():
        estimate_drep(row['filename'])
