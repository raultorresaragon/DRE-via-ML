# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_single_stage.py
# Single-stage forward DRE estimator using NN outcome and propensity models.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Algorithm
# ---------
# Stage 1
#   Outcome:  For each a, fit NN on A1=a subset using X1 as features → model_Y_a
#             Predict Yhat_1_a for all individuals using model_Y_a
#   Pscore:   pi_hat_1_a = P(A1=a | X1)
#   Modified: mu_hat_1_a = Yhat_1_a + I(A1=a)*(Y - Yhat_1_a)/pi_a / mean(I(A1=a)/pi_a)
#
# Decision
#   d_star_1 = argmax_a mu_hat_1_a
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import random
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn

DEFAULT_HIDUNITS = [random.randint(10, 115) for _ in range(12)]
DEFAULT_EPS      = [random.randint(40, 150) for _ in range(12)]
DEFAULT_PENALS   = [0.001, 0.005, 0.01]


# ============================================================
# Helpers
# ============================================================

def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6)


def _fit_pscore_nn(features_df, a_values, hidunits, eps, penals, tag=''):
    dat = features_df.copy()
    dat['A'] = a_values.astype(int)
    print(f"    fitting pscore NN {tag}...")
    return A_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals)


def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    Self-normalized AIPW (Hajek) modified outcome.

    mu_hat_a = Yhat_a  +  I(A=a) * (Y - Yhat_a) / pi_a  /  mean(I(A=a) / pi_a)
    """
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

def estimate_dre(filename,
                 hidunits=DEFAULT_HIDUNITS,
                 eps=DEFAULT_EPS,
                 penals=DEFAULT_PENALS,
                 verbose=False):
    """
    Single-stage DRE estimator.

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
    X1 = dat[X1_cols].reset_index(drop=True)
    A1 = dat['A1'].values
    Y  = dat['Y'].values
    n  = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE estimation (single-stage): {filename}")
    print(f"  n={n}, k={k}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # STAGE 1 — outcome models
    # ==================================================================
    print("\n[Stage 1 — outcome models]")
    Y1_hat_all = np.zeros((n, k))
    for a in range(k):
        mask    = (A1 == a)
        model_a = _fit_outcome_nn(X1[mask].reset_index(drop=True),
                                  Y[mask], hidunits, eps, penals, tag=f'Y a={a} ')
        Y1_hat_all[:, a] = model_a.predict(X1)

    # ------------------------------------------------------------------
    # STAGE 1 — propensity scores
    # ------------------------------------------------------------------
    print("\n[Stage 1 — propensity score]")
    pscore_A1 = _fit_pscore_nn(X1, A1, hidunits, eps, penals, tag='A1 ')
    pi1_hat   = pscore_A1.predict_proba(X1)    # (n, k)

    mu_hat_1 = compute_mu_hat(A1, Y, Y1_hat_all, pi1_hat, k)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"\n  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ==================================================================
    # Save output
    # ==================================================================
    out = pd.DataFrame({'d_star_1': d_star_1})
    for a in range(k):
        out[f'mu_hat_1_a{a}'] = mu_hat_1[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_DRE.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}_DRE.csv")
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
        estimate_dre(row['filename'])
