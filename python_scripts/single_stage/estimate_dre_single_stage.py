# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_single_stage.py
# Single-stage DRE-ML estimator using neural-network outcome and propensity models.
#
# Algorithm
# ---------
#   Outcome:  For each arm a, fit NN on A=a subset using X as features → model_Y_a
#             Predict Y_hat_a for all individuals using model_Y_a
#   Pscore:   pi_hat_a = P(A=a | X)   via multinomial NN
#   Modified: mu_hat_a = Y_hat_a + I(A=a) * (Y - Y_hat_a) / pi_a / mean(I(A=a) / pi_a)
#             [self-normalized AIPW / Hajek-style]
#   Decision: d_star = argmax_a mu_hat_a
#
# Saved pkl keys: models_Y, pscore_A, Y_hat_all, X_cols, k
# Output CSV:     {filename}_DRE.csv   columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import random
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
info_path    = os.path.join(datasets_dir, '_info_single.csv')

DEFAULT_HIDUNITS = [random.randint(10, 115) for _ in range(15)]
DEFAULT_EPS      = [random.randint(40, 150) for _ in range(10)]
DEFAULT_PENALS   = [0.001, 0.01, 0.1]


# ============================================================
# Helpers
# ============================================================

def _predict_proba_ordered(model, X_df, k):
    """Return (n, k) probability matrix with columns in arm order 0..k-1."""
    raw = model.predict_proba(X_df)          # (n, len(classes_))
    out = np.zeros((len(X_df), k))
    for col_idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col_idx]
    return out


def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    Self-normalized AIPW (Hajek-style) modified outcome.

    mu_hat_a = Y_hat_a  +  I(A=a) * (Y - Y_hat_a) / pi_a  /  mean(I(A=a) / pi_a)

    Parameters
    ----------
    A_obs      : (n,)    observed treatment
    Y_obs      : (n,)    observed outcome
    Y_hat_all  : (n, k)  predicted outcome under each treatment level
    pi_hat_all : (n, k)  estimated P(A=a | X) for each level a

    Returns
    -------
    mu_hat : (n, k)
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

def estimate_dre(filename, dgp='single', hidunits=None, eps=None, penals=None):
    """
    Single-stage DRE-ML estimator.

    Parameters
    ----------
    filename : str          Base filename without extension (e.g. 's1_k2_expo_0')
    dgp      : str          Reserved for compatibility; always reads _info_single.csv
    hidunits : list or None NN hidden layer configurations for grid search
    eps      : list or None NN epoch counts for grid search
    penals   : list or None NN regularization strengths for grid search

    Returns
    -------
    DataFrame with columns: d_star, mu_hat_a0, ..., mu_hat_a{k-1}
    """
    hidunits = hidunits or DEFAULT_HIDUNITS
    eps      = eps      or DEFAULT_EPS
    penals   = penals   or DEFAULT_PENALS

    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k        = int(row['k'])
    flavor_Y = row['flavor_Y']

    X_cols = [c for c in dat.columns if c.startswith('X')]
    X      = dat[X_cols].reset_index(drop=True)   # DataFrame for NN models
    A      = dat['A'].values
    Y      = dat['Y'].values
    n      = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE-ML estimation (single-stage): {filename}")
    print(f"  n={n}, k={k}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # Outcome models (T-learner: one NN per arm, fitted on arm subset)
    # ==================================================================
    print("\n[Outcome models — T-learner]")
    models_Y  = {}
    Y_hat_all = np.zeros((n, k))
    for a in range(k):
        mask   = (A == a)
        dat_a  = pd.DataFrame(X.values[mask], columns=X_cols)
        dat_a['Y'] = Y[mask]
        n_a    = mask.sum()
        small  = n_a < 30
        print(f"    fitting outcome NN a={a} (n={n_a})...")
        model_a = Y_model_nn(dat=dat_a, y_func='Y~.',
                              hidunits=hidunits, eps=eps, penals=penals,
                              cvs=min(3, n_a // 2) if small else 6)
        Y_hat_all[:, a] = model_a.predict(X)
        models_Y[a]     = model_a

    # ==================================================================
    # Propensity score model
    # ==================================================================
    print("\n[Propensity score model]")
    dat_ps    = X.copy()
    dat_ps['A'] = A.astype(int)
    print(f"    fitting pscore NN A (n={n})...")
    pscore_A  = A_model_nn(dat=dat_ps, a_func='A~.',
                            hidunits=hidunits, eps=eps, penals=penals)
    pi_hat    = _predict_proba_ordered(pscore_A, X, k)   # (n, k)

    # ==================================================================
    # AIPW modified outcomes and decisions
    # ==================================================================
    mu_hat = compute_mu_hat(A, Y, Y_hat_all, pi_hat, k)
    d_star = np.argmax(mu_hat, axis=1)
    print(f"\n  d_star distribution: {np.bincount(d_star)}")

    # ==================================================================
    # Save models (pkl)
    # ==================================================================
    os.makedirs(models_dir, exist_ok=True)
    models_path = os.path.join(models_dir, f'{filename}_DRE_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models_Y':  models_Y,
            'pscore_A':  pscore_A,
            'Y_hat_all': Y_hat_all,
            'X_cols':    X_cols,
            'k':         k,
        }, f)
    print(f"  ✓ Models saved: {filename}_DRE_models.pkl")

    # ==================================================================
    # Save output CSV
    # ==================================================================
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'mu_hat_a{a}'] = mu_hat[:, a]

    out_path = os.path.join(datasets_dir, f'{filename}_DRE.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}_DRE.csv")
    return out


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5 to run only that k; None = run all
    FLAVOR_FILTER = None   # set to 'expo', 'gamma', etc.; None = all

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        estimate_dre(row['filename'])
