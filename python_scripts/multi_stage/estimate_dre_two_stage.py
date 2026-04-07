# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_two_stage.py
# Two-stage forward DRE estimator using AIPW-style modified outcomes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Algorithm (forward, stage 1 → stage 2)
# ---------------------------------------
# Stage 1
#   Outcome:  Y_hat_1_a  = NN(X1, A=a, A*X_bin)  for each a
#   Pscore:   pi_hat_1_a = P(A1=a | X1)
#   Modified: mu_hat_1_a = I(A1=a)*Y1/pi_1_a + (I(A1=a)-pi_1_a)/pi_1_a * Y_hat_1_a
#
# Stage 2
#   Outcome:  Y_hat_2_a  = NN(X1, Y_1, X2, A1, Y_1-Y_hat_1_a)  fitted on A2=a subset, for each a
#   Pscore:   pi_hat_2_a = P(A2=a | X1, A1, Y_1, X2)
#   Modified: mu_hat_2_a = I(A2=a)*Y/pi_2_a + (I(A2=a)-pi_2_a)/pi_2_a * Y_hat_2_a
#
# Decision
#   d_star_k = argmax_a mu_hat_k_a
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
info_path    = os.path.join(datasets_dir, '_info.csv')

DEFAULT_HIDUNITS = [5, 20]
DEFAULT_EPS      = [100, 250]
DEFAULT_PENALS   = [0.001, 0.01]


# ============================================================
# Helpers
# ============================================================

def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    """Attach target column and fit Y_model_nn.
    For small subsets (n < 30) early stopping is disabled and CV folds are reduced
    to avoid sklearn errors from undersized validation splits."""
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6,
                      early_stopping=not small)


def _fit_pscore_nn(features_df, a_values, hidunits, eps, penals, tag=''):
    """Attach treatment column and fit A_model_nn."""
    dat = features_df.copy()
    dat['A'] = a_values.astype(int)
    print(f"    fitting pscore NN {tag}...")
    return A_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals)


def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    AIPW-style modified outcome for each treatment level.

    mu_hat_a = I(A=a)*Y / pi_a  +  (I(A=a) - pi_a) / pi_a * Y_hat_a

    Parameters
    ----------
    A_obs      : (n,)    observed treatment
    Y_obs      : (n,)    observed outcome
    Y_hat_all  : (n, k)  predicted outcome under each treatment level
    pi_hat_all : (n, k)  estimated P(A=a | history) for each level a

    Returns
    -------
    mu_hat : (n, k)
    """
    n = len(A_obs)
    mu_hat = np.zeros((n, k))
    for a in range(k):
        I_a  = (A_obs == a).astype(float)
        pi_a = np.clip(pi_hat_all[:, a], 1e-6, 1 - 1e-6)
        mu_hat[:, a] = (I_a * Y_obs / pi_a
                        + (I_a - pi_a) / pi_a * Y_hat_all[:, a])
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
    Forward two-stage DRE estimator.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_logit_expo_0')

    Returns
    -------
    DataFrame with columns: d_star_1, d_star_2, mu_hat_1_max, mu_hat_2_max
    """
    # ------------------------------------------------------------------
    # Load dataset + metadata
    # ------------------------------------------------------------------
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(info_path)
    row  = info[info['filename'] == filename].iloc[0]

    k1, k2   = int(row['k1']), int(row['k2'])
    flavor_Y = row['flavor_Y']
    i        = int(row['i'])

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X2_cols = [c for c in dat.columns if c.startswith('X2_')]   # excludes Y_1

    X1    = dat[X1_cols].reset_index(drop=True)
    X2    = dat[X2_cols].reset_index(drop=True)
    A1    = dat['A1'].values
    A2    = dat['A2'].values
    Y1    = dat['Y_1'].values
    Y     = dat['Y'].values
    X_bin = X1.iloc[:, -1].values    # binary modifier: last column of X1
    n     = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE estimation: {filename}")
    print(f"  n={n}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # STAGE 1
    # ==================================================================
    print("\n[Stage 1 — outcome model]")

    # Fit one model on all data with A1 as a covariate
    feat_1_obs = pd.concat([X1,
                             pd.Series(A1,         name='A1'),
                             pd.Series(A1 * X_bin, name='A1_Xbin')], axis=1)
    model_Y1 = _fit_outcome_nn(feat_1_obs, Y1, hidunits, eps, penals, tag='Y1 ')

    # Predict counterfactually for each treatment level a
    Y1_hat_all = np.zeros((n, k1))
    for a in range(k1):
        feat_1_a = pd.concat([X1,
                               pd.Series(np.full(n, a), name='A1'),
                               pd.Series(a * X_bin,     name='A1_Xbin')], axis=1)
        Y1_hat_all[:, a] = model_Y1.predict(feat_1_a)

    print("\n[Stage 1 — propensity score]")
    pscore_A1   = _fit_pscore_nn(X1, A1, hidunits, eps, penals, tag='A1 ')
    pi1_hat     = pscore_A1.predict_proba(X1)    # (n, k1): P(A1=a | X1)

    mu_hat_1     = compute_mu_hat(A1, Y1, Y1_hat_all, pi1_hat, k1)
    d_star_1     = np.argmax(mu_hat_1, axis=1)
    mu_hat_1_max = mu_hat_1[np.arange(n), d_star_1]

    print(f"\n  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ==================================================================
    # STAGE 2
    # ==================================================================
    print("\n[Stage 2 — outcome models]")

    # For each a: build features using Y_1 - Y_hat_1_a as residual,
    # fit on A2=a subset, predict for all individuals
    Y2_hat_all = np.zeros((n, k2))

    for a in range(k2):
        resid_a  = Y1 - Y1_hat_all[:, a]

        feat_2_a = pd.concat([X1,
                               pd.Series(Y1,      name='Y_1'),
                               X2,
                               pd.Series(A1,      name='A1'),
                               pd.Series(resid_a, name='Y1_resid')], axis=1)

        mask    = (A2 == a)
        model_a = _fit_outcome_nn(feat_2_a[mask].reset_index(drop=True),
                                  Y[mask], hidunits, eps, penals, tag=f'Y2 a={a} ')
        Y2_hat_all[:, a] = model_a.predict(feat_2_a)

    print("\n[Stage 2 — propensity score]")
    feat_ps2  = pd.concat([X1,
                            pd.Series(A1, name='A1'),
                            pd.Series(Y1, name='Y_1'),
                            X2], axis=1)
    pscore_A2 = _fit_pscore_nn(feat_ps2, A2, hidunits, eps, penals, tag='A2 ')
    pi2_hat   = pscore_A2.predict_proba(feat_ps2)    # (n, k2): P(A2=a | history)

    mu_hat_2     = compute_mu_hat(A2, Y, Y2_hat_all, pi2_hat, k2)
    d_star_2     = np.argmax(mu_hat_2, axis=1)
    mu_hat_2_max = mu_hat_2[np.arange(n), d_star_2]

    print(f"\n  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ==================================================================
    # Save output
    # ==================================================================
    out = pd.DataFrame({
        'd_star_1':     d_star_1,
        'd_star_2':     d_star_2,
        'mu_hat_1_max': mu_hat_1_max,
        'mu_hat_2_max': mu_hat_2_max,
    })

    out_filename = f'{filename}_DRE_{i}'
    out_path = os.path.join(datasets_dir, f'{out_filename}.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_filename}.csv")

    return out


# ============================================================
# Run over all datasets in _info.csv
# ============================================================
info = pd.read_csv(info_path)
for _, row in info.iterrows():
    estimate_dre(row['filename'])
