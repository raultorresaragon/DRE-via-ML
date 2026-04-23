# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate_dre_three_stage.py
# Three-stage forward DRE estimator using self-normalized AIPW modified outcomes.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Algorithm (forward, stage 1 → stage 2 → stage 3)
# --------------------------------------------------
# Stage 1
#   Outcome:  For each a1, fit NN on A1=a1 subset using X1 → model_Y1_a1
#             Predict Yhat_1_a1 for all individuals
#   Pscore:   pi_hat_1_a = P(A1=a | X1)
#   Modified: mu_hat_1_a  (self-normalized AIPW on Y1)
#
# Stage 2
#   Outcome:  For each a2, fit NN on A2=a2 subset using [X1, Y1_resid_a2] → model_Y2_a2
#             Y1_resid_a2 = Y1 - Yhat_1_a2  (links stages)
#             Predict Yhat_2_a2 for all individuals
#   Pscore:   pi_hat_2_a = P(A2=a | X1, A1, Y1)
#   Modified: mu_hat_2_a  (self-normalized AIPW on Y2)
#
# Stage 3
#   Outcome:  For each a3, fit NN on A3=a3 subset using [X1, Y1_resid_a3, Y2_resid_a3]
#             Y2_resid_a3 = Y2 - Yhat_2_a3  (links stages 2→3)
#             Predict Yhat_3_a3 for all individuals
#   Pscore:   pi_hat_3_a = P(A3=a | X1, A1, Y1, A2, Y2)
#   Modified: mu_hat_3_a  (self-normalized AIPW on Y)
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
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
info_path    = os.path.join(datasets_dir, '_info.csv')
info_path_simple = os.path.join(datasets_dir, '_info_simple.csv')

DEFAULT_HIDUNITS = [5, 20]
DEFAULT_EPS      = [100, 250]
DEFAULT_PENALS   = [0.001, 0.01]


# ============================================================
# Helpers
# ============================================================

def _fit_outcome_nn(features_df, y_values, hidunits, eps, penals, tag=''):
    """Fit Y_model_nn on a feature DataFrame and target array."""
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals,
                      cvs=min(3, n // 2) if small else 6,
                      early_stopping=not small)


def _fit_pscore_nn(features_df, a_values, hidunits, eps, penals, tag=''):
    """Fit A_model_nn on a feature DataFrame and treatment array."""
    dat = features_df.copy()
    dat['A'] = a_values.astype(int)
    print(f"    fitting pscore NN {tag}...")
    return A_model_nn(dat=dat, hidunits=hidunits, eps=eps, penals=penals)


def compute_mu_hat(A_obs, Y_obs, Y_hat_all, pi_hat_all, k):
    """
    Self-normalized AIPW (Hajek-style) modified outcome for each treatment level.

    mu_hat_a = Yhat_a  +  I(A=a) * (Y - Yhat_a) / pi_a  /  mean(I(A=a) / pi_a)

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
                 verbose=False,
                 dgp='simple'):
    """
    Forward three-stage DRE estimator.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_simple_expo_0')
    dgp      : str   'simple' reads from _info_simple.csv (default);
                     'standard' reads from _info.csv

    Returns
    -------
    DataFrame with columns: d_star_1, d_star_2, d_star_3,
                             mu_hat_1_a0..., mu_hat_2_a0..., mu_hat_3_a0...
    """
    # ------------------------------------------------------------------
    # Load dataset + metadata
    # ------------------------------------------------------------------
    info_fname = '_info_simple.csv' if dgp == 'simple' else '_info.csv'
    dat  = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    info = pd.read_csv(os.path.join(datasets_dir, info_fname))
    row  = info[info['filename'] == filename].iloc[0]

    k1, k2   = int(row['k1']), int(row['k2'])
    k3       = int(row['k3'])
    flavor_Y = row['flavor_Y']

    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X1 = dat[X1_cols].reset_index(drop=True)
    A1 = dat['A1'].values
    Y1 = dat['Y_1'].values
    A2 = dat['A2'].values
    Y2 = dat['Y_2'].values
    A3 = dat['A3'].values
    Y  = dat['Y'].values
    n  = len(dat)

    print(f"\n{'='*60}")
    print(f"DRE estimation (3-stage): {filename}")
    print(f"  n={n}, k1={k1}, k2={k2}, k3={k3}, flavor_Y={flavor_Y}")
    print(f"{'='*60}")

    # ==================================================================
    # STAGE 1 — outcome models
    # Features: X1 only (arm-specific fits; A1 constant within each arm)
    # ==================================================================
    print("\n[Stage 1 — outcome models]")

    Y1_hat_all = np.zeros((n, k1))
    for a in range(k1):
        mask    = (A1 == a)
        model_a = _fit_outcome_nn(X1[mask].reset_index(drop=True),
                                  Y1[mask], hidunits, eps, penals, tag=f'Y1 a={a} ')
        Y1_hat_all[:, a] = model_a.predict(X1)

    # Stage 1 — propensity scores
    print("\n[Stage 1 — propensity score]")
    pscore_A1 = _fit_pscore_nn(X1, A1, hidunits, eps, penals, tag='A1 ')
    pi1_hat   = pscore_A1.predict_proba(X1)    # (n, k1)

    mu_hat_1 = compute_mu_hat(A1, Y1, Y1_hat_all, pi1_hat, k1)
    d_star_1 = np.argmax(mu_hat_1, axis=1)
    print(f"\n  d_star_1 distribution: {np.bincount(d_star_1)}")

    # ==================================================================
    # STAGE 2 — outcome models
    # Features: [X1, Y1_resid_a2]  where Y1_resid_a2 = Y1 - Yhat_1_a2
    # ==================================================================
    print("\n[Stage 2 — outcome models]")

    Y2_hat_all = np.zeros((n, k2))
    for a in range(k2):
        resid_a  = Y1 - Y1_hat_all[:, a]
        feat_2_a = pd.concat([X1,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        mask    = (A2 == a)
        model_a = _fit_outcome_nn(feat_2_a[mask].reset_index(drop=True),
                                  Y2[mask], hidunits, eps, penals, tag=f'Y2 a={a} ')
        Y2_hat_all[:, a] = model_a.predict(feat_2_a)

    # Stage 2 — propensity scores
    # Features: [X1, A1, Y1]
    print("\n[Stage 2 — propensity score]")
    feat_ps2  = pd.concat([X1,
                            pd.Series(A1, name='A1'),
                            pd.Series(Y1, name='Y_1')], axis=1)
    pscore_A2 = _fit_pscore_nn(feat_ps2, A2, hidunits, eps, penals, tag='A2 ')
    pi2_hat   = pscore_A2.predict_proba(feat_ps2)    # (n, k2)

    mu_hat_2 = compute_mu_hat(A2, Y2, Y2_hat_all, pi2_hat, k2)
    d_star_2 = np.argmax(mu_hat_2, axis=1)
    print(f"\n  d_star_2 distribution: {np.bincount(d_star_2)}")

    # ==================================================================
    # STAGE 3 — outcome models
    # Features: [X1, Y1_resid_a3, Y2_resid_a3]
    #   Y1_resid_a3 = Y1 - Yhat_1_a3  (use arm a3 prediction — links stage 1)
    #   Y2_resid_a3 = Y2 - Yhat_2_a3  (links stage 2)
    # For k3 == k1 == k2, we reuse Y1_hat_all and Y2_hat_all columns by arm index.
    # If k3 differs from k1 or k2, clamp the index to the available arms.
    # ==================================================================
    print("\n[Stage 3 — outcome models]")

    Y3_hat_all = np.zeros((n, k3))
    for a in range(k3):
        a1_idx   = min(a, k1 - 1)   # clamp if k3 > k1
        a2_idx   = min(a, k2 - 1)   # clamp if k3 > k2
        resid1_a = Y1 - Y1_hat_all[:, a1_idx]
        resid2_a = Y2 - Y2_hat_all[:, a2_idx]
        feat_3_a = pd.concat([X1,
                               pd.Series(resid1_a, name='Y1_resid'),
                               pd.Series(resid2_a, name='Y2_resid')], axis=1)
        mask    = (A3 == a)
        model_a = _fit_outcome_nn(feat_3_a[mask].reset_index(drop=True),
                                  Y[mask], hidunits, eps, penals, tag=f'Y3 a={a} ')
        Y3_hat_all[:, a] = model_a.predict(feat_3_a)

    # Stage 3 — propensity scores
    # Features: [X1, A1, Y1, A2, Y2]
    print("\n[Stage 3 — propensity score]")
    feat_ps3  = pd.concat([X1,
                            pd.Series(A1, name='A1'),
                            pd.Series(Y1, name='Y_1'),
                            pd.Series(A2, name='A2'),
                            pd.Series(Y2, name='Y_2')], axis=1)
    pscore_A3 = _fit_pscore_nn(feat_ps3, A3, hidunits, eps, penals, tag='A3 ')
    pi3_hat   = pscore_A3.predict_proba(feat_ps3)    # (n, k3)

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

    out_path = os.path.join(datasets_dir, f'{filename}_DRE.csv')
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {filename}_DRE.csv")

    return out


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    info = pd.read_csv(info_path_simple)
    for _, row in info.iterrows():
        estimate_dre(row['filename'], dgp='simple')
