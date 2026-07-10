# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_dre_two_stage.py
# Load DRE-ML pkl, apply outcome models to eval data, save per-replication predictions.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_dre_two_stage.py and applies the stored outcome models to the
# evaluation dataset to compute d_star and V(DRE-ML).
#
# No AIPW is applied — only outcome model predictions are used for decision-making.
#
# Input:  datasets/models/{filename}_DRE_models.pkl
#         eval_per_i/{filename}_eval.csv
# Output: eval_per_i/{filename}_eval_DRE.csv
#   columns: d_star_1, d_star_2,
#            Y_hat_1_a0..Y_hat_1_a{k1-1},
#            Y_hat_2_a0..Y_hat_2_a{k2-1}
#
# V(DRE-ML) = mean(Y_hat_2[i, d_star_2[i]])  — summarised by vplot_eval_two_stage.py
#
# Optional filters in __main__:
#   K_FILTER      : int or None  — only run for k = K_FILTER
#   FLAVOR_FILTER : str or None  — only run for a specific flavor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
eval_dir     = os.path.join(datasets_dir, 'eval_per_i')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


# ============================================================
# Per-dataset runner
# ============================================================

def run_eval_dre(filename):
    """
    Load the DRE pkl for `filename`, apply to eval data, save _eval_DRE.csv.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's2_k2_simple_expo_0')
    """
    pkl_path  = os.path.join(models_dir, f'{filename}_DRE_models.pkl')
    eval_path = os.path.join(eval_dir,   f'{filename}_eval.csv')
    out_path  = os.path.join(eval_dir,   f'{filename}_eval_DRE.csv')

    if not os.path.exists(pkl_path):
        print(f"  Skipping {filename}: pkl not found ({pkl_path})")
        return None
    if not os.path.exists(eval_path):
        print(f"  Skipping {filename}: eval CSV not found ({eval_path})")
        return None

    print(f"\n{'='*60}")
    print(f"run_eval_dre: {filename}")
    print(f"{'='*60}")

    # ---- Load pkl ----
    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    models_Y1 = pkg['models_Y1']   # dict: arm → fitted NN outcome model
    models_Y2 = pkg['models_Y2']   # dict: arm → fitted NN outcome model
    X1_cols   = pkg['X1_cols']
    X2_cols   = pkg['X2_cols']
    k1        = pkg['k1']
    k2        = pkg['k2']

    # ---- Load eval data ----
    dat_ev = pd.read_csv(eval_path)
    n_ev   = len(dat_ev)

    # DRE NNs were fitted on DataFrames — predict with DataFrames
    X1_ev_df = dat_ev[X1_cols].reset_index(drop=True)
    X2_ev_df = dat_ev[X2_cols].reset_index(drop=True)
    Y1_ev    = dat_ev['Y_1'].values

    print(f"  n_ev={n_ev}, k1={k1}, k2={k2}")

    # ---- Stage 1: apply outcome models to eval X1 ----
    print("\n  [Stage 1 — applying outcome models to eval data]")
    Y1_hat_ev = np.zeros((n_ev, k1))
    for a in range(k1):
        Y1_hat_ev[:, a] = models_Y1[a].predict(X1_ev_df)
    d_star_1 = np.argmax(Y1_hat_ev, axis=1)
    print(f"  d_star_1: {np.bincount(d_star_1)}")

    # ---- Stage 2: apply outcome models to eval data ----
    # Residuals are computed from eval predictions (not training Y1_hat_all).
    # feat_2_a for eval: [X1_ev, X2_ev, Y1_ev - Y1_hat_ev[:, a]]
    print("\n  [Stage 2 — applying outcome models to eval data]")
    Y2_hat_ev = np.zeros((n_ev, k2))
    for a in range(k2):
        resid_a  = Y1_ev - Y1_hat_ev[:, a]
        feat_2_a = pd.concat([X1_ev_df,
                               X2_ev_df,
                               pd.Series(resid_a, name='Y1_resid')], axis=1)
        Y2_hat_ev[:, a] = models_Y2[a].predict(feat_2_a)
    d_star_2 = np.argmax(Y2_hat_ev, axis=1)
    print(f"  d_star_2: {np.bincount(d_star_2)}")

    V = float(np.mean(Y2_hat_ev[np.arange(n_ev), d_star_2]))
    print(f"\n  V(DRE-ML) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star_1': d_star_1, 'd_star_2': d_star_2})
    for a in range(k1):
        out[f'Y_hat_1_a{a}'] = Y1_hat_ev[:, a]
    for a in range(k2):
        out[f'Y_hat_2_a{a}'] = Y2_hat_ev[:, a]
    out.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval_DRE.csv")
    return V


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all

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
        run_eval_dre(filename)

    print('\nDone.')
