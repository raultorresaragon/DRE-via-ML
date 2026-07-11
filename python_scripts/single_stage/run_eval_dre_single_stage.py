# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_eval_dre_single_stage.py
# Load DRE-ML pkl, apply outcome models to eval data, save per-replication predictions.
#
# This script does NOT re-train models. It loads the pkl produced by
# estimate_dre_single_stage.py and applies the stored outcome models to the
# evaluation dataset to compute d_star and V(DRE-ML).
#
# No AIPW is applied — only outcome model predictions are used for decision-making.
# V(d*) = mean(Y_hat[i, d_star[i]])   (single-stage)
#
# Input:  datasets/models/{filename}_DRE_models.pkl
#         eval_sets/{filename}_eval.csv
# Output: eval_sets/{filename}_eval_DRE.csv
#   columns: d_star, Y_hat_a0, ..., Y_hat_a{k-1}
#
# Optional filters in __main__:
#   K_FILTER      : int or None
#   FLAVOR_FILTER : str or None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys
import pickle

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
models_dir   = os.path.join(datasets_dir, 'models')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


# ============================================================
# Per-dataset runner
# ============================================================

def run_eval_dre(filename):
    """
    Load the DRE pkl for `filename`, apply to eval data, save _eval_DRE.csv.

    Parameters
    ----------
    filename : str   Base filename without extension (e.g. 's1_k2_expo_0')
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

    models_Y = pkg['models_Y']   # dict: arm → fitted NN outcome model
    X_cols   = pkg['X_cols']
    k        = pkg['k']

    # ---- Load eval data ----
    dat_ev = pd.read_csv(eval_path)
    n_ev   = len(dat_ev)

    # DRE NNs were fitted on DataFrames — predict with DataFrame
    X_ev = dat_ev[X_cols].reset_index(drop=True)

    print(f"  n_ev={n_ev}, k={k}")

    # ---- Apply outcome models to eval data ----
    print("\n  [Outcome models — predicting on eval data]")
    Y_hat_ev = np.zeros((n_ev, k))
    for a in range(k):
        Y_hat_ev[:, a] = models_Y[a].predict(X_ev)
    d_star = np.argmax(Y_hat_ev, axis=1)
    print(f"  d_star: {np.bincount(d_star)}")

    V = float(np.mean(Y_hat_ev[np.arange(n_ev), d_star]))
    print(f"\n  V(DRE-ML) = {V:.4f}")

    # ---- Save predictions ----
    out = pd.DataFrame({'d_star': d_star})
    for a in range(k):
        out[f'Y_hat_a{a}'] = Y_hat_ev[:, a]
    out.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved: {filename}_eval_DRE.csv")
    return V


# ============================================================
# Run over all datasets in _info_single.csv
# ============================================================
if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'gamma', etc.; None = all

    os.makedirs(eval_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k        = int(row['k'])
        flavor   = row['flavor_Y']
        filename = row['filename']
        print(f'\n{"="*60}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*60}')
        run_eval_dre(filename)

    print('\nDone.')
