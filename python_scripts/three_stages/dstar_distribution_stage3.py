# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dstar_distribution_stage3.py
# Treatment allocation distribution table for a chosen eval dataset (three-stage).
#
# Reads from eval_sets/:
#   {filename}_eval.csv        — observed (A1, A2, A3)
#   {filename}_eval_DRE.csv    — DRE-ML d_star_1, d_star_2, d_star_3
#   {filename}_eval_DREp.csv   — DRE-Param (ols) d_star_1, d_star_2, d_star_3  [optional]
#
# Output columns:
#   d_star | observed A | d_star_DRE-ML | d_star_DREp-ols
#
# Rows:
#   Stage-1 marginal    : {0}, {1}, … {k-1}
#   Stage-1×2 joint     : {0,0}, {0,1}, … {k-1,k-1}
#   Stage-1×2×3 joint   : {0,0,0}, … {k-1,k-1,k-1}
#
# Control in __main__:
#   FILENAME  — base training filename (e.g. 's3_k2_simple_expo_0'); '_eval' appended auto.
#               Set to None to run over all rows in _info_simple.csv.
#   K_FILTER, FLAVOR_FILTER — used only when FILENAME is None.
#
# Output: _1trt_effect/3stages/tables/eval_sets/_dstar_dist_{filename}_eval.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from itertools import product

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
tables_dir   = os.path.join(script_dir,   '../_1trt_effect/3stages/tables/eval_sets')


def make_dist_table(filename, include_drep_ols=True):
    """
    Build and print/save the d_star distribution table for one eval dataset.

    Parameters
    ----------
    filename      : str   Base training filename (e.g. 's3_k2_simple_expo_0').
                          '_eval' is appended to form the eval file stem.
    include_drep_ols : bool  Include DRE-Param (ols) column if file exists.

    Returns
    -------
    pd.DataFrame or None
    """
    eval_stem    = f'{filename}_eval'
    dat_path     = os.path.join(eval_dir, f'{eval_stem}.csv')
    dre_path     = os.path.join(eval_dir, f'{eval_stem}_DRE.csv')
    drep_path    = os.path.join(eval_dir, f'{eval_stem}_DREp.csv')
    out_path     = os.path.join(tables_dir, f'_dstar_dist_{eval_stem}.csv')

    if not os.path.exists(dat_path):
        print(f"  Missing eval dataset: {eval_stem}.csv — skipping.")
        return None
    if not os.path.exists(dre_path):
        print(f"  Missing DRE-ML predictions: {eval_stem}_DRE.csv — skipping.")
        return None

    df_dat = pd.read_csv(dat_path)
    df_dre = pd.read_csv(dre_path)

    has_ols = include_drep_ols and os.path.exists(drep_path)
    df_ols  = pd.read_csv(drep_path) if has_ols else None

    k = int(df_dat['A1'].max()) + 1
    n = len(df_dat)
    arms = list(range(k))

    print(f"\n{'='*60}")
    print(f"d* distribution: {eval_stem}  (k={k}, n={n})")
    print(f"{'='*60}")

    rows = []

    # ------------------------------------------------------------------
    # Stage-1 marginal rows
    # ------------------------------------------------------------------
    for a in arms:
        row = {
            'd_star':        f'{{{a}}}',
            'observed A':    round((df_dat['A1'] == a).sum() / n, 4),
            'd_star_DRE-ML': round((df_dre['d_star_1'] == a).sum() / n, 4),
        }
        if has_ols:
            row['d_star_DREp-ols'] = round((df_ols['d_star_1'] == a).sum() / n, 4)
        rows.append(row)

    # ------------------------------------------------------------------
    # Stage-1×2 joint rows
    # ------------------------------------------------------------------
    for a1, a2 in product(arms, arms):
        obs_mask = (df_dat['A1'] == a1) & (df_dat['A2'] == a2)
        dre_mask = (df_dre['d_star_1'] == a1) & (df_dre['d_star_2'] == a2)
        row = {
            'd_star':        f'{{{a1},{a2}}}',
            'observed A':    round(obs_mask.sum() / n, 4),
            'd_star_DRE-ML': round(dre_mask.sum() / n, 4),
        }
        if has_ols:
            ols_mask = (df_ols['d_star_1'] == a1) & (df_ols['d_star_2'] == a2)
            row['d_star_DREp-ols'] = round(ols_mask.sum() / n, 4)
        rows.append(row)

    # ------------------------------------------------------------------
    # Stage-1×2×3 joint rows
    # ------------------------------------------------------------------
    for a1, a2, a3 in product(arms, arms, arms):
        obs_mask = (df_dat['A1'] == a1) & (df_dat['A2'] == a2) & (df_dat['A3'] == a3)
        dre_mask = (df_dre['d_star_1'] == a1) & (df_dre['d_star_2'] == a2) & (df_dre['d_star_3'] == a3)
        row = {
            'd_star':        f'{{{a1},{a2},{a3}}}',
            'observed A':    round(obs_mask.sum() / n, 4),
            'd_star_DRE-ML': round(dre_mask.sum() / n, 4),
        }
        if has_ols:
            ols_mask = (df_ols['d_star_1'] == a1) & (df_ols['d_star_2'] == a2) & (df_ols['d_star_3'] == a3)
            row['d_star_DREp-ols'] = round(ols_mask.sum() / n, 4)
        rows.append(row)

    dist_df = pd.DataFrame(rows)
    os.makedirs(tables_dir, exist_ok=True)
    dist_df.to_csv(out_path, index=False)
    print(dist_df.to_string(index=False))
    print(f"\n  ✓ Saved: _dstar_dist_{eval_stem}.csv")
    return dist_df


if __name__ == '__main__':
    # Set FILENAME to the base training name (without '_eval') to inspect one dataset.
    # Set to None to loop over all rows in _info_simple.csv (filtered by K/FLAVOR below).
    FILENAME      = 's3_k2_simple_expo_0'   # e.g. 's3_k3_simple_gamma_5' or None
    K_FILTER      = None    # used only when FILENAME is None; set to 2, 3, or 5
    FLAVOR_FILTER = None    # used only when FILENAME is None; e.g. 'expo', 'gamma'

    INCLUDE_DREP_OLS = True

    os.makedirs(tables_dir, exist_ok=True)

    if FILENAME is not None:
        make_dist_table(FILENAME, include_drep_ols=INCLUDE_DREP_OLS)
    else:
        info = pd.read_csv(info_path)
        if K_FILTER is not None:
            info = info[info['k1'] == K_FILTER]
        if FLAVOR_FILTER is not None:
            info = info[info['flavor_Y'] == FLAVOR_FILTER]
        for _, row in info.iterrows():
            make_dist_table(row['filename'], include_drep_ols=INCLUDE_DREP_OLS)

    print('\nDone.')
