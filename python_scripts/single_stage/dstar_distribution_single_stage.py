# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dstar_distribution_single_stage.py
# Treatment allocation distribution table for a chosen eval dataset (single-stage).
#
# Reads from eval_sets/:
#   {filename}_eval.csv             — observed treatment A
#   {filename}_eval_DRE.csv         — DRE-ML d_star
#   {filename}_eval_DREp_expo.csv   — DRE-Param (expo) d_star  [optional]
#   {filename}_eval_DREp_ols.csv    — DRE-Param (ols)  d_star  [optional]
#
# Output columns:
#   d_star | observed A | d_star_DRE-ML | d_star_DREp-expo | d_star_DREp-ols
#
# Rows (marginal only — no joint rows for single stage):
#   {0}, {1}, ..., {k-1}
#
# Control in __main__:
#   FILENAME  — base training filename (e.g. 's1_k2_expo_0'); '_eval' appended auto.
#               Set to None to run over all rows in _info_single.csv.
#   K_FILTER, FLAVOR_FILTER — used only when FILENAME is None.
#
# Output: _1trt_effect/1stage/tables/eval_sets/_dstar_dist_{eval_stem}.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')
tables_dir   = os.path.join(script_dir,   '../_1trt_effect/1stage/tables/eval_sets')


def make_dist_table(filename, include_drep_expo=True, include_drep_ols=True):
    """
    Build and print/save the d_star distribution table for one eval dataset.

    Parameters
    ----------
    filename          : str   Base training filename (e.g. 's1_k2_expo_0').
                               '_eval' is appended to form the eval file stem.
    include_drep_expo : bool  Include DRE-Param (expo) column if file exists.
    include_drep_ols  : bool  Include DRE-Param (ols) column if file exists.

    Returns
    -------
    pd.DataFrame or None
    """
    eval_stem      = f'{filename}_eval'
    dat_path       = os.path.join(eval_dir, f'{eval_stem}.csv')
    dre_path       = os.path.join(eval_dir, f'{eval_stem}_DRE.csv')
    drep_expo_path = os.path.join(eval_dir, f'{eval_stem}_DREp_expo.csv')
    drep_ols_path  = os.path.join(eval_dir, f'{eval_stem}_DREp_ols.csv')
    out_path       = os.path.join(tables_dir, f'_dstar_dist_{eval_stem}.csv')

    if not os.path.exists(dat_path):
        print(f"  Missing eval dataset: {eval_stem}.csv — skipping.")
        return None
    if not os.path.exists(dre_path):
        print(f"  Missing DRE-ML predictions: {eval_stem}_DRE.csv — skipping.")
        return None

    df_dat = pd.read_csv(dat_path)
    df_dre = pd.read_csv(dre_path)

    has_expo = include_drep_expo and os.path.exists(drep_expo_path)
    has_ols  = include_drep_ols  and os.path.exists(drep_ols_path)
    df_expo  = pd.read_csv(drep_expo_path) if has_expo else None
    df_ols   = pd.read_csv(drep_ols_path)  if has_ols  else None

    # Infer k from observed treatment levels (single-stage: column 'A')
    k = int(df_dat['A'].max()) + 1
    n = len(df_dat)
    arms = list(range(k))

    print(f"\n{'='*60}")
    print(f"d* distribution: {eval_stem}  (k={k}, n={n})")
    print(f"{'='*60}")

    rows = []

    # ------------------------------------------------------------------
    # Marginal rows only — no joint rows for single stage
    # ------------------------------------------------------------------
    for a in arms:
        row = {
            'd_star':         f'{{{a}}}',
            'observed A':     round((df_dat['A'] == a).sum() / n, 4),
            'd_star_DRE-ML':  round((df_dre['d_star'] == a).sum() / n, 4),
        }
        if has_expo:
            row['d_star_DREp-expo'] = round((df_expo['d_star'] == a).sum() / n, 4)
        if has_ols:
            row['d_star_DREp-ols']  = round((df_ols['d_star'] == a).sum() / n, 4)
        rows.append(row)

    dist_df = pd.DataFrame(rows)
    os.makedirs(tables_dir, exist_ok=True)
    dist_df.to_csv(out_path, index=False)
    print(dist_df.to_string(index=False))
    print(f"\n  ✓ Saved: _dstar_dist_{eval_stem}.csv")
    return dist_df


if __name__ == '__main__':
    # Set FILENAME to the base training name (without '_eval') to inspect one dataset.
    # Set to None to loop over all rows in _info_single.csv (filtered by K/FLAVOR below).
    FILENAME      = 's1_k2_expo_0'   # e.g. 's1_k3_gamma_5' or None
    K_FILTER      = None    # used only when FILENAME is None; set to 2, 3, or 5
    FLAVOR_FILTER = None    # used only when FILENAME is None; e.g. 'expo', 'gamma'

    INCLUDE_DREP_EXPO = True
    INCLUDE_DREP_OLS  = True

    os.makedirs(tables_dir, exist_ok=True)

    if FILENAME is not None:
        make_dist_table(FILENAME,
                        include_drep_expo=INCLUDE_DREP_EXPO,
                        include_drep_ols=INCLUDE_DREP_OLS)
    else:
        info = pd.read_csv(info_path)
        if K_FILTER is not None:
            info = info[info['k'] == K_FILTER]
        if FLAVOR_FILTER is not None:
            info = info[info['flavor_Y'] == FLAVOR_FILTER]
        for _, row in info.iterrows():
            make_dist_table(row['filename'],
                            include_drep_expo=INCLUDE_DREP_EXPO,
                            include_drep_ols=INCLUDE_DREP_OLS)

    print('\nDone.')
