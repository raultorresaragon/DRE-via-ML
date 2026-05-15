# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dstar_freq_new_i0_stage2.py
# Decision distribution table for the new_i0 evaluation — two-stage simple DGP.
#
# For each (k, flavor) outputs a CSV with four columns:
#   d_star         : combination label, e.g. {0}, {1}, {0,0}, …
#   observed A     : count in the new_i0 dataset (A1 for stage-1 rows; (A1,A2) for joint)
#   d_star_DRE-ML  : count recommended by DRE-ML policy
#   d_star_DRE-p   : count recommended by DRE-Param policy  [optional]
#
# Rows (two-stage):
#   Stage-1 marginal : {0}, {1}, … {k-1}
#   Joint (d1, d2)   : {0,0}, {0,1}, … {k-1,k-1}
#
# Output: _1trt_effect/2stages/tables/new_i0/_dstar_freq_k{k}_{flavor}.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from itertools import product

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
tables_dir   = os.path.join(script_dir,   '../_1trt_effect/2stages/tables/new_i0')


def make_freq_table(k, flavor_Y, include_drep=True):
    """Build and save the d_star frequency table for one (k, flavor)."""
    fname_new = f"s2_k{k}_simple_{flavor_Y}_new_i0"
    dre_path  = os.path.join(new_i0_dir, f'{fname_new}_DRE.csv')
    drep_path = os.path.join(new_i0_dir, f'{fname_new}_DREp.csv')
    dat_path  = os.path.join(new_i0_dir, f'{fname_new}.csv')
    out_path  = os.path.join(tables_dir,  f'_dstar_freq_k{k}_{flavor_Y}.csv')

    if not os.path.exists(dre_path):
        print(f"  Missing {fname_new}_DRE.csv — skipping.")
        return

    df_dre = pd.read_csv(dre_path)
    df_dat = pd.read_csv(dat_path)

    has_drep = include_drep and os.path.exists(drep_path)
    df_drep  = pd.read_csv(drep_path) if has_drep else None

    n     = len(df_dat)
    arms  = list(range(k))
    rows  = []

    # ------------------------------------------------------------------
    # Stage-1 marginal rows
    # ------------------------------------------------------------------
    for a in arms:
        row = {
            'd_star':        f'{{{a}}}',
            'observed A':    round((df_dat['A1'] == a).sum() / n, 4),
            'd_star_DRE-ML': round((df_dre['d_star_1'] == a).sum() / n, 4),
        }
        if has_drep:
            row['d_star_DRE-p'] = round((df_drep['d_star_1'] == a).sum() / n, 4)
        rows.append(row)

    # ------------------------------------------------------------------
    # Joint (d1, d2) rows
    # ------------------------------------------------------------------
    for a1, a2 in product(arms, arms):
        obs_mask = (df_dat['A1'] == a1) & (df_dat['A2'] == a2)
        dre_mask = (df_dre['d_star_1'] == a1) & (df_dre['d_star_2'] == a2)
        row = {
            'd_star':        f'{{{a1},{a2}}}',
            'observed A':    round(obs_mask.sum() / n, 4),
            'd_star_DRE-ML': round(dre_mask.sum() / n, 4),
        }
        if has_drep:
            drep_mask = (df_drep['d_star_1'] == a1) & (df_drep['d_star_2'] == a2)
            row['d_star_DRE-p'] = round(drep_mask.sum() / n, 4)
        rows.append(row)

    freq_df = pd.DataFrame(rows)
    os.makedirs(tables_dir, exist_ok=True)
    freq_df.to_csv(out_path, index=False)
    print(f"  Saved: _dstar_freq_k{k}_{flavor_Y}.csv")
    print(freq_df.to_string(index=False))
    return freq_df


if __name__ == '__main__':
    INCLUDE_DREP = True    # set False to omit d_star_DRE-p column
    K_FILTER     = None    # set to 2, 3, or 5; None = all

    os.makedirs(tables_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == 0].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        k      = int(row['k1'])
        flavor = row['flavor_Y']
        print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
        make_freq_table(k=k, flavor_Y=flavor, include_drep=INCLUDE_DREP)

    print('\nDone.')
