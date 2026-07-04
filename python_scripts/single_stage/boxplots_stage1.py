# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_stage1.py
# ATE bias boxplots for the single-stage simple DGP.
#
# For each Y flavor, produces:
#   - One table : bias_naive, bias_DRE-ML, bias_DRE-Param per dataset i
#                 Saved to: _1trt_effect/1stages/tables/_ate_bias_{flavor}.csv
#   - One figure: Naive | DRE-ML | DRE-Param boxplots (per arm contrast)
#                 Saved to: _1trt_effect/1stages/images/_ate_bias_k{k}_{flavor}.jpeg
#
# Bias = estimated ATE - true ATE   (positive = overestimate)
# True ATE is computed analytically via ate_single_stage_simple.true_ate_simple.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/1stages/tables')
images_dir   = os.path.join(script_dir, '../_1trt_effect/1stages/images')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from ate_single_stage_simple import true_ate_simple

# Greyscale palette — Naive lightest, DRE-ML darkest, DRE-Param mid
C_BW = {'naive': '0.82', 'dre': '0.20', 'drep': '0.42'}

# Color palette (used when greyscale=False)
C_COLOR = {'naive': '#E57373', 'dre': '#64B5F6', 'drep': '#FFB74D'}


def _drop_extreme(arr, k=3.0):
    """Remove values beyond k×IQR from Q1/Q3."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


def _ate_from_file(filename, suffix, k):
    """Compute per-arm ATE from mu_hat columns in a _DRE, _DREp, or _NAIVE csv."""
    path = os.path.join(datasets_dir, f'{filename}{suffix}.csv')
    df   = pd.read_csv(path)
    return {a: float(np.mean(df[f'mu_hat_1_a{a}'] - df['mu_hat_1_a0']))
            for a in range(1, k)}


def build_records(sub_info):
    """Loop over rows in sub_info; return list of bias records."""
    records = []
    for _, row in sub_info.iterrows():
        fname = row['filename']
        k     = int(row['k1'])
        i_val = int(row['i'])
        try:
            true  = true_ate_simple(fname, verbose=False)
            naive = _ate_from_file(fname, '_NAIVE', k)
            dre   = _ate_from_file(fname, '_DRE',   k)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        try:
            drep = _ate_from_file(fname, '_DREp', k)
        except FileNotFoundError:
            drep = {a: np.nan for a in range(1, k)}

        for a in sorted(true['ATE_1'].keys()):
            t   = true['ATE_1'][a]
            n_  = naive.get(a, np.nan)
            d   = dre.get(a,   np.nan)
            dp  = drep.get(a,  np.nan)
            records.append({
                'i':               i_val,
                'k':               k,
                'arm':             a,
                'ATE_true':        t,
                'ATE_naive':       n_,
                'ATE_dre':         d,
                'ATE_drep':        dp,
                'rel_bias_naive':  (n_  - t) / abs(t) * 100 if t != 0 else np.nan,
                'rel_bias_dre':    (d   - t) / abs(t) * 100 if t != 0 else np.nan,
                'rel_bias_drep':   (dp  - t) / abs(t) * 100 if t != 0 else np.nan,
            })
    return records


def make_figure(df, flavor, arms, include_drep=True, greyscale=False):
    """
    One figure per (k, flavor): one set of boxplots (Naive | DRE-ML [| DRE-Param])
    per arm contrast.
    """
    n_arms   = len(arms)
    n_boxes  = 3 if include_drep else 2
    fig_w    = max(5, n_arms * n_boxes * 1.5)
    fig, ax  = plt.subplots(figsize=(fig_w, 5))

    title_flavor = 'loggamma' if flavor == 'gamma' else flavor
    ax.set_title(f'ATE Bias — Single-Stage DGP  ({title_flavor})', fontsize=12)

    palette     = C_BW if greyscale else C_COLOR
    all_data    = []
    all_colors  = []
    tick_labels = []
    positions   = []
    pos         = 1

    for a in arms:
        sub = df[df['arm'] == a]

        all_data.append(_drop_extreme(sub['rel_bias_naive'].values))
        all_colors.append(palette['naive'])
        tick_labels.append(f'Naive\n(a={a} vs 0)')
        positions.append(pos); pos += 1

        all_data.append(_drop_extreme(sub['rel_bias_dre'].values))
        all_colors.append(palette['dre'])
        tick_labels.append(f'DRE-ML\n(a={a} vs 0)')
        positions.append(pos); pos += 1

        if include_drep:
            all_data.append(_drop_extreme(sub['rel_bias_drep'].values))
            all_colors.append(palette['drep'])
            tick_labels.append(f'DRE-Param\n(a={a} vs 0)')
            positions.append(pos); pos += 1

        pos += 1   # gap between arm groups

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel('Relative Bias × 100  [(Est − True) / |True| × 100]', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, pos)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    INCLUDE_DREP = True    # set False to omit DRE-Param from boxplots
    GREYSCALE    = True    # set True for grey shades

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info        = pd.read_csv(info_path)
    k_vals      = sorted(info['k1'].unique())
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []

    for k in k_vals:
        for flavor in flavors:
            print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
            sub     = info[(info['k1'] == k) & (info['flavor_Y'] == flavor)].copy()
            records = build_records(sub)

            if not records:
                print(f'  No data for k={k}, flavor={flavor}, skipping.')
                continue

            df   = pd.DataFrame(records)
            arms = sorted(df['arm'].unique())
            df['flavor_Y'] = flavor
            all_records.append(df)

            fig    = make_figure(df, flavor, arms, include_drep=INCLUDE_DREP,
                                 greyscale=GREYSCALE)
            suffix   = '_bw' if GREYSCALE else ''
            img_path = os.path.join(images_dir, f'_ate_bias_k{k}_{flavor}{suffix}.jpeg')
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Figure saved: _ate_bias_k{k}_{flavor}{suffix}.jpeg')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        summary = (
            all_df.groupby(['k', 'flavor_Y'])
            .agg(
                n_datasets        =('i',              'nunique'),
                rel_bias_naive    =('rel_bias_naive',  'mean'),
                rel_bias_dre      =('rel_bias_dre',    'mean'),
                rel_bias_drep     =('rel_bias_drep',   'mean'),
            )
            .reset_index()
            .rename(columns={'flavor_Y': 'DGP'})
            .round(4)
        )

        tbl_path = os.path.join(tables_dir, '_ate_bias_summary.csv')
        summary.to_csv(tbl_path, index=False)
        print(f'\n✓ Summary table saved: _ate_bias_summary.csv')
        print(summary.to_string(index=False))

    print('\nDone.')
