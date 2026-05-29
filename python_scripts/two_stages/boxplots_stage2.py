# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_stage2.py
# ATE bias boxplots for the two-stage simple DGP.
#
# For each Y flavor, produces:
#   - One table: bias_naive and bias_DRE at stages 1 and 2, one row per dataset i
#     Saved to: _1trt_effect/2stages/tables/_ate_bias_{flavor}.csv
#   - One figure: 1x2 subplot — stage 1 (light colors) | stage 2 (dark colors)
#     Each subplot has two boxplots side by side: Naive (red) and DRE-ML (blue)
#     Saved to: _1trt_effect/2stages/images/_ate_bias_{flavor}.jpeg
#
# Bias = estimated ATE - true ATE   (positive = overestimate)
# True ATE is computed analytically via ate_two_stage_simple.true_ate_simple.
# Naive ATE and DRE ATE are computed from mu_hat columns in _NAIVE.csv / _DRE.csv.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/2stages/tables')
images_dir   = os.path.join(script_dir, '../_1trt_effect/2stages/images')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')

sys.path.insert(0, script_dir)
from ate_two_stage_simple import true_ate_simple

C_DREP = '#FFB74D'   # yellow — matches OLS color in boxplots.py

# Greyscale palette — same shades across all stages, matching boxplots.py (single_stage)
C_BW = {'naive': '0.82', 'dre': '0.20', 'drep': '0.42'}


def _drop_extreme(arr, k=3.0):
    """Remove values beyond k×IQR from Q1/Q3 (extreme outliers, k=3 by default)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


# Colors — match boxplots.py shades; darken for later stages
C = {
    1: {'naive': '#E57373', 'dre': '#64B5F6'},   # stage 1: light red / light blue
    2: {'naive': '#B71C1C', 'dre': '#0D47A1'},   # stage 2: dark red  / dark blue
}


def _ate_from_file(filename, suffix, k1, k2):
    """Compute per-arm ATE from mu_hat columns in a _DRE or _NAIVE csv."""
    path = os.path.join(datasets_dir, f'{filename}{suffix}.csv')
    df   = pd.read_csv(path)
    ATE_1 = {a: float(np.mean(df[f'mu_hat_1_a{a}'] - df['mu_hat_1_a0']))
             for a in range(1, k1)}
    ATE_2 = {a: float(np.mean(df[f'mu_hat_2_a{a}'] - df['mu_hat_2_a0']))
             for a in range(1, k2)}
    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


def build_records(sub_info):
    """Loop over rows in sub_info DataFrame; return list of bias records."""
    records = []
    for _, row in sub_info.iterrows():
        fname  = row['filename']
        k1, k2 = int(row['k1']), int(row['k2'])
        i_val  = int(row['i'])
        k_val  = k1   # k1 == k2 for simple DGP
        try:
            true  = true_ate_simple(fname, verbose=False)
            naive = _ate_from_file(fname, '_NAIVE', k1, k2)
            dre   = _ate_from_file(fname, '_DRE',   k1, k2)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        drep = _ate_from_file(fname, '_DREp', k1, k2)

        for a in sorted(true['ATE_1'].keys()):
            t1  = true['ATE_1'][a]
            n1  = naive['ATE_1'].get(a, np.nan)
            d1  = dre['ATE_1'].get(a, np.nan)
            dp1 = drep['ATE_1'].get(a, np.nan)
            t2  = true['ATE_2'].get(a, np.nan)
            n2  = naive['ATE_2'].get(a, np.nan)
            d2  = dre['ATE_2'].get(a, np.nan)
            dp2 = drep['ATE_2'].get(a, np.nan)
            records.append({
                'i':                i_val,
                'k':                k_val,
                'arm':              a,
                'ATE_true_1':       t1,
                'ATE_naive_1':      n1,
                'ATE_dre_1':        d1,
                'ATE_drep_1':       dp1,
                'rel_bias_naive_1': (n1  - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                'rel_bias_dre_1':   (d1  - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                'rel_bias_drep_1':  (dp1 - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                'ATE_true_2':       t2,
                'ATE_naive_2':      n2,
                'ATE_dre_2':        d2,
                'ATE_drep_2':       dp2,
                'rel_bias_naive_2': (n2  - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
                'rel_bias_dre_2':   (d2  - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
                'rel_bias_drep_2':  (dp2 - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
            })
    return records


def make_figure(df, flavor, arms, include_drep=True, greyscale=False):
    """
    k=2 : 1×2 figure (stages side by side).
    k>2 : 2×1 figure (stages stacked vertically).
    Each subplot: Naive | DRE-ML [| DRE-Param] boxplots.
    Set include_drep=False to omit the DRE-Param boxplot.
    Set greyscale=True for grey shades (same across stages): Naive lightest, DRE-ML darkest.
    """
    n_arms    = len(arms)
    n_boxes   = 3 if include_drep else 2
    subplot_w = max(5, n_arms * n_boxes * 1.5)
    vertical  = n_arms > 1

    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=(subplot_w, 2 * 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(2 * subplot_w, 5))
    title_flavor = 'loggamma' if flavor == 'gamma' else flavor
    fig.suptitle(f'ATE Bias — Two-Stage Simple DGP  ({title_flavor})', fontsize=12)

    for col_idx, stage in enumerate([1, 2]):
        ax = axes[col_idx]
        bias_naive_col = f'rel_bias_naive_{stage}'
        bias_dre_col   = f'rel_bias_dre_{stage}'
        bias_drep_col  = f'rel_bias_drep_{stage}'

        all_data    = []
        all_colors  = []
        tick_labels = []
        positions   = []
        pos         = 1

        for a in arms:
            sub = df[df['arm'] == a]
            all_data.append(_drop_extreme(sub[bias_naive_col].values))
            all_colors.append(C_BW['naive'] if greyscale else C[stage]['naive'])
            tick_labels.append(f'Naive\n(A{stage}={a} vs 0)')
            positions.append(pos)
            pos += 1

            all_data.append(_drop_extreme(sub[bias_dre_col].values))
            all_colors.append(C_BW['dre'] if greyscale else C[stage]['dre'])
            tick_labels.append(f'DRE-ML\n(A{stage}={a} vs 0)')
            positions.append(pos)
            pos += 1

            if include_drep:
                all_data.append(_drop_extreme(sub[bias_drep_col].values))
                all_colors.append(C_BW['drep'] if greyscale else C_DREP)
                tick_labels.append(f'DRE-Param\n(A{stage}={a} vs 0)')
                positions.append(pos)
                pos += 1
            pos += 1   # gap between arm groups

        bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(f'Stage {stage}', fontsize=11)
        ax.set_ylabel('Relative Bias × 100  [(Est − True) / |True| × 100]', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, pos)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    INCLUDE_DREP = True    # set to False to omit DRE-Param from boxplots
    GREYSCALE    = True   # set to True for grey shades (Naive lightest, DRE-ML darkest)

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info        = pd.read_csv(info_path)
    k_vals      = sorted(info['k1'].unique())
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []   # accumulate across (k, flavor) for summary table

    for k in k_vals:
        for flavor in flavors:
            print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
            sub = info[(info['k1'] == k) & (info['flavor_Y'] == flavor)].copy()
            records = build_records(sub)

            if not records:
                print(f'  No data for k={k}, flavor={flavor}, skipping.')
                continue

            df   = pd.DataFrame(records)
            arms = sorted(df['arm'].unique())

            # Tag each record then accumulate
            df['flavor_Y'] = flavor
            all_records.append(df)

            # --- Figure ---
            fig = make_figure(df, flavor, arms, include_drep=INCLUDE_DREP,
                              greyscale=GREYSCALE)
            suffix   = '_bw' if GREYSCALE else ''
            img_path = os.path.join(images_dir, f'_ate_bias_k{k}_{flavor}{suffix}.jpeg')
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Figure saved: _ate_bias_k{k}_{flavor}.jpeg')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        summary = (
            all_df.groupby(['k', 'flavor_Y'])
            .agg(
                n_datasets             =('i', 'nunique'),
                rel_bias_naive_stage1  =('rel_bias_naive_1', 'mean'),
                rel_bias_dre_stage1    =('rel_bias_dre_1',   'mean'),
                rel_bias_drep_stage1   =('rel_bias_drep_1',  'mean'),
                rel_bias_naive_stage2  =('rel_bias_naive_2', 'mean'),
                rel_bias_dre_stage2    =('rel_bias_dre_2',   'mean'),
                rel_bias_drep_stage2   =('rel_bias_drep_2',  'mean'),
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
