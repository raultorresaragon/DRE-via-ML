# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_stage2.py
# ATE bias boxplots for the two-stage simple DGP.
#
# For each Y flavor, produces:
#   - One table: bias_naive and bias_DRE at stages 1 and 2, one row per dataset i
#     Saved to: _1trt_effect/2stages/tables/_ate_bias_{flavor}.csv
#   - One figure: 1x2 subplot — stage 1 (light colors) | stage 2 (dark colors)
#     Each subplot has boxplots: Naive | DRE-ML [| DRE-Param(expo)] [| DRE-Param(ols)]
#     Saved to: _1trt_effect/2stages/images/_ate_bias_k{k}_{flavor}_with{VARIANT}{suffix}.jpeg
#
# Bias = estimated ATE - true ATE   (positive = overestimate)
# True ATE is computed analytically via ate_two_stage_simple.true_ate_simple.
# Naive ATE and DRE ATE are computed from mu_hat columns in _NAIVE.csv / _DRE.csv.
# DRE-Param reads from _DREp_expo.csv (EXPO), _DREp.csv (OLS), or both (BOTH).
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

# Color palette
C_DREP_EXPO = '#FF8A65'   # orange  — DRE-Param (expo)
C_DREP_OLS  = '#FFB74D'   # yellow  — DRE-Param (OLS)

# Greyscale: Naive lightest, DRE-ML darkest, expo/ols in between
C_BW = {'naive': '0.82', 'dre': '0.20', 'drep_expo': '0.42', 'drep_ols': '0.60'}

# Colors — stage 1: light, stage 2: dark
C = {
    1: {'naive': '#E57373', 'dre': '#64B5F6'},
    2: {'naive': '#B71C1C', 'dre': '#0D47A1'},
}


def _drop_extreme(arr, k=3):
    """Remove values beyond k×IQR from Q1/Q3 (extreme outliers, k=3 by default)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


def _ate_from_file(filename, suffix, k1, k2):
    """Compute per-arm ATE from mu_hat columns in a _DRE or _NAIVE csv."""
    path = os.path.join(datasets_dir, f'{filename}{suffix}.csv')
    df   = pd.read_csv(path)
    ATE_1 = {a: float(np.mean(df[f'mu_hat_1_a{a}'] - df['mu_hat_1_a0']))
             for a in range(1, k1)}
    ATE_2 = {a: float(np.mean(df[f'mu_hat_2_a{a}'] - df['mu_hat_2_a0']))
             for a in range(1, k2)}
    return {'ATE_1': ATE_1, 'ATE_2': ATE_2}


def build_records(sub_info, drep_variant='EXPO'):
    """
    Loop over rows in sub_info; return list of bias records.

    drep_variant : 'EXPO' reads _DREp_expo.csv
                   'OLS'  reads _DREp.csv
                   'BOTH' reads both
    """
    drep_variant = drep_variant.upper()
    load_expo = drep_variant in ('EXPO', 'BOTH')
    load_ols  = drep_variant in ('OLS',  'BOTH')

    records = []
    for _, row in sub_info.iterrows():
        fname  = row['filename']
        k1, k2 = int(row['k1']), int(row['k2'])
        i_val  = int(row['i'])
        k_val  = k1

        try:
            true  = true_ate_simple(fname, verbose=False)
            naive = _ate_from_file(fname, '_NAIVE', k1, k2)
            dre   = _ate_from_file(fname, '_DRE',   k1, k2)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        drep_expo = None
        drep_ols  = None
        if load_expo:
            try:
                drep_expo = _ate_from_file(fname, '_DREp_expo', k1, k2)
            except FileNotFoundError as exc:
                print(f'  Warning — expo file missing for {fname}: {exc}')
        if load_ols:
            try:
                drep_ols = _ate_from_file(fname, '_DREp_ols', k1, k2)
            except FileNotFoundError as exc:
                print(f'  Warning — OLS file missing for {fname}: {exc}')

        for a in sorted(true['ATE_1'].keys()):
            t1 = true['ATE_1'][a]
            t2 = true['ATE_2'].get(a, np.nan)
            n1 = naive['ATE_1'].get(a, np.nan)
            n2 = naive['ATE_2'].get(a, np.nan)
            d1 = dre['ATE_1'].get(a, np.nan)
            d2 = dre['ATE_2'].get(a, np.nan)

            rec = {
                'i':                i_val,
                'k':                k_val,
                'arm':              a,
                'ATE_true_1':       t1,
                'ATE_naive_1':      n1,
                'ATE_dre_1':        d1,
                'rel_bias_naive_1': (n1 - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                'rel_bias_dre_1':   (d1 - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                'ATE_true_2':       t2,
                'ATE_naive_2':      n2,
                'ATE_dre_2':        d2,
                'rel_bias_naive_2': (n2 - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
                'rel_bias_dre_2':   (d2 - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
            }

            if drep_expo is not None:
                dp1e = drep_expo['ATE_1'].get(a, np.nan)
                dp2e = drep_expo['ATE_2'].get(a, np.nan)
                rec.update({
                    'ATE_drep_expo_1':       dp1e,
                    'ATE_drep_expo_2':       dp2e,
                    'rel_bias_drep_expo_1':  (dp1e - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                    'rel_bias_drep_expo_2':  (dp2e - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
                })

            if drep_ols is not None:
                dp1o = drep_ols['ATE_1'].get(a, np.nan)
                dp2o = drep_ols['ATE_2'].get(a, np.nan)
                rec.update({
                    'ATE_drep_ols_1':       dp1o,
                    'ATE_drep_ols_2':       dp2o,
                    'rel_bias_drep_ols_1':  (dp1o - t1) / abs(t1) * 100 if t1 != 0 else np.nan,
                    'rel_bias_drep_ols_2':  (dp2o - t2) / abs(t2) * 100 if t2 != 0 else np.nan,
                })

            records.append(rec)
    return records


def make_figure(df, flavor, arms, drep_variant='BOTH', greyscale=False):
    """
    k=2 : 1×2 figure (stages side by side).
    k>2 : 2×1 figure (stages stacked vertically).

    Boxes per arm group: Naive | DRE-ML [| DRE-Param(expo)] [| DRE-Param(ols)]

    drep_variant : 'EXPO'  — include DRE-Param(expo) only
                   'OLS'   — include DRE-Param(ols) only
                   'BOTH'  — include both DRE-Param variants
                   None    — omit DRE-Param entirely
    """
    drep_variant = drep_variant.upper() if drep_variant else None
    show_expo = drep_variant in ('EXPO', 'BOTH')
    show_ols  = drep_variant in ('OLS',  'BOTH')

    n_arms    = len(arms)
    n_boxes   = 2 + int(show_expo) + int(show_ols)
    subplot_w = max(5, n_arms * n_boxes * 1.5)
    vertical  = n_arms > 1

    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=(subplot_w, 2 * 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(2 * subplot_w, 5))
    title_flavor = 'loggamma' if flavor == 'gamma' else flavor
    fig.suptitle(f'ATE Rel Bias — Two-Stage DGP  ({title_flavor})', fontsize=12)

    for col_idx, stage in enumerate([1, 2]):
        ax = axes[col_idx]

        all_data    = []
        all_colors  = []
        tick_labels = []
        positions   = []
        pos         = 1

        for i_arm, a in enumerate(arms):
            sub = df[df['arm'] == a]

            # Naive
            all_data.append(_drop_extreme(sub[f'rel_bias_naive_{stage}'].values, k=1))
            all_colors.append(C_BW['naive'] if greyscale else C[stage]['naive'])
            tick_labels.append(f'Naive\n(A{stage}={a} vs 0)')
            positions.append(pos); pos += 1

            # DRE-ML
            all_data.append(_drop_extreme(sub[f'rel_bias_dre_{stage}'].values))
            all_colors.append(C_BW['dre'] if greyscale else C[stage]['dre'])
            tick_labels.append(f'DRE-ML')#\n(A{stage}={a} vs 0)')
            positions.append(pos); pos += 1

            # DRE-Param (expo)
            if show_expo and f'rel_bias_drep_expo_{stage}' in sub.columns:
                all_data.append(_drop_extreme(sub[f'rel_bias_drep_expo_{stage}'].values))
                all_colors.append(C_BW['drep_expo'] if greyscale else C_DREP_EXPO)
                tick_labels.append(f'DRE-(expo)') #\n(A{stage}={a} vs 0)')
                positions.append(pos); pos += 1

            # DRE-Param (ols)
            if show_ols and f'rel_bias_drep_ols_{stage}' in sub.columns:
                all_data.append(_drop_extreme(sub[f'rel_bias_drep_ols_{stage}'].values))
                all_colors.append(C_BW['drep_ols'] if greyscale else C_DREP_OLS)
                tick_labels.append(f'DRE-(ols)')#\n(A{stage}={a} vs 0)')
                positions.append(pos); pos += 1

            if i_arm < len(arms) - 1:
                pos += 1   # gap between arm groups

        bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_title(f'Stage {stage}', fontsize=11)
        ax.set_ylabel('Relative Bias (%)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # DREP_VARIANT : 'EXPO' (default) — DRE-Param with GLM expo link
    #                'OLS'            — DRE-Param with OLS
    #                'BOTH'           — include both DRE-Param variants side by side
    #                None             — omit DRE-Param entirely
    DREP_VARIANT = 'BOTH'
    GREYSCALE    = True

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Image filename suffix based on variant
    _variant_tag = {
        'EXPO': 'withEXPO',
        'OLS':  'withOLS',
        'BOTH': 'withBOTH',
        None:   'noParam',
    }.get(DREP_VARIANT.upper() if DREP_VARIANT else None, 'withEXPO')
    _bw_suffix = '_bw' if GREYSCALE else ''

    info        = pd.read_csv(info_path)
    k_vals      = sorted(info['k1'].unique())
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []

    for k in k_vals:
        for flavor in flavors:
            print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
            sub     = info[(info['k1'] == k) & (info['flavor_Y'] == flavor)].copy()
            records = build_records(sub, drep_variant=DREP_VARIANT)

            if not records:
                print(f'  No data for k={k}, flavor={flavor}, skipping.')
                continue

            df   = pd.DataFrame(records)
            arms = sorted(df['arm'].unique())
            df['flavor_Y'] = flavor
            all_records.append(df)

            fig = make_figure(df, flavor, arms,
                              drep_variant=DREP_VARIANT, greyscale=GREYSCALE)
            img_name = f'_ate_bias_k{k}_{flavor}_{_variant_tag}{_bw_suffix}.jpeg'
            img_path = os.path.join(images_dir, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Figure saved: {img_name}')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        # Build summary aggregation dynamically based on available columns
        agg_dict = {
            'n_datasets':            ('i',                'nunique'),
            'rel_bias_naive_stage1': ('rel_bias_naive_1', 'mean'),
            'rel_bias_dre_stage1':   ('rel_bias_dre_1',   'mean'),
            'rel_bias_naive_stage2': ('rel_bias_naive_2', 'mean'),
            'rel_bias_dre_stage2':   ('rel_bias_dre_2',   'mean'),
        }
        for col, label in [
            ('rel_bias_drep_expo_1', 'rel_bias_drep_expo_stage1'),
            ('rel_bias_drep_expo_2', 'rel_bias_drep_expo_stage2'),
            ('rel_bias_drep_ols_1',  'rel_bias_drep_ols_stage1'),
            ('rel_bias_drep_ols_2',  'rel_bias_drep_ols_stage2'),
        ]:
            if col in all_df.columns:
                agg_dict[label] = (col, 'mean')

        summary = (
            all_df.groupby(['k', 'flavor_Y'])
            .agg(**agg_dict)
            .reset_index()
            .rename(columns={'flavor_Y': 'DGP'})
            .round(4)
        )

        tbl_name = f'_ate_bias_summary_{_variant_tag}.csv'
        tbl_path = os.path.join(tables_dir, tbl_name)
        summary.to_csv(tbl_path, index=False)
        print(f'\n✓ Summary table saved: {tbl_name}')
        print(summary.to_string(index=False))

    print('\nDone.')
