# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_single_stage.py
# ATE bias boxplots for the single-stage DGP.
#
# For each (k, Y flavor), produces:
#   - One figure: single subplot with boxplots per arm group
#     Boxes per group: Naive | DRE-ML [| DRE-Param(expo)] [| DRE-Param(ols)]
#     Saved to: _1trt_effect/1stage/images/_ate_bias_k{k}_{flavor}_with{VARIANT}_bw.jpeg
#   - One summary table:
#     Saved to: _1trt_effect/1stage/tables/_ate_bias_summary_with{VARIANT}.csv
#
# Bias = estimated ATE - true ATE   (positive = overestimate)
# True ATE is computed analytically via ate_single_stage.true_ate_single.
# Naive ATE and DRE ATE are from mu_hat columns in _NAIVE.csv / _DRE.csv.
# DRE-Param reads from _DREp_expo.csv (EXPO), _DREp_ols.csv (OLS), or both (BOTH).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/tables')
images_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/images')
info_path    = os.path.join(datasets_dir, '_info_single.csv')

from ate_single_stage import true_ate_single

# Color palette (mirrors boxplots_stage2.py)
C_DREP_EXPO = '#FF8A65'   # orange  — DRE-Param (expo)
C_DREP_OLS  = '#FFB74D'   # yellow  — DRE-Param (OLS)
C_NAIVE     = '#E57373'   # red     — Naive
C_DRE       = '#64B5F6'   # blue    — DRE-ML

# Greyscale: Naive lightest, DRE-ML darkest, expo/ols in between
C_BW = {'naive': '0.82', 'dre': '0.20', 'drep_expo': '0.42', 'drep_ols': '0.60'}


def _drop_extreme(arr, k=3):
    """Remove values beyond k×IQR from Q1/Q3 (extreme outliers)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


def _ate_from_file(filename, suffix, k):
    """Compute per-arm ATE from mu_hat columns in a _DRE or _NAIVE csv."""
    path = os.path.join(datasets_dir, f'{filename}{suffix}.csv')
    df   = pd.read_csv(path)
    ATE  = {a: float(np.mean(df[f'mu_hat_a{a}'] - df['mu_hat_a0']))
            for a in range(1, k)}
    return {'ATE': ATE}


def build_records(sub_info, drep_variant='EXPO'):
    """
    Loop over rows in sub_info; return list of bias records.

    drep_variant : 'EXPO' reads _DREp_expo.csv
                   'OLS'  reads _DREp_ols.csv
                   'BOTH' reads both
    """
    drep_variant = drep_variant.upper()
    load_expo = drep_variant in ('EXPO', 'BOTH')
    load_ols  = drep_variant in ('OLS',  'BOTH')

    records = []
    for _, row in sub_info.iterrows():
        fname = row['filename']
        k     = int(row['k'])
        i_val = int(row['i'])

        try:
            true  = true_ate_single(fname, verbose=False)
            naive = _ate_from_file(fname, '_NAIVE', k)
            dre   = _ate_from_file(fname, '_DRE',   k)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        drep_expo = None
        drep_ols  = None
        if load_expo:
            try:
                drep_expo = _ate_from_file(fname, '_DREp_expo', k)
            except FileNotFoundError as exc:
                print(f'  Warning — expo file missing for {fname}: {exc}')
        if load_ols:
            try:
                drep_ols = _ate_from_file(fname, '_DREp_ols', k)
            except FileNotFoundError as exc:
                print(f'  Warning — OLS file missing for {fname}: {exc}')

        for a in sorted(true['ATE'].keys()):
            t  = true['ATE'][a]
            n_ = naive['ATE'].get(a, np.nan)
            d  = dre['ATE'].get(a, np.nan)

            rec = {
                'i':              i_val,
                'k':              k,
                'arm':            a,
                'ATE_true':       t,
                'ATE_naive':      n_,
                'ATE_dre':        d,
                'rel_bias_naive': (n_ - t) / abs(t) * 100 if t != 0 else np.nan,
                'rel_bias_dre':   (d  - t) / abs(t) * 100 if t != 0 else np.nan,
            }

            if drep_expo is not None:
                dpe = drep_expo['ATE'].get(a, np.nan)
                rec.update({
                    'ATE_drep_expo':       dpe,
                    'rel_bias_drep_expo':  (dpe - t) / abs(t) * 100 if t != 0 else np.nan,
                })

            if drep_ols is not None:
                dpo = drep_ols['ATE'].get(a, np.nan)
                rec.update({
                    'ATE_drep_ols':       dpo,
                    'rel_bias_drep_ols':  (dpo - t) / abs(t) * 100 if t != 0 else np.nan,
                })

            records.append(rec)
    return records


def make_figure(df, flavor, arms, drep_variant='EXPO', greyscale=False):
    """
    Single subplot — one boxplot group per arm.

    Boxes per arm group: Naive | DRE-ML [| DRE-Param(expo)] [| DRE-Param(ols)]

    drep_variant : 'EXPO'  — include DRE-Param(expo) only
                   'OLS'   — include DRE-Param(ols) only
                   'BOTH'  — include both DRE-Param variants
                   None    — omit DRE-Param entirely
    greyscale    : bool    Use greyscale palette if True
    """
    drep_variant = drep_variant.upper() if drep_variant else None
    show_expo = drep_variant in ('EXPO', 'BOTH')
    show_ols  = drep_variant in ('OLS',  'BOTH')

    n_arms    = len(arms)
    n_boxes   = 2 + int(show_expo) + int(show_ols)
    fig_w     = max(5, n_arms * n_boxes * 1.5)

    title_flavor = 'loggamma' if flavor == 'gamma' else flavor
    fig, ax   = plt.subplots(figsize=(fig_w, 5))
    fig.suptitle(f'ATE Bias — Single-Stage DGP  ({title_flavor})', fontsize=12)

    all_data    = []
    all_colors  = []
    tick_labels = []
    positions   = []
    pos         = 1

    for i_arm, a in enumerate(arms):
        sub = df[df['arm'] == a]

        # Naive
        all_data.append(_drop_extreme(sub['rel_bias_naive'].values, k=1))
        all_colors.append(C_BW['naive'] if greyscale else C_NAIVE)
        tick_labels.append(f'Naive\n(A={a} vs 0)')
        positions.append(pos); pos += 1

        # DRE-ML
        all_data.append(_drop_extreme(sub['rel_bias_dre'].values))
        all_colors.append(C_BW['dre'] if greyscale else C_DRE)
        tick_labels.append('DRE-ML')
        positions.append(pos); pos += 1

        # DRE-Param (expo)
        if show_expo and 'rel_bias_drep_expo' in sub.columns:
            all_data.append(_drop_extreme(sub['rel_bias_drep_expo'].values))
            all_colors.append(C_BW['drep_expo'] if greyscale else C_DREP_EXPO)
            tick_labels.append('DRE-(expo)')
            positions.append(pos); pos += 1

        # DRE-Param (ols)
        if show_ols and 'rel_bias_drep_ols' in sub.columns:
            all_data.append(_drop_extreme(sub['rel_bias_drep_ols'].values))
            all_colors.append(C_BW['drep_ols'] if greyscale else C_DREP_OLS)
            tick_labels.append('DRE-(ols)')
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
    ax.set_title('Single Stage', fontsize=11)
    ax.set_ylabel('Relative Bias (%)', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # DREP_VARIANT : 'EXPO' (default) — DRE-Param with GLM expo link
    #                'OLS'            — DRE-Param with OLS
    #                'BOTH'           — include both DRE-Param variants
    #                None             — omit DRE-Param entirely
    DREP_VARIANT = 'BOTH'
    GREYSCALE    = True

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    _variant_tag = {
        'EXPO': 'withEXPO',
        'OLS':  'withOLS',
        'BOTH': 'withBOTH',
        None:   'noParam',
    }.get(DREP_VARIANT.upper() if DREP_VARIANT else None, 'withEXPO')
    _bw_suffix = '_bw' if GREYSCALE else ''

    info        = pd.read_csv(info_path)
    k_vals      = sorted(info['k'].unique())
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []

    for k in k_vals:
        for flavor in flavors:
            print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
            sub     = info[(info['k'] == k) & (info['flavor_Y'] == flavor)].copy()
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
            'n_datasets':     ('i',              'nunique'),
            'rel_bias_naive': ('rel_bias_naive',  'mean'),
            'rel_bias_dre':   ('rel_bias_dre',    'mean'),
        }
        for col, label in [
            ('rel_bias_drep_expo', 'rel_bias_drep_expo'),
            ('rel_bias_drep_ols',  'rel_bias_drep_ols'),
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
