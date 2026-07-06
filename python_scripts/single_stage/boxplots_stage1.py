# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_stage1.py
# ATE bias boxplots for the single-stage simple DGP.
#
# For each Y flavor, produces:
#   - One table : relative bias for all estimators, one row per dataset i
#                 Saved to: _1trt_effect/1stages/tables/_ate_bias_summary.csv
#   - One figure: Naive | DRE-ML [| DREp(OLS)] [| DREp(EXPO)] boxplots per arm contrast
#                 Saved to: _1trt_effect/1stages/images/_ate_bias_k{k}_{flavor}.jpeg
#
# DREp options (set in __main__):
#   INCLUDE_DREP_OLS  : include DRE-Param (OLS)  — reads {filename}_DREp.csv
#   INCLUDE_DREP_EXPO : include DRE-Param (EXPO)  — reads {filename}_DREp_expo.csv
#   Default: EXPO only
#
# Bias = estimated ATE - true ATE.  Reported as relative bias × 100:
#   [(Est − True) / |True|] × 100
# True ATE computed analytically via ate_single_stage_simple.true_ate_simple.
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

# Greyscale palette
C_BW = {
    'naive':      '0.82',
    'dre':        '0.20',
    'drep_ols':   '0.55',
    'drep_expo':  '0.38',
}

# Color palette
C_COLOR = {
    'naive':      '#E57373',
    'dre':        '#64B5F6',
    'drep_ols':   '#FFB74D',
    'drep_expo':  '#81C784',
}


def _drop_extreme(arr, k=3.0):
    """Remove values beyond k×IQR from Q1/Q3."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


def _ate_from_file(filename, suffix, k):
    """Compute per-arm ATE from mu_hat columns. Returns NaN dict if file missing."""
    path = os.path.join(datasets_dir, f'{filename}{suffix}.csv')
    if not os.path.exists(path):
        return {a: np.nan for a in range(1, k)}
    df = pd.read_csv(path)
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
            naive = _ate_from_file(fname, '_NAIVE',      k)
            dre   = _ate_from_file(fname, '_DRE',        k)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        drep_ols  = _ate_from_file(fname, '_DREp',      k)
        drep_expo = _ate_from_file(fname, '_DREp_expo', k)

        for a in sorted(true['ATE_1'].keys()):
            t        = true['ATE_1'][a]
            n_       = naive.get(a,     np.nan)
            d        = dre.get(a,       np.nan)
            dp_ols   = drep_ols.get(a,  np.nan)
            dp_expo  = drep_expo.get(a, np.nan)

            def _rb(est):
                return (est - t) / abs(t) * 100 if t != 0 else np.nan

            records.append({
                'i':                  i_val,
                'k':                  k,
                'arm':                a,
                'ATE_true':           t,
                'ATE_naive':          n_,
                'ATE_dre':            d,
                'ATE_drep_ols':       dp_ols,
                'ATE_drep_expo':      dp_expo,
                'rel_bias_naive':     _rb(n_),
                'rel_bias_dre':       _rb(d),
                'rel_bias_drep_ols':  _rb(dp_ols),
                'rel_bias_drep_expo': _rb(dp_expo),
            })
    return records


def _draw_arm_boxplot(ax, df, a, include_drep_ols, include_drep_expo, palette):
    """Draw boxplots for one arm contrast onto ax. Returns the axis."""
    all_data    = []
    all_colors  = []
    tick_labels = []
    positions   = []
    pos         = 1

    sub = df[df['arm'] == a]

    all_data.append(_drop_extreme(sub['rel_bias_naive'].values))
    all_colors.append(palette['naive'])
    tick_labels.append('Naive'); positions.append(pos); pos += 1

    all_data.append(_drop_extreme(sub['rel_bias_dre'].values))
    all_colors.append(palette['dre'])
    tick_labels.append('DRE-ML'); positions.append(pos); pos += 1

    if include_drep_ols:
        all_data.append(_drop_extreme(sub['rel_bias_drep_ols'].values))
        all_colors.append(palette['drep_ols'])
        tick_labels.append('DREp(OLS)'); positions.append(pos); pos += 1

    if include_drep_expo:
        all_data.append(_drop_extreme(sub['rel_bias_drep_expo'].values))
        all_colors.append(palette['drep_expo'])
        tick_labels.append('DREp(EXPO)'); positions.append(pos); pos += 1

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_title(f'a={a} vs 0', fontsize=10)
    ax.set_ylabel('Rel. Bias × 100', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, pos)


def make_figure(df, flavor, arms,
                include_drep_ols=False, include_drep_expo=True,
                greyscale=False):
    """
    One figure per (k, flavor).
    k=5 (4 arm contrasts): 2×2 subplot grid, one panel per contrast.
    Otherwise          : single panel, all contrasts side by side.
    Order within each panel: Naive | DRE-ML [| DREp(OLS)] [| DREp(EXPO)]
    """
    title_flavor = 'loggamma' if flavor == 'gamma' else flavor
    palette      = C_BW if greyscale else C_COLOR
    n_boxes      = 2 + int(include_drep_ols) + int(include_drep_expo)
    n_arms       = len(arms)

    # ── 2×2 layout for k=5 (4 contrasts) ────────────────────────────────────
    if n_arms == 4:
        fig, axes = plt.subplots(2, 2, figsize=(n_boxes * 3, 10))
        fig.suptitle(f'ATE Relative Bias — Single-Stage DGP  ({title_flavor})',
                     fontsize=12)
        for idx, a in enumerate(arms):
            ax = axes[idx // 2][idx % 2]
            _draw_arm_boxplot(ax, df, a, include_drep_ols, include_drep_expo, palette)
        plt.tight_layout()
        return fig

    # ── Single-panel layout for k=2 (1 contrast) and k=3 (2 contrasts) ──────
    fig_w = max(5, n_arms * n_boxes * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.set_title(f'ATE Relative Bias — Single-Stage DGP  ({title_flavor})', fontsize=12)

    all_data    = []
    all_colors  = []
    tick_labels = []
    positions   = []
    pos         = 1

    for i_arm, a in enumerate(arms):
        sub = df[df['arm'] == a]

        all_data.append(_drop_extreme(sub['rel_bias_naive'].values))
        all_colors.append(palette['naive'])
        tick_labels.append(f'Naive\n(a={a} vs 0)'); positions.append(pos); pos += 1

        all_data.append(_drop_extreme(sub['rel_bias_dre'].values))
        all_colors.append(palette['dre'])
        tick_labels.append(f'DRE-ML\n(a={a} vs 0)'); positions.append(pos); pos += 1

        if include_drep_ols:
            all_data.append(_drop_extreme(sub['rel_bias_drep_ols'].values))
            all_colors.append(palette['drep_ols'])
            tick_labels.append(f'DREp(OLS)\n(a={a} vs 0)'); positions.append(pos); pos += 1

        if include_drep_expo:
            all_data.append(_drop_extreme(sub['rel_bias_drep_expo'].values))
            all_colors.append(palette['drep_expo'])
            tick_labels.append(f'DREp(EXPO)\n(a={a} vs 0)'); positions.append(pos); pos += 1

        if i_arm < n_arms - 1:
            pos += 1   # gap between arm groups, not after the last one

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel('Relative Bias × 100  [(Est − True) / |True| × 100]', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    INCLUDE_DREP_OLS  = True   # include DRE-Param (OLS)
    INCLUDE_DREP_EXPO = True    # include DRE-Param (EXPO)  ← default
    GREYSCALE         = True
    K_FILTER          = None    # set to 2, 3, or 5; None = all

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info    = pd.read_csv(info_path)
    k_vals  = sorted(info['k1'].unique())
    flavors = sorted(info['flavor_Y'].unique())
    if K_FILTER is not None:
        k_vals = [k for k in k_vals if k == K_FILTER]

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

            fig = make_figure(df, flavor, arms,
                              include_drep_ols=INCLUDE_DREP_OLS,
                              include_drep_expo=INCLUDE_DREP_EXPO,
                              greyscale=GREYSCALE)
            suffix   = '_bw' if GREYSCALE else ''
            img_path = os.path.join(images_dir, f'_ate_bias_k{k}_{flavor}{suffix}.jpeg')
            img_path = os.path.join(images_dir, f'boxplot_k{k}_{flavor}_relbias{suffix}.jpeg')
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Figure saved: _ate_bias_k{k}_{flavor}{suffix}.jpeg')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        agg_cols = {
            'n_datasets':           ('i',                   'nunique'),
            'rel_bias_naive':       ('rel_bias_naive',       'mean'),
            'rel_bias_dre':         ('rel_bias_dre',         'mean'),
            'rel_bias_drep_ols':    ('rel_bias_drep_ols',    'mean'),
            'rel_bias_drep_expo':   ('rel_bias_drep_expo',   'mean'),
        }
        summary = (
            all_df.groupby(['k', 'flavor_Y'])
            .agg(**agg_cols)
            .reset_index()
            .rename(columns={'flavor_Y': 'DGP'})
            .round(4)
        )

        tbl_path = os.path.join(tables_dir, '_ate_bias_summary.csv')
        summary.to_csv(tbl_path, index=False)
        print(f'\n✓ Summary table saved: _ate_bias_summary.csv')
        print(summary.to_string(index=False))

    print('\nDone.')
