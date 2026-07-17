# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# boxplots_deltas_single_stage.py
# delta_1 and Delta_1 relative bias boxplots for the single-stage DGP.
#
# For each (k, Y flavor), produces one figure with one subplot per arm (a vs 0).
# Each subplot shows 3 estimator groups × 2 boxes:
#   Group 1: DRE-ML      — solid box: delta_1  |  hatched box: Delta_1
#   Group 2: DRE-Param(expo) — idem
#   Group 3: DRE-Param(ols)  — idem
#
# Layout: k=2 → single subplot  |  k=3 → 1×2  |  k=5 → 2×2 grid
#
# Bias = (estimated - true) / |true| × 100  (relative %)
# True delta_1 and Delta_1 come from _info_single.csv columns 'delta' and 'Delta'
# (stored as Python list strings; parsed with ast.literal_eval).
#
# Reads:
#   {filename}_deltas_DRE.csv
#   {filename}_deltas_DREp_expo.csv
#   {filename}_deltas_DREp_ols.csv
#
# Saved to:
#   Images: _1trt_effect/1stage/images/_deltas_bias_k{k}_{flavor}[_bw].jpeg
#   Table:  _1trt_effect/1stage/tables/_deltas_bias_summary.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import ast
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/tables')
images_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/images')
info_path    = os.path.join(datasets_dir, '_info_single.csv')

# Color palette per estimator (mirrors boxplots_single_stage.py)
C_DRE       = '#64B5F6'   # blue
C_DREP_EXPO = '#FF8A65'   # orange
C_DREP_OLS  = '#FFB74D'   # yellow

# Greyscale shades
C_BW = {'dre': '0.20', 'drep_expo': '0.42', 'drep_ols': '0.60'}


def _drop_extreme(arr, k=3):
    """Remove values beyond k×IQR from Q1/Q3 (extreme outliers)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]


def build_records(sub_info):
    """
    Loop over rows in sub_info; return list of delta/Delta relative bias records.
    """
    records = []
    for _, row in sub_info.iterrows():
        fname   = row['filename']
        k       = int(row['k'])
        i_val   = int(row['i'])

        # True parameter lists: one value per arm a > 0
        true_delta = ast.literal_eval(str(row['delta']))
        true_Delta = ast.literal_eval(str(row['Delta']))

        dre_path  = os.path.join(datasets_dir, f'{fname}_deltas_DRE.csv')
        expo_path = os.path.join(datasets_dir, f'{fname}_deltas_DREp_expo.csv')
        ols_path  = os.path.join(datasets_dir, f'{fname}_deltas_DREp_ols.csv')

        if not os.path.exists(dre_path):
            print(f'  Skipping {fname}: {dre_path} not found')
            continue
        df_dre  = pd.read_csv(dre_path)
        df_expo = pd.read_csv(expo_path) if os.path.exists(expo_path) else None
        df_ols  = pd.read_csv(ols_path)  if os.path.exists(ols_path)  else None

        def _rb(est, true):
            return (est - true) / abs(true) * 100 if true != 0 else np.nan

        for idx, a in enumerate(range(1, k)):
            t_d = true_delta[idx]
            t_D = true_Delta[idx]

            row_dre = df_dre[df_dre['arm'] == a].iloc[0]
            rec = {
                'i':            i_val,
                'k':            k,
                'arm':          a,
                'true_delta_1': t_d,
                'true_Delta_1': t_D,
                'rb_dre_delta': _rb(row_dre['delta_1_hat'], t_d),
                'rb_dre_Delta': _rb(row_dre['Delta_1_hat'], t_D),
            }

            if df_expo is not None:
                row_expo = df_expo[df_expo['arm'] == a].iloc[0]
                rec['rb_expo_delta'] = _rb(row_expo['delta_1_hat'], t_d)
                rec['rb_expo_Delta'] = _rb(row_expo['Delta_1_hat'], t_D)

            if df_ols is not None:
                row_ols = df_ols[df_ols['arm'] == a].iloc[0]
                rec['rb_ols_delta'] = _rb(row_ols['delta_1_hat'], t_d)
                rec['rb_ols_Delta'] = _rb(row_ols['Delta_1_hat'], t_D)

            records.append(rec)
    return records


def _draw_arm_subplot(ax, sub, greyscale=False):
    """
    Draw 3 estimator groups × 2 boxes (delta_1 solid, Delta_1 hatched) on ax.
    Returns the positions list (used for xlim).
    """
    estimators = [
        ('DRE-ML',    'rb_dre_delta',  'rb_dre_Delta',  'dre'),
        ('DREp-expo', 'rb_expo_delta', 'rb_expo_Delta', 'drep_expo'),
        ('DREp-ols',  'rb_ols_delta',  'rb_ols_Delta',  'drep_ols'),
    ]

    C_MAP = {'dre': C_DRE, 'drep_expo': C_DREP_EXPO, 'drep_ols': C_DREP_OLS}

    all_data   = []
    all_colors = []
    all_hatch  = []
    positions  = []
    tick_pos   = []
    tick_lbl   = []
    pos        = 1

    for est_lbl, col_d, col_D, bw_key in estimators:
        if col_d not in sub.columns:
            continue
        c = C_BW[bw_key] if greyscale else C_MAP[bw_key]

        # delta_1 box (solid)
        all_data.append(_drop_extreme(sub[col_d].values))
        all_colors.append(c)
        all_hatch.append('')
        positions.append(pos)
        tick_pos.append(pos + 0.5)
        tick_lbl.append(est_lbl)
        pos += 1

        # Delta_1 box (hatched)
        all_data.append(_drop_extreme(sub[col_D].values))
        all_colors.append(c)
        all_hatch.append('///')
        positions.append(pos)
        pos += 2   # gap after each pair

    if not all_data:
        return positions

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color, hatch in zip(bp['boxes'], all_colors, all_hatch):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_hatch(hatch)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=10)
    ax.set_ylabel('Relative Bias (%)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    return positions


def make_figure(df, flavor, arms, greyscale=False):
    """
    One subplot per arm; layout depends on n_arms.
      n_arms == 1 → single subplot
      n_arms == 2 → 1×2 side-by-side
      n_arms  > 2 → 2×2 grid
    """
    n_arms       = len(arms)
    grid2x2      = n_arms > 2
    title_flavor = 'loggamma' if flavor == 'gamma' else flavor

    if grid2x2:
        fig, ax_grid = plt.subplots(2, 2, figsize=(14, 9))
        axes = ax_grid.flatten()
    elif n_arms == 2:
        fig, ax_arr = plt.subplots(1, 2, figsize=(12, 5))
        axes = list(ax_arr)
    else:
        fig, ax_single = plt.subplots(figsize=(7, 5))
        axes = [ax_single]

    fig.suptitle(
        fr'Relative Bias Single-Stage DGP  ({title_flavor})',
        fontsize=12)

    for i_arm, a in enumerate(arms):
        ax  = axes[i_arm]
        sub = df[df['arm'] == a]
        _draw_arm_subplot(ax, sub, greyscale=greyscale)
        ax.set_title(f'A={a} vs 0', fontsize=11)

    # Hide unused axes in 2×2 grid
    if grid2x2:
        for j in range(n_arms, 4):
            axes[j].set_visible(False)

    # Legend: solid = delta_1, hatched = Delta_1
    legend_patches = [
        mpatches.Patch(facecolor='grey', hatch='',    alpha=0.8, label=r'$\boldsymbol{\delta}_1$'),
        mpatches.Patch(facecolor='grey', hatch='///', alpha=0.8, label=r'$\boldsymbol{\delta}_2$'),
    ]
    fig.legend(handles=legend_patches, loc='upper left', fontsize=9,
               title='', ncol=1)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    K_FILTER      = 2    # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None    # set to 'expo', 'gamma', etc.; None = all
    GREYSCALE     = True

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    _bw_suffix = '_bw' if GREYSCALE else ''

    info        = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    k_vals      = sorted(info['k'].unique())
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []

    for k in k_vals:
        for flavor in flavors:
            print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
            sub     = info[(info['k'] == k) & (info['flavor_Y'] == flavor)].copy()
            records = build_records(sub)

            if not records:
                print(f'  No data for k={k}, flavor={flavor}, skipping.')
                continue

            df      = pd.DataFrame(records)
            arms    = sorted(df['arm'].unique())
            df['flavor_Y'] = flavor
            all_records.append(df)

            fig = make_figure(df, flavor, arms, greyscale=GREYSCALE)
            img_name = f'_deltas_bias_k{k}_{flavor}{_bw_suffix}.jpeg'
            img_path = os.path.join(images_dir, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Figure saved: {img_name}')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        # Build summary aggregation dynamically
        agg_dict = {
            'n_datasets':    ('i',             'nunique'),
            'rb_dre_delta':  ('rb_dre_delta',  'mean'),
            'rb_dre_Delta':  ('rb_dre_Delta',  'mean'),
        }
        for col in ('rb_expo_delta', 'rb_expo_Delta', 'rb_ols_delta', 'rb_ols_Delta'):
            if col in all_df.columns:
                agg_dict[col] = (col, 'mean')

        summary = (
            all_df.groupby(['k', 'flavor_Y', 'arm'])
            .agg(**agg_dict)
            .reset_index()
            .rename(columns={'flavor_Y': 'DGP'})
            .round(4)
        )

        tbl_path = os.path.join(tables_dir, '_deltas_bias_summary.csv')
        summary.to_csv(tbl_path, index=False)
        print(f'\n✓ Summary table saved: _deltas_bias_summary.csv')
        print(summary.to_string(index=False))

    print('\nDone.')
