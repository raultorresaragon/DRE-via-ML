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
        try:
            true  = true_ate_simple(fname, verbose=False)
            naive = _ate_from_file(fname, '_NAIVE', k1, k2)
            dre   = _ate_from_file(fname, '_DRE',   k1, k2)
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        for a in sorted(true['ATE_1'].keys()):
            t1 = true['ATE_1'][a]
            n1 = naive['ATE_1'].get(a, np.nan)
            d1 = dre['ATE_1'].get(a, np.nan)
            t2 = true['ATE_2'].get(a, np.nan)
            n2 = naive['ATE_2'].get(a, np.nan)
            d2 = dre['ATE_2'].get(a, np.nan)
            records.append({
                'i':            i_val,
                'arm':          a,
                'ATE_true_1':   t1,
                'ATE_naive_1':  n1,
                'ATE_dre_1':    d1,
                'bias_naive_1': n1 - t1,
                'bias_dre_1':   d1 - t1,
                'ATE_true_2':   t2,
                'ATE_naive_2':  n2,
                'ATE_dre_2':    d2,
                'bias_naive_2': n2 - t2,
                'bias_dre_2':   d2 - t2,
            })
    return records


def make_figure(df, flavor, arms):
    """
    1 × 2 figure: left = stage 1 bias, right = stage 2 bias.
    Each subplot: Naive (red) | DRE-ML (blue) boxplots side by side.
    """
    n_arms = len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(8 * n_arms, 5))
    if n_arms == 1:
        axes = [axes[0], axes[1]]   # already a list from subplots(1,2)
    fig.suptitle(f'ATE Bias — Two-Stage Simple DGP  ({flavor})', fontsize=12)

    for col_idx, stage in enumerate([1, 2]):
        ax = axes[col_idx]
        bias_naive_col = f'bias_naive_{stage}'
        bias_dre_col   = f'bias_dre_{stage}'

        all_data   = []
        all_colors = []
        tick_labels = []
        positions   = []
        pos         = 1

        for a in arms:
            sub = df[df['arm'] == a]
            all_data.append(sub[bias_naive_col].dropna().values)
            all_colors.append(C[stage]['naive'])
            tick_labels.append(f'Naive\n(A{stage}={a} vs 0)')
            positions.append(pos)
            pos += 1

            all_data.append(sub[bias_dre_col].dropna().values)
            all_colors.append(C[stage]['dre'])
            tick_labels.append(f'DRE-ML\n(A{stage}={a} vs 0)')
            positions.append(pos)
            pos += 2   # gap between arm groups

        bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(f'Stage {stage}', fontsize=11)
        ax.set_ylabel('Bias  (Estimated − True ATE)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, pos)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info        = pd.read_csv(info_path)
    flavors     = sorted(info['flavor_Y'].unique())
    all_records = []   # accumulate across flavors for summary table

    for flavor in flavors:
        print(f'\n{"="*55}\nFlavor: {flavor}\n{"="*55}')
        sub     = info[info['flavor_Y'] == flavor].copy()
        records = build_records(sub)

        if not records:
            print(f'  No data for flavor={flavor}, skipping.')
            continue

        df   = pd.DataFrame(records)
        arms = sorted(df['arm'].unique())

        # Tag each record with its flavor then accumulate
        df['flavor_Y'] = flavor
        all_records.append(df)

        # --- Figure ---
        fig = make_figure(df, flavor, arms)
        img_path = os.path.join(images_dir, f'_ate_bias_{flavor}.jpeg')
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Figure saved: _ate_bias_{flavor}.jpeg')

    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)

        summary = (
            all_df.groupby('flavor_Y')
            .agg(
                n_datasets        =('i', 'nunique'),
                bias_naive_stage1 =('bias_naive_1', 'mean'),
                bias_dre_stage1   =('bias_dre_1',   'mean'),
                bias_naive_stage2 =('bias_naive_2', 'mean'),
                bias_dre_stage2   =('bias_dre_2',   'mean'),
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
