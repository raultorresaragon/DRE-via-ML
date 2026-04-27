# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dstar_results_stage3.py
# Summary of estimated OTR decisions pooled across all 30 datasets, per Y flavor.
#
# For each flavor outputs:
#   1. Frequency table (CSV):
#        d1 marginal      (2 rows  for k=2)
#        d1×d2 joint      (4 rows  for k=2)
#        d1×d2×d3 joint   (8 rows  for k=2)   — 14 rows total
#        columns: combination | true_OTR | DRE_ML | naive
#      Saved to: _1trt_effect/3stages/tables/_dstar_freq_{flavor}.csv
#
#   2. Confusion matrix figure (3×2 grid): rows = stage, cols = model (DRE-ML | naive)
#        rows = optimal d*, cols = estimated d* — pooled over all 30 datasets
#      Saved to: _1trt_effect/3stages/images/_dstar_confmat_{flavor}.jpeg
#
#   3. Value bar chart: pooled V(d*) for true OTR, DRE-ML, naive
#        V computed via oracle Q3 from the OTR file
#      Saved to: _1trt_effect/3stages/images/_dstar_value_{flavor}.jpeg
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
tables_dir   = os.path.join(script_dir, '../_1trt_effect/3stages/tables')
images_dir   = os.path.join(script_dir, '../_1trt_effect/3stages/images')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')


def load_flavor(sub_info):
    """
    Load and pool OTR/DRE/NAIVE files for one flavor.

    Returns
    -------
    df       : pooled DataFrame with d1/d2/d3 columns + Q3_a* columns
    k1,k2,k3 : treatment arm counts
    """
    frames = []
    k1 = k2 = k3 = None

    for _, row in sub_info.iterrows():
        fname       = row['filename']
        k1, k2, k3 = int(row['k1']), int(row['k2']), int(row['k3'])
        try:
            otr   = pd.read_csv(os.path.join(datasets_dir, f'{fname}_OTR.csv'))
            dre   = pd.read_csv(os.path.join(datasets_dir, f'{fname}_DRE.csv'))
            naive = pd.read_csv(os.path.join(datasets_dir, f'{fname}_NAIVE.csv'))
        except FileNotFoundError as exc:
            print(f'  Skipping {fname}: {exc}')
            continue

        try:
            drep = pd.read_csv(os.path.join(datasets_dir, f'{fname}_DREp.csv'))
            d1_drep = drep['d_star_1'].values
            d2_drep = drep['d_star_2'].values
            d3_drep = drep['d_star_3'].values
        except FileNotFoundError:
            d1_drep = np.full(len(otr), np.nan)
            d2_drep = np.full(len(otr), np.nan)
            d3_drep = np.full(len(otr), np.nan)

        chunk = pd.DataFrame({
            'd1_true':  otr['d1_star'].values,
            'd2_true':  otr['d2_star'].values,
            'd3_true':  otr['d3_star'].values,
            'd1_dre':   dre['d_star_1'].values,
            'd2_dre':   dre['d_star_2'].values,
            'd3_dre':   dre['d_star_3'].values,
            'd1_drep':  d1_drep,
            'd2_drep':  d2_drep,
            'd3_drep':  d3_drep,
            'd1_naive': naive['d_star_1'].values,
            'd2_naive': naive['d_star_2'].values,
            'd3_naive': naive['d_star_3'].values,
        })
        for a in range(k3):
            chunk[f'Q3_a{a}'] = otr[f'Q3_a{a}'].values
        frames.append(chunk)

    if not frames:
        return None, k1, k2, k3
    return pd.concat(frames, ignore_index=True), k1, k2, k3


def make_freq_table(df, k1, k2, k3):
    """
    Frequency table:
      d1 marginal     (k1 rows)
      d1×d2 joint     (k1*k2 rows)
      d1×d2×d3 joint  (k1*k2*k3 rows)
    Columns: combination | true_OTR | DRE_ML | naive
    """
    col = '{d1*, d2*, d3*}'
    rows = []

    # d1 marginal
    for a1 in range(k1):
        rows.append({
            col:         f'{{{a1}}}',
            'true_OTR':  (df['d1_true']  == a1).mean(),
            'DRE_ML':    (df['d1_dre']   == a1).mean(),
            'DRE_Param': (df['d1_drep']  == a1).mean(),
            'naive':     (df['d1_naive'] == a1).mean(),
        })

    # d1×d2 joint
    for a1 in range(k1):
        for a2 in range(k2):
            rows.append({
                col:         f'{{{a1},{a2}}}',
                'true_OTR':  ((df['d1_true']  == a1) & (df['d2_true']  == a2)).mean(),
                'DRE_ML':    ((df['d1_dre']   == a1) & (df['d2_dre']   == a2)).mean(),
                'DRE_Param': ((df['d1_drep']  == a1) & (df['d2_drep']  == a2)).mean(),
                'naive':     ((df['d1_naive'] == a1) & (df['d2_naive'] == a2)).mean(),
            })

    # d1×d2×d3 joint
    for a1 in range(k1):
        for a2 in range(k2):
            for a3 in range(k3):
                rows.append({
                    col: f'{{{a1},{a2},{a3}}}',
                    'true_OTR': (
                        (df['d1_true']  == a1) &
                        (df['d2_true']  == a2) &
                        (df['d3_true']  == a3)
                    ).mean(),
                    'DRE_ML': (
                        (df['d1_dre']   == a1) &
                        (df['d2_dre']   == a2) &
                        (df['d3_dre']   == a3)
                    ).mean(),
                    'DRE_Param': (
                        (df['d1_drep']  == a1) &
                        (df['d2_drep']  == a2) &
                        (df['d3_drep']  == a3)
                    ).mean(),
                    'naive': (
                        (df['d1_naive'] == a1) &
                        (df['d2_naive'] == a2) &
                        (df['d3_naive'] == a3)
                    ).mean(),
                })

    return pd.DataFrame(rows).round(4)


def make_value_dict(df, k3):
    """Pooled V(d*) using oracle Q3 for each OTR type."""
    n  = len(df)
    Q3 = df[[f'Q3_a{a}' for a in range(k3)]].values

    def _v(d_col):
        d = df[d_col].values.astype(int)
        return float(np.mean(Q3[np.arange(n), d]))

    out = {
        'True OTR': _v('d3_true'),
        'DRE-ML':   _v('d3_dre'),
    }
    if df['d3_drep'].notna().any():
        out['DRE-Param'] = _v('d3_drep')
    out['naive'] = _v('d3_naive')
    return out


def _cm_annot(cm):
    total = cm.sum()
    annot = np.empty(cm.shape, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot[r, c] = f'{cm[r, c]}\n({cm[r, c] / total * 100:.1f}%)'
    return annot


def plot_confmats(df, k1, k2, k3, flavor):
    """
    3×2 figure: rows = stage (1, 2, 3), cols = model (DRE-ML, naive).
    Rows in each heatmap = optimal d*, cols = estimated d*.
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        f'Confusion Matrices — Three-Stage, pooled 30 datasets  ({flavor})',
        fontsize=11
    )

    stages = [
        (1, k1, 'd1_true', 'd1_dre', 'd1_naive'),
        (2, k2, 'd2_true', 'd2_dre', 'd2_naive'),
        (3, k3, 'd3_true', 'd3_dre', 'd3_naive'),
    ]

    for row_idx, (stage, k, true_col, dre_col, naive_col) in enumerate(stages):
        for col_idx, (model_label, est_col) in enumerate([
            ('DRE-ML', dre_col),
            ('naive',  naive_col),
        ]):
            ax  = axes[row_idx, col_idx]
            cm  = confusion_matrix(df[true_col], df[est_col], labels=list(range(k)))
            acc = np.diag(cm).sum() / cm.sum()

            sns.heatmap(
                cm, annot=_cm_annot(cm), fmt='', cmap='Greys',
                xticklabels=[f'est a={a}' for a in range(k)],
                yticklabels=[f'opt a={a}' for a in range(k)],
                ax=ax
            )
            ax.set_title(f'[{model_label}]  Stage {stage}   acc={acc:.1%}')
            ax.set_xlabel(f'Estimated  d{stage}')
            ax.set_ylabel(f'Optimal  d{stage}')

    plt.tight_layout()
    path = os.path.join(images_dir, f'_dstar_confmat_{flavor}.jpeg')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Confusion matrix figure saved: _dstar_confmat_{flavor}.jpeg')


def plot_value(vals, flavor):
    """Bar chart: pooled V(d*) for True OTR, DRE-ML, DRE-Param, naive."""
    labels = list(vals.keys())
    values = list(vals.values())
    palette = {
        'True OTR':  'steelblue',
        'DRE-ML':    '#64B5F6',
        'DRE-Param': '#FFB74D',
        'naive':     '#E57373',
    }
    colors = [palette.get(lbl, 'gray') for lbl in labels]
    v_max  = max(values)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 * abs(v_max),
            f'{val:.4f}', ha='center', va='bottom', fontsize=9
        )
    ax.set_title(f'V(d*) by OTR Type — Three-Stage  ({flavor})', fontsize=11)
    ax.set_ylabel('Pooled mean  E[Y]  under policy')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(images_dir, f'_dstar_value_{flavor}.jpeg')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Value figure saved: _dstar_value_{flavor}.jpeg')


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info    = pd.read_csv(info_path)
    flavors = sorted(info['flavor_Y'].unique())

    for flavor in flavors:
        print(f'\n{"="*55}\nFlavor: {flavor}\n{"="*55}')
        sub = info[info['flavor_Y'] == flavor].copy()

        df, k1, k2, k3 = load_flavor(sub)
        if df is None:
            print('  No data, skipping.')
            continue
        print(f'  Pooled {len(df):,} rows across {len(sub)} datasets')

        # 1. Frequency table
        freq      = make_freq_table(df, k1, k2, k3)
        freq_path = os.path.join(tables_dir, f'_dstar_freq_{flavor}.csv')
        freq.to_csv(freq_path, index=False)
        print(f'  Frequency table saved: _dstar_freq_{flavor}.csv')
        print(freq.to_string(index=False))

        # 2. Confusion matrices
        plot_confmats(df, k1, k2, k3, flavor)

        # 3. Value bar chart
        vals = make_value_dict(df, k3)
        plot_value(vals, flavor)
        print(f'  V: true={vals["True OTR"]:.4f}  DRE={vals["DRE-ML"]:.4f}  naive={vals["naive"]:.4f}')

    print('\nDone.')
