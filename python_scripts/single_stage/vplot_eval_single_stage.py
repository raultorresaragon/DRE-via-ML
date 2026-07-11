# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vplot_eval_single_stage.py
# Collect V(d*) values across all replications and compare estimators.
#
# For each row i in _info_single.csv:
#   V(DRE-ML)    = mean(Y_hat[j, d_star[j]])  from {filename}_eval_DRE.csv
#   V(DREp-expo) = mean(Y_hat[j, d_star[j]])  from {filename}_eval_DREp_expo.csv
#   V(DREp-ols)  = mean(Y_hat[j, d_star[j]])  from {filename}_eval_DREp_ols.csv
#   Obs Y        = mean(Y)                      from {filename}_eval.csv  (no t-test)
#
# V(d*) = mean(Y_hat[i, d_star[i]])  — single-stage formula
#
# Paired t-tests (DRE-ML as reference):
#   DRE-ML vs DREp-expo
#   DRE-ML vs DREp-ols
#
# Output (per k × flavor group):
#   Bar plot: _1trt_effect/1stage/images/eval_sets/_vplot_eval_k{k}_{flavor}_bw.jpeg
#     4 bars: DRE-ML | DREp-expo | DREp-ols | Obs Y
#     Each bar top shows mean V (2 d.p.) + significance stars vs DRE-ML:
#       *** p < 0.01 | ** p < 0.05 | * p < 0.10  (DRE-ML and Obs Y show no stars)
#   Tables: _1trt_effect/1stage/tables/eval_sets/_v_summary.csv
#           _1trt_effect/1stage/tables/eval_sets/_v_per_replication.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

script_dir   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')
tables_dir   = os.path.join(script_dir,   '../_1trt_effect/1stage/tables/eval_sets')
images_dir   = os.path.join(script_dir,   '../_1trt_effect/1stage/images/eval_sets')

C_BW = {
    'DRE-ML':    '0.20',
    'DREp-expo': '0.42',
    'DREp-ols':  '0.60',
    'Obs Y':     '0.82',
}
C_COLOR = {
    'DRE-ML':    '#64B5F6',
    'DREp-expo': '#FFB74D',
    'DREp-ols':  '#81C784',
    'Obs Y':     '#E57373',
}


def _compute_v(csv_path, k):
    """Return V = mean(Y_hat[i, d_star[i]]); np.nan if file missing."""
    if not os.path.exists(csv_path):
        return np.nan
    df    = pd.read_csv(csv_path)
    n     = len(df)
    d     = df['d_star'].values.astype(int)
    cols  = [f'Y_hat_a{a}' for a in range(k)]
    Y_hat = df[cols].values
    return float(np.mean(Y_hat[np.arange(n), d]))


def _stars(pval):
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.10: return '*'
    return ''


def collect_v_values(info, eval_dir):
    """
    Loop over all rows in info; collect V per estimator per replication.

    Returns
    -------
    pd.DataFrame with columns: i, k, flavor_Y, V_dre, V_drep_expo, V_drep_ols, V_obs
    """
    records = []
    for _, row in info.iterrows():
        k        = int(row['k'])
        flavor   = row['flavor_Y']
        filename = row['filename']
        i_val    = int(row['i'])

        dre_path       = os.path.join(eval_dir, f'{filename}_eval_DRE.csv')
        drep_expo_path = os.path.join(eval_dir, f'{filename}_eval_DREp_expo.csv')
        drep_ols_path  = os.path.join(eval_dir, f'{filename}_eval_DREp_ols.csv')
        obs_path       = os.path.join(eval_dir, f'{filename}_eval.csv')

        v_dre       = _compute_v(dre_path,       k)
        v_drep_expo = _compute_v(drep_expo_path,  k)
        v_drep_ols  = _compute_v(drep_ols_path,   k)
        v_obs       = (float(pd.read_csv(obs_path)['Y'].mean())
                       if os.path.exists(obs_path) else np.nan)

        if np.isnan(v_dre) and np.isnan(v_drep_expo) and np.isnan(v_drep_ols):
            print(f'  Skipping i={i_val} k={k} {flavor}: no eval predictions found.')
            continue

        records.append({
            'i':           i_val,
            'k':           k,
            'flavor_Y':    flavor,
            'V_dre':       v_dre,
            'V_drep_expo': v_drep_expo,
            'V_drep_ols':  v_drep_ols,
            'V_obs':       v_obs,
        })
    return pd.DataFrame(records)


def make_figure(sub_df, k, flavor, greyscale=False):
    """
    Bar plot of mean V(d*) for one (k, flavor) group — four bars:
      DRE-ML | DREp-expo | DREp-ols | Obs Y

    Each bar top shows: mean V to 2 d.p. + significance stars vs DRE-ML
      *** p < 0.01 | ** p < 0.05 | * p < 0.10
    DRE-ML and Obs Y show no stars (reference and no-test, respectively).
    """
    v_dre       = sub_df['V_dre'].dropna().values
    v_drep_expo = sub_df['V_drep_expo'].dropna().values
    v_drep_ols  = sub_df['V_drep_ols'].dropna().values
    v_obs       = sub_df['V_obs'].dropna().values

    palette      = C_BW if greyscale else C_COLOR
    title_flavor = 'log-gamma' if flavor == 'gamma' else flavor

    # Paired t-test p-values (DRE-ML as reference)
    pval_expo = np.nan
    pval_ols  = np.nan
    if len(v_dre) > 0 and len(v_drep_expo) == len(v_dre):
        _, pval_expo = stats.ttest_rel(v_dre, v_drep_expo)
    if len(v_dre) > 0 and len(v_drep_ols) == len(v_dre):
        _, pval_ols = stats.ttest_rel(v_dre, v_drep_ols)

    entries = [
        ('DRE-ML',    v_dre,       np.nan),
        ('DREp-expo', v_drep_expo, pval_expo),
        ('DREp-ols',  v_drep_ols,  pval_ols),
        ('Obs Y',     v_obs,       np.nan),
    ]
    entries = [(lbl, arr, pv) for lbl, arr, pv in entries if len(arr) > 0]

    if not entries:
        return None

    labels = [e[0] for e in entries]
    means  = [float(np.mean(e[1])) for e in entries]
    pvals  = [e[2] for e in entries]
    colors = [palette[lbl] for lbl in labels]

    fig_w = max(5, len(entries) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    xs   = list(range(len(entries)))
    bars = ax.bar(xs, means, color=colors, alpha=0.85, width=0.55, edgecolor='grey',
                  linewidth=0.6)

    # Annotate bar tops: mean + stars
    y_range = max(means) - min(means) if len(means) > 1 else abs(means[0])
    offset  = y_range * 0.03 + abs(max(means)) * 0.01

    for x, mean, pv in zip(xs, means, pvals):
        stars = '' if np.isnan(pv) else _stars(pv)
        label = f'{mean:.2f}{stars}'
        ax.text(x, mean + offset, label, ha='center', va='bottom', fontsize=9)

    ax.set_title(f'V(d) for single stage  |A|={k}  ({title_flavor})', fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean V(d*)', fontsize=9)
    ax.set_ylim(bottom=min(means) * 0.95 if min(means) > 0 else min(means) * 1.05,
                top=max(means) + y_range * 0.20 + abs(max(means)) * 0.05)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'gamma', etc.; None = all
    GREYSCALE     = True

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    print('\nCollecting V(d*) values...')
    df_all = collect_v_values(info, eval_dir)

    if df_all.empty:
        print('No data found. Run gen_eval_datasets_single_stage.py, '
              'run_eval_dre_single_stage.py, and run_eval_drep_single_stage.py first.')
    else:
        summary_rows = []

        for (k, flavor), sub in df_all.groupby(['k', 'flavor_Y']):
            v_dre       = sub['V_dre'].dropna().values
            v_drep_expo = sub['V_drep_expo'].dropna().values
            v_drep_ols  = sub['V_drep_ols'].dropna().values
            v_obs       = sub['V_obs'].dropna().values
            n_rep       = len(sub)

            row = {
                'k':              k,
                'flavor_Y':       flavor,
                'n_replications': n_rep,
                'mean_V_dre':       float(np.mean(v_dre))       if len(v_dre)       > 0 else np.nan,
                'mean_V_drep_expo': float(np.mean(v_drep_expo)) if len(v_drep_expo) > 0 else np.nan,
                'mean_V_drep_ols':  float(np.mean(v_drep_ols))  if len(v_drep_ols)  > 0 else np.nan,
                'mean_V_obs':       float(np.mean(v_obs))       if len(v_obs)       > 0 else np.nan,
                'sd_V_dre':         float(np.std(v_dre))        if len(v_dre)       > 1 else np.nan,
                'sd_V_drep_expo':   float(np.std(v_drep_expo))  if len(v_drep_expo) > 1 else np.nan,
                'sd_V_drep_ols':    float(np.std(v_drep_ols))   if len(v_drep_ols)  > 1 else np.nan,
                'sd_V_obs':         float(np.std(v_obs))        if len(v_obs)       > 1 else np.nan,
            }

            # t-test: DRE-ML vs DREp-expo
            if len(v_dre) > 0 and len(v_drep_expo) > 0 and len(v_dre) == len(v_drep_expo):
                t, p = stats.ttest_rel(v_dre, v_drep_expo)
                row['tstat_vs_expo'] = round(float(t), 4)
                row['pval_vs_expo']  = round(float(p), 4)
                row['sig_vs_expo']   = _stars(p) or 'n.s.'
            else:
                row['tstat_vs_expo'] = np.nan
                row['pval_vs_expo']  = np.nan
                row['sig_vs_expo']   = ''

            # t-test: DRE-ML vs DREp-ols
            if len(v_dre) > 0 and len(v_drep_ols) > 0 and len(v_dre) == len(v_drep_ols):
                t, p = stats.ttest_rel(v_dre, v_drep_ols)
                row['tstat_vs_ols'] = round(float(t), 4)
                row['pval_vs_ols']  = round(float(p), 4)
                row['sig_vs_ols']   = _stars(p) or 'n.s.'
            else:
                row['tstat_vs_ols'] = np.nan
                row['pval_vs_ols']  = np.nan
                row['sig_vs_ols']   = ''

            print(f'\n  k={k}  {flavor:10s}  '
                  f'DRE-ML={row["mean_V_dre"]:.4f}  '
                  f'DREp-expo={row["mean_V_drep_expo"]:.4f}  '
                  f'DREp-ols={row["mean_V_drep_ols"]:.4f}  '
                  f'Obs Y={row["mean_V_obs"]:.4f}')
            print(f'    DRE-ML vs DREp-expo: t={row["tstat_vs_expo"]}  '
                  f'p={row["pval_vs_expo"]}  {row["sig_vs_expo"]}')
            print(f'    DRE-ML vs DREp-ols:  t={row["tstat_vs_ols"]}  '
                  f'p={row["pval_vs_ols"]}  {row["sig_vs_ols"]}')

            summary_rows.append(row)

            # ---- Figure ----
            fig = make_figure(sub, k, flavor, greyscale=GREYSCALE)
            if fig is not None:
                suffix   = '_bw' if GREYSCALE else ''
                img_path = os.path.join(images_dir, f'_vplot_eval_k{k}_{flavor}{suffix}.jpeg')
                fig.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'  Figure saved: _vplot_eval_k{k}_{flavor}{suffix}.jpeg')

        summary = pd.DataFrame(summary_rows).round(4)
        summary.to_csv(os.path.join(tables_dir, '_v_summary.csv'), index=False)
        print(f'\n✓ Summary table saved: _v_summary.csv')
        print(summary.to_string(index=False))

        df_all[['i', 'k', 'flavor_Y', 'V_dre', 'V_drep_expo',
                'V_drep_ols', 'V_obs']].round(4).to_csv(
            os.path.join(tables_dir, '_v_per_replication.csv'), index=False)
        print(f'✓ Per-replication values saved: _v_per_replication.csv')

    print('\nDone.')
