# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vplot_summary.py
# Summary V(d*) figure: 2×3 panel comparing DRE-ML vs DREp across k values and DGP.
#
# Layout (one figure per stage):
#
#   Top row    — Y_flavor = expo  : DRE-ML | DREp-expo | Obs Y
#   Bottom row — Y_flavor = gamma : DRE-ML | DREp-ols  | Obs Y
#
#   Columns: k=2 (left) | k=3 (middle) | k=5 (right)
#
# Stars on the DREp bar indicate paired t-test significance vs DRE-ML:
#   *** p < 0.01 | ** p < 0.05 | * p < 0.10
# DRE-ML and Obs Y bars carry no stars.
#
# Control in __main__:
#   STAGES    : list of stages to run, e.g. [1], [2], [3], or [1, 2, 3]
#   GREYSCALE : bool
#
# Output (per stage):
#   _1trt_effect/{n}stage(s)/images/_vplot_summary[_bw].jpeg
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Stage configuration ───────────────────────────────────────────────────────
STAGE_CFG = {
    1: {
        'datasets_dir': os.path.join(script_dir, '_1trt_effect/1stage/datasets'),
        'images_dir':   os.path.join(script_dir, '_1trt_effect/1stage/images'),
        'info_file':    '_info_single.csv',
        'k_col':        'k',
        'final_stage':  1,
        'label':        'Single Stage',
    },
    2: {
        'datasets_dir': os.path.join(script_dir, '_1trt_effect/2stages/datasets'),
        'images_dir':   os.path.join(script_dir, '_1trt_effect/2stages/images'),
        'info_file':    '_info_simple.csv',
        'k_col':        'k1',
        'final_stage':  2,
        'label':        'Two Stage',
    },
    3: {
        'datasets_dir': os.path.join(script_dir, '_1trt_effect/3stages/datasets'),
        'images_dir':   os.path.join(script_dir, '_1trt_effect/3stages/images'),
        'info_file':    '_info_simple.csv',
        'k_col':        'k1',
        'final_stage':  3,
        'label':        'Three Stage',
    },
}

K_VALS = [2, 3, 5]

C_BW = {
    'DRE-ML':    '0.20',
    'DREp-expo': '0.42',
    'DREp-ols':  '0.60',
    'Obs Y':     '0.82',
}
C_COLOR = {
    'DRE-ML':    '#64B5F6',
    'DREp-expo': '#FF8A65',
    'DREp-ols':  '#81C784',
    'Obs Y':     '#E57373',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stars(pval):
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.10: return '*'
    return ''


def _compute_v(csv_path, k, final_stage):
    """Return V = mean(Y_hat[i, d_star[i]]); np.nan if file missing or columns absent."""
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    n  = len(df)
    if final_stage == 1:
        d_col = 'd_star'
        cols  = [f'Y_hat_a{a}' for a in range(k)]
    else:
        d_col = f'd_star_{final_stage}'
        cols  = [f'Y_hat_{final_stage}_a{a}' for a in range(k)]
    if d_col not in df.columns or not all(c in df.columns for c in cols):
        return np.nan
    d     = df[d_col].values.astype(int)
    Y_hat = df[cols].values
    return float(np.mean(Y_hat[np.arange(n), d]))


def collect_v_group(info, eval_dir, k_val, flavor, k_col, final_stage):
    """
    Collect V(d*) arrays for one (k, flavor) group across all replications i.

    Returns dict with numpy arrays: v_dre, v_drep_expo, v_drep_ols, v_obs.
    """
    sub = info[(info[k_col] == k_val) & (info['flavor_Y'] == flavor)]
    lists = {'v_dre': [], 'v_drep_expo': [], 'v_drep_ols': [], 'v_obs': []}

    for _, row in sub.iterrows():
        fname    = row['filename']
        dre_path  = os.path.join(eval_dir, f'{fname}_eval_DRE.csv')
        expo_path = os.path.join(eval_dir, f'{fname}_eval_DREp_expo.csv')
        ols_path  = os.path.join(eval_dir, f'{fname}_eval_DREp_ols.csv')
        obs_path  = os.path.join(eval_dir, f'{fname}_eval.csv')

        lists['v_dre'].append(_compute_v(dre_path,  k_val, final_stage))
        lists['v_drep_expo'].append(_compute_v(expo_path, k_val, final_stage))
        lists['v_drep_ols'].append(_compute_v(ols_path,  k_val, final_stage))
        lists['v_obs'].append(
            float(pd.read_csv(obs_path)['Y'].mean()) if os.path.exists(obs_path) else np.nan
        )

    return {key: np.array(val) for key, val in lists.items()}


def draw_panel(ax, v_dre, v_drep, v_obs, drep_label, title, greyscale=False):
    """
    Draw a 3-bar V(d*) panel on ax: DRE-ML | DREp (expo or ols) | Obs Y.
    Stars on the DREp bar only (paired t-test vs DRE-ML).
    """
    palette = C_BW if greyscale else C_COLOR

    v_dre  = v_dre[~np.isnan(v_dre)]
    v_drep = v_drep[~np.isnan(v_drep)]
    v_obs  = v_obs[~np.isnan(v_obs)]

    pval = np.nan
    if len(v_dre) > 0 and len(v_drep) == len(v_dre):
        _, pval = stats.ttest_rel(v_dre, v_drep)

    entries = []
    if len(v_dre)  > 0: entries.append(('DRE-ML',   v_dre,  np.nan))
    if len(v_drep) > 0: entries.append((drep_label, v_drep, pval))
    if len(v_obs)  > 0: entries.append(('Obs Y',    v_obs,  np.nan))

    if not entries:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10)
        return

    labels = [e[0] for e in entries]
    means  = [float(np.mean(e[1])) for e in entries]
    pvals  = [e[2] for e in entries]
    colors = [palette[lbl] for lbl in labels]

    xs = list(range(len(entries)))
    ax.bar(xs, means, color=colors, alpha=0.85, width=0.55,
           edgecolor='grey', linewidth=0.6)

    y_range = max(means) - min(means) if len(means) > 1 else abs(means[0])
    offset  = y_range * 0.03 + abs(max(means)) * 0.01

    for x, mean, pv in zip(xs, means, pvals):
        stars = '' if np.isnan(pv) else _stars(pv)
        ax.text(x, mean + offset, f'{mean:.3f}{stars}',
                ha='center', va='bottom', fontsize=10)

    ax.set_title(title, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean V(d*)', fontsize=8)
    ax.set_ylim(
        bottom=min(means) * 0.95 if min(means) > 0 else min(means) * 1.05,
        top=max(means) + y_range * 0.20 + abs(max(means)) * 0.05,
    )
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)


# ── Main figure builder ───────────────────────────────────────────────────────

def make_summary_figure(stage, greyscale=False):
    """
    Build 2×3 summary V(d*) figure for one stage.

    Top row    — expo  flavor: DRE-ML | DREp-expo | Obs Y
    Bottom row — gamma flavor: DRE-ML | DREp-ols  | Obs Y
    Columns: k=2, k=3, k=5
    """
    cfg          = STAGE_CFG[stage]
    datasets_dir = cfg['datasets_dir']
    eval_dir     = os.path.join(datasets_dir, 'eval_sets')
    info_path    = os.path.join(datasets_dir, cfg['info_file'])
    k_col        = cfg['k_col']
    final_st     = cfg['final_stage']

    if not os.path.exists(info_path):
        print(f'  Info file not found for stage {stage}: {info_path}')
        return None

    info = pd.read_csv(info_path)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f'V(d*) Summary — {cfg["label"]}', fontsize=15)

    row_cfg = [
        ('expo',  'DREp-expo', 'v_drep_expo'),
        ('gamma', 'DREp-ols',  'v_drep_ols'),
    ]

    for row_idx, (flavor, drep_label, drep_key) in enumerate(row_cfg):
        flavor_title = 'log-gamma' if flavor == 'gamma' else flavor
        for col_idx, k_val in enumerate(K_VALS):
            ax = axes[row_idx, col_idx]

            # Check this (k, flavor) combination exists in info
            has_data = (k_col in info.columns and
                        flavor in info['flavor_Y'].values and
                        ((info[k_col] == k_val) & (info['flavor_Y'] == flavor)).any())

            if not has_data:
                ax.set_visible(False)
                continue

            print(f'  stage={stage}  k={k_val}  flavor={flavor}  drep={drep_label}')
            data  = collect_v_group(info, eval_dir, k_val, flavor, k_col, final_st)
            title = f'|A|={k_val}  ({flavor_title})'
            draw_panel(ax, data['v_dre'], data[drep_key], data['v_obs'],
                       drep_label=drep_label, title=title, greyscale=greyscale)

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    STAGES    = [1, 2, 3]   # e.g. [1], [2], [3], or [1, 2, 3]
    GREYSCALE = True

    _bw_suffix = '_bw' if GREYSCALE else ''

    for stage in STAGES:
        print(f'\n{"="*55}\nStage {stage}\n{"="*55}')
        fig = make_summary_figure(stage, greyscale=GREYSCALE)
        if fig is not None:
            img_name = f'_vplot_summary_stage{stage}{_bw_suffix}.jpeg'
            img_path = os.path.join(script_dir, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {img_path}')

    print('\nDone.')
