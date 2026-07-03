# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vplot_new_i0_stage1.py
# V(d*) bar chart for the new_i0 evaluation — single-stage simple DGP.
#
# V is the mean predicted outcome under each policy's recommended d_star:
#   V(DRE-ML)   = mean(Y_hat_1[i, d_star_1[i]])  from  _new_i0_DRE.csv
#   V(DRE-Param)= mean(Y_hat_1[i, d_star_1[i]])  from  _new_i0_DREp.csv   [optional]
#   Observed Y  = mean(Y)  from  new_i0 dataset                            [optional]
#
# Output: _1trt_effect/1stages/images/new_i0/_vplot_k{k}_{flavor}.jpeg
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
images_dir   = os.path.join(script_dir,   '../_1trt_effect/1stages/images/new_i0')

PALETTE_COLOR = {
    'DRE-ML':    '#64B5F6',
    'DRE-Param': '#FFB74D',
    'Obs Y':     '#90A4AE',
}
PALETTE_BW = {
    'DRE-ML':    '0.20',
    'DRE-Param': '0.42',
    'Obs Y':     '0.82',
}


def _compute_v_arr(csv_path, k, final_stage=1):
    """Load a new_i0 prediction CSV and return per-subject V_i = Y_hat_1[i, d_star_1[i]]."""
    df = pd.read_csv(csv_path)
    n  = len(df)
    d  = df[f'd_star_{final_stage}'].values.astype(int)
    cols  = [f'Y_hat_{final_stage}_a{a}' for a in range(k)]
    Y_hat = df[cols].values
    return Y_hat[np.arange(n), d]


def _stars(pval):
    if pval < 0.01:  return '***'
    if pval < 0.05:  return '**'
    if pval < 0.1:   return '*'
    return ''


def make_vplot(k, flavor_Y, include_drep=True, include_obs_y=True, greyscale=False):
    """Bar chart of V(d*) for one (k, flavor) combination."""
    fname_new  = f"s1_k{k}_simple_{flavor_Y}_new_i0"
    vals = {}
    arrs = {}

    dre_path  = os.path.join(new_i0_dir, f'{fname_new}_DRE.csv')
    drep_path = os.path.join(new_i0_dir, f'{fname_new}_DREp.csv')
    dat_path  = os.path.join(new_i0_dir, f'{fname_new}.csv')

    if not os.path.exists(dre_path):
        print(f"  Missing {fname_new}_DRE.csv — skipping.")
        return
    arrs['DRE-ML'] = _compute_v_arr(dre_path, k)
    vals['DRE-ML'] = float(arrs['DRE-ML'].mean())

    if include_drep and os.path.exists(drep_path):
        arrs['DRE-Param'] = _compute_v_arr(drep_path, k)
        vals['DRE-Param'] = float(arrs['DRE-Param'].mean())

    if include_obs_y and os.path.exists(dat_path):
        vals['Obs Y'] = float(pd.read_csv(dat_path)['Y'].mean())

    if not vals:
        print(f"  No data for k={k}, flavor={flavor_Y}.")
        return

    # Paired t-test: DRE-ML vs DRE-Param
    dre_stars = ''
    if 'DRE-ML' in arrs and 'DRE-Param' in arrs:
        pval = stats.ttest_rel(arrs['DRE-ML'], arrs['DRE-Param']).pvalue
        dre_stars = _stars(pval)
        print(f"  t-test DRE-ML vs DRE-Param: p={pval:.4f}  {dre_stars or 'n.s.'}")

    labels = list(vals.keys())
    values = list(vals.values())
    palette = PALETTE_BW if greyscale else PALETTE_COLOR
    colors = [palette.get(lbl, 'gray') for lbl in labels]
    v_max  = max(values)
    title_flavor = 'log-gamma' if flavor_Y == 'gamma' else flavor_Y

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.85)
    for bar, lbl, val in zip(bars, labels, values):
        annot = f'{val:.4f}{dre_stars if lbl == "DRE-ML" else ""}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * abs(v_max),
                annot, ha='center', va='bottom', fontsize=9)
    ax.set_title(f'V(d*) by model on new data ({title_flavor})', fontsize=11)
    ax.set_ylabel('Mean predicted Y under policy')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    suffix   = '_bw' if greyscale else ''
    out_path = os.path.join(images_dir, f'_vplot_k{k}_{flavor_Y}{suffix}.jpeg')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: _vplot_k{k}_{flavor_Y}{suffix}.jpeg")
    for lbl, val in vals.items():
        print(f"    {lbl}: {val:.4f}")


if __name__ == '__main__':
    INCLUDE_DREP  = True    # set False to omit DRE-Param bar
    INCLUDE_OBS_Y = True    # set False to omit observed Y bar
    GREYSCALE     = True    # set True for grey shades (DRE-ML darkest, Obs Y lightest)
    K_FILTER      = None    # set to 2, 3, or 5; None = all

    os.makedirs(images_dir, exist_ok=True)

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == 0].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        k      = int(row['k1'])
        flavor = row['flavor_Y']
        print(f'\n{"="*55}\nk={k}  Flavor: {flavor}\n{"="*55}')
        make_vplot(k=k, flavor_Y=flavor,
                   include_drep=INCLUDE_DREP,
                   include_obs_y=INCLUDE_OBS_Y,
                   greyscale=GREYSCALE)

    print('\nDone.')
