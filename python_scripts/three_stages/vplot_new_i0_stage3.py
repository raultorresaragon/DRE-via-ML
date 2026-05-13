# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vplot_new_i0_stage3.py
# V(d*) bar chart for the new_i0 evaluation — three-stage simple DGP.
#
# V is the mean predicted final-stage outcome under each policy's recommended d_star:
#   V(DRE-ML)   = mean(Y_hat_3[i, d_star_3[i]])  from  _new_i0_DRE.csv
#   V(DRE-Param)= mean(Y_hat_3[i, d_star_3[i]])  from  _new_i0_DREp.csv   [optional]
#   V(Naive)    = mean(Y_hat_3[i, d_star_3[i]])  from  _new_i0_NAIVE.csv
#   Observed Y  = mean(Y)  from  new_i0 dataset                            [optional]
#
# Output: _1trt_effect/3stages/images/new_i0/_vplot_k{k}_{flavor}.jpeg
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/3stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
images_dir   = os.path.join(script_dir,   '../_1trt_effect/3stages/images/new_i0')

PALETTE = {
    'DRE-ML':    '#64B5F6',
    'DRE-Param': '#FFB74D',
    'Naive':     '#E57373',
    'Obs Y':     '#90A4AE',
}


def _compute_v(csv_path, k, final_stage=3):
    """Load a new_i0 prediction CSV and compute V = mean(Y_hat_final[i, d_star[i]])."""
    df = pd.read_csv(csv_path)
    n  = len(df)
    d  = df[f'd_star_{final_stage}'].values.astype(int)
    cols  = [f'Y_hat_{final_stage}_a{a}' for a in range(k)]
    Y_hat = df[cols].values
    return float(np.mean(Y_hat[np.arange(n), d]))


def make_vplot(k, flavor_Y, include_drep=True, include_naive=True, include_obs_y=True):
    """Bar chart of V(d*) for one (k, flavor) combination."""
    fname_new  = f"s3_k{k}_simple_{flavor_Y}_new_i0"
    vals = {}

    dre_path   = os.path.join(new_i0_dir, f'{fname_new}_DRE.csv')
    drep_path  = os.path.join(new_i0_dir, f'{fname_new}_DREp.csv')
    naive_path = os.path.join(new_i0_dir, f'{fname_new}_NAIVE.csv')
    dat_path   = os.path.join(new_i0_dir, f'{fname_new}.csv')

    if not os.path.exists(dre_path):
        print(f"  Missing {fname_new}_DRE.csv — skipping.")
        return
    vals['DRE-ML'] = _compute_v(dre_path,  k)

    if include_drep and os.path.exists(drep_path):
        vals['DRE-Param'] = _compute_v(drep_path, k)

    if include_naive:
        if not os.path.exists(naive_path):
            print(f"  Missing {fname_new}_NAIVE.csv — skipping Naive bar.")
        else:
            vals['Naive'] = _compute_v(naive_path, k)

    if include_obs_y and os.path.exists(dat_path):
        vals['Obs Y'] = float(pd.read_csv(dat_path)['Y'].mean())

    if not vals:
        print(f"  No data for k={k}, flavor={flavor_Y}.")
        return

    labels = list(vals.keys())
    values = list(vals.values())
    colors = [PALETTE.get(lbl, 'gray') for lbl in labels]
    v_max  = max(values)

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * abs(v_max),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(f'V(d*) — Three-Stage new_i0  k={k}  ({flavor_Y})', fontsize=11)
    ax.set_ylabel('Mean predicted Y under policy')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(images_dir, f'_vplot_k{k}_{flavor_Y}.jpeg')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: _vplot_k{k}_{flavor_Y}.jpeg")
    for lbl, val in vals.items():
        print(f"    {lbl}: {val:.4f}")


if __name__ == '__main__':
    INCLUDE_DREP  = True    # set False to omit DRE-Param bar
    INCLUDE_NAIVE = False    # set False to omit Naive bar
    INCLUDE_OBS_Y = True    # set False to omit observed Y bar
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
                   include_naive=INCLUDE_NAIVE,
                   include_obs_y=INCLUDE_OBS_Y)

    print('\nDone.')
