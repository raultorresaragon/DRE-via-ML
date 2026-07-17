# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gen_single_stage_data.py
# Generate and save single-stage DTR datasets.
#
# Single-stage DGP:
#   X  ~ correlated multivariate normal (last column binarized)
#   A  ~ multinomial logit(X @ beta_A)          [k treatment levels]
#   Y  ~ f(X @ beta_Y + delta[A-1] + Delta[A-1] * X_bin)  [flavor-dependent link]
#
# Dataset columns: X_1, ..., X_p, A, Y
# Saved to:        _1trt_effect/1stage/datasets/s{s}_k{k}_{flavor_Y}_{i}.csv
# Histogram:       _1trt_effect/1stage/images/s{s}_k{k}_{flavor_Y}_{i}.jpeg
# Info file:       _1trt_effect/1stage/datasets/_info_single.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from YAX_funs import gen_X, gen_A, gen_Y
from sim_params_single import make_sim_params_single, print_param_shapes_single

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
images_dir   = os.path.join(script_dir, '../_1trt_effect/1stage/images')
info_path    = os.path.join(datasets_dir, '_info_single.csv')


def gen_single_stage_data(s, n, p, k, flavor_Y, i=0, seed=None):
    """
    Generate one single-stage DTR dataset and save it to disk.

    Parameters
    ----------
    s        : int          Simulation study number
    n        : int          Sample size
    p        : int          Number of covariates
    k        : int          Number of treatment levels
    flavor_Y : str          DGP flavor for Y: 'expo', 'gamma', 'sigmoid', 'lognormal'
    i        : int          Replication index (default 0)
    seed     : int or None  Random seed; defaults to 1810 + i

    Returns
    -------
    pd.DataFrame   The generated dataset
    """
    if seed is None:
        seed = 1810 + i

    params = make_sim_params_single(p=p, k=k, seed=seed, flavor_Y=flavor_Y)

    print(f"\nGenerating: n={n}, p={p}, k={k}, flavor_Y={flavor_Y}, i={i}, seed={seed}")
    print_param_shapes_single(params, p=p, k=k)
    print(f"main trt effect(s)    on Y: delta={params['delta']}")
    print(f"interaction effect(s) on Y: Delta={params['Delta']}")

    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------
    X        = gen_X(n=n, p=p, rho=0.5, p_bin=1)
    A        = gen_A(X=X, beta_A=params['beta_A'], flavor_A='logit', k=k)
    Y_result = gen_Y(delta=params['delta'], X=X, A=A,
                     beta_Y=params['beta_Y'], Delta=params['Delta'],
                     flavor_Y=flavor_Y)
    Y = Y_result['Y']

    print(f"\nA distribution: {np.bincount(A)}  proportions: {np.bincount(A) / n}")
    print(f"Y summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")

    print("\nY mean by treatment group:")
    for a in range(k):
        mask = (A == a)
        if mask.sum() > 0:
            print(f"  A={a}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

    # --------------------------------------------------------
    # Assemble dataset — columns: X_1...X_p, A, Y
    # --------------------------------------------------------
    dat      = pd.concat([X,
                           pd.Series(A, name='A'),
                           pd.Series(Y, name='Y')], axis=1)
    filename = f's{s}_k{k}_{flavor_Y}_{i}'

    print(f"\nDataset shape: {dat.shape}  columns: {list(dat.columns)}")

    # --------------------------------------------------------
    # Histogram — Y only (single axis)
    # --------------------------------------------------------
    os.makedirs(images_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dat['Y'], bins=30, alpha=0.7)
    ax.set_title(f'Y  (i={i})')
    plt.suptitle(f"s={s}, k={k}, {flavor_Y}, i={i}")
    plt.tight_layout()
    img_path = os.path.join(images_dir, f'{filename}.jpeg')
    fig.savefig(img_path)
    plt.close(fig)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    os.makedirs(datasets_dir, exist_ok=True)
    dat_path = os.path.join(datasets_dir, f'{filename}.csv')
    dat.to_csv(dat_path, index=False)

    # --------------------------------------------------------
    # Update _info_single.csv (append; write header only once)
    # --------------------------------------------------------
    row = pd.DataFrame([{
        'i':        i,
        's':        s,
        'n':        n,
        'p':        p,
        'k':        k,
        'flavor_Y': flavor_Y,
        'seed':     seed,
        'filename': filename,
        'delta':    str([round(float(x), 2) for x in params['delta']]),
        'Delta':    str([round(float(x), 2) for x in params['Delta']]),
    }])
    write_header = not os.path.exists(info_path)
    row.to_csv(info_path, mode='a', header=write_header, index=False)

    print(f"✓ Saved: {filename}")
    return dat


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    info_path_main = os.path.join(script_dir, '../_1trt_effect/1stage/datasets/_info_single.csv')
    if os.path.exists(info_path_main):
        os.remove(info_path_main)

    s = 1
    for k in [2,3,5]: #[2, 3, 5]:
        p = {2: 3, 3: 8, 5: 12}[k]
        n = k * 200
        for fY in ['expo', 'gamma']: #, 'sigmoid', 'lognormal']:
            for i in range(30):
                gen_single_stage_data(s=s, n=n, p=p, k=k, flavor_Y=fY, i=i)

    print('\nDone.')
