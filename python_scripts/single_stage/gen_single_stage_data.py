# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gen_single_stage_data.py
# Generate and save single-stage DTR datasets.
#
# Mirrors gen_two_stage_data.py / gen_three_stage_data.py for the single-stage case.
# Dataset column structure: X1_1 ... X1_p1 | A1 | Y
# Saves to: _1trt_effect/1stages/datasets/s1_k{k}_simple_{flavor}_{i}.csv
# Info file: _1trt_effect/1stages/datasets/_info_simple.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from YAX_funs import gen_X, gen_A, gen_Y

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stages/datasets')
images_dir   = os.path.join(script_dir, '../_1trt_effect/1stages/images')


def _make_params(p1, k, seed):
    """Generate single-stage simulation parameters."""
    rng = np.random.default_rng(seed)
    return {
        'beta_A1': rng.uniform(-0.5, 0.5, size=(p1 + 1, k - 1)),
        'beta_Y1': rng.uniform(-1.0, 1.0, size=(p1 + 1,)),
        'delta1':  np.array([0.6, 0.4, 0.75, 0.17])[:k - 1],
        'Delta1':  np.array([-1.2, -1.0, -1.0,  0.8])[:k - 1],
    }


def gen_single_stage_data(n, p1, k, flavor_Y, i=0, seed=None):
    """
    Generate a single-stage DTR dataset and save to disk.

    Parameters
    ----------
    n        : int   Sample size
    p1       : int   Number of baseline covariates
    k        : int   Number of treatment levels
    flavor_Y : str   Y distribution: 'expo', 'lognormal', 'gamma', 'sigmoid'
    i        : int   Dataset index (used in filename and seed)
    seed     : int   Random seed (default: 1810 + i)
    """
    if seed is None:
        seed = 1810 + i

    params   = _make_params(p1=p1, k=k, seed=seed)
    beta_A1  = params['beta_A1']
    beta_Y1  = params['beta_Y1']
    delta1   = params['delta1']
    Delta1   = params['Delta1']

    print(f"\nGenerating (single-stage): n={n}, p1={p1}, k={k}, flavor_Y={flavor_Y}, i={i}, seed={seed}")
    print(f"  delta1={delta1}   Delta1={Delta1}")

    # ── Generate X, A, Y ─────────────────────────────────────────────────────
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    # Rename columns to X1_1 ... X1_p1 (consistent with two/three-stage scripts)
    X1.columns = [f'X1_{j+1}' for j in range(p1)]

    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A='logit', k=k)
    print(f"  A1 distribution: {np.bincount(A1)}  proportions: {np.bincount(A1)/n}")

    Y_result = gen_Y(delta=delta1, X=X1, A=A1, beta_Y=beta_Y1,
                     Delta=Delta1, flavor_Y=flavor_Y)
    Y = Y_result['Y']
    print(f"  Y summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")

    print("  Y mean by treatment arm:")
    for a in range(k):
        mask = (A1 == a)
        if mask.sum() > 0:
            print(f"    A1={a}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

    # ── Assemble dataset ──────────────────────────────────────────────────────
    dat = pd.concat([
        X1,
        pd.Series(A1, name='A1'),
        pd.Series(Y,  name='Y'),
    ], axis=1)

    filename = f"s1_k{k}_simple_{flavor_Y}_{i}"
    print(f"  Dataset shape: {dat.shape}  columns: {list(dat.columns)}")

    # ── Histogram ─────────────────────────────────────────────────────────────
    os.makedirs(images_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(Y, bins=30, alpha=0.7)
    ax.set_title(f'Y  (s=1, k={k}, {flavor_Y}, i={i})')
    ax.set_xlabel('Y')
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, f'{filename}.jpeg'))
    plt.close(fig)

    # ── Save dataset ──────────────────────────────────────────────────────────
    os.makedirs(datasets_dir, exist_ok=True)
    dat_path = os.path.join(datasets_dir, f'{filename}.csv')
    dat.to_csv(dat_path, index=False)

    # ── Update _info_simple.csv ───────────────────────────────────────────────
    info_path = os.path.join(datasets_dir, '_info_simple.csv')
    row = pd.DataFrame([{
        'i':        i,
        's':        1,
        'n':        n,
        'p1':       p1,
        'k1':       k,
        'flavor_Y': flavor_Y,
        'seed':     seed,
        'filename': filename,
        'delta1':   str(delta1.tolist()),
        'Delta1':   str(Delta1.tolist()),
    }])
    write_header = not os.path.exists(info_path)
    row.to_csv(info_path, mode='a', header=write_header, index=False)

    print(f"  ✓ Saved: {filename}.csv")


# ── Also generate new_i0 dataset (large holdout with same DGP as i=0) ────────
def gen_new_i0(n_new, p1, k, flavor_Y, seed_new=9999):
    """
    Generate a large new_i0 evaluation dataset using the same DGP parameters
    as i=0 (seed=1810) but a fresh random draw of size n_new.

    Saves to: datasets/new_i0/s1_k{k}_simple_{flavor_Y}_new_i0.csv
    """
    new_i0_dir = os.path.join(datasets_dir, 'new_i0')
    os.makedirs(new_i0_dir, exist_ok=True)

    # Use same DGP params as i=0 (seed=1810)
    params  = _make_params(p1=p1, k=k, seed=1810)
    beta_A1 = params['beta_A1']
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']

    print(f"\nGenerating new_i0: n={n_new}, p1={p1}, k={k}, flavor_Y={flavor_Y}, seed={seed_new}")

    rng_state = np.random.get_state()
    np.random.seed(seed_new)

    X1 = gen_X(n=n_new, p=p1, rho=0.5, p_bin=1)
    X1.columns = [f'X1_{j+1}' for j in range(p1)]

    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A='logit', k=k)
    Y_result = gen_Y(delta=delta1, X=X1, A=A1, beta_Y=beta_Y1,
                     Delta=Delta1, flavor_Y=flavor_Y)
    Y = Y_result['Y']

    np.random.set_state(rng_state)

    dat = pd.concat([
        X1,
        pd.Series(A1, name='A1'),
        pd.Series(Y,  name='Y'),
    ], axis=1)

    print(f"  Y summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")

    fname    = f"s1_k{k}_simple_{flavor_Y}_new_i0"
    out_path = os.path.join(new_i0_dir, f'{fname}.csv')
    dat.to_csv(out_path, index=False)
    print(f"  ✓ Saved: {fname}.csv")


# ════════════════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Delete old _info_simple.csv so it's rebuilt from scratch
    info_path = os.path.join(datasets_dir, '_info_simple.csv')
    if os.path.exists(info_path):
        os.remove(info_path)

    N_DATASETS = 1   # number of replications per (k, flavor) combination
    N_NEW_I0   = 400  # size of new_i0 evaluation datasets (match two-stage k=2)

    for k in [2, 3, 5]:
        if k == 2:  p1 = 3
        if k == 3:  p1 = 8
        if k == 5:  p1 = 12
        n = k * 200   # same rule as two/three-stage scripts

        for flavor_Y in ['expo', 'gamma']: #'lognormal', 'sigmoid' 
            # ── Training/simulation datasets (i = 0 ... N_DATASETS-1) ──────
            for i in range(N_DATASETS):
                gen_single_stage_data(n=n, p1=p1, k=k, flavor_Y=flavor_Y, i=i)

            # ── new_i0 evaluation dataset ────────────────────────────────────
            gen_new_i0(n_new=N_NEW_I0, p1=p1, k=k, flavor_Y=flavor_Y)

    print('\nDone.')
