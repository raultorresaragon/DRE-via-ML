# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gen_new_i0_two_stage.py
# Generate one "new_i0" dataset per (k, flavor) for the two-stage simple DGP.
#
# Uses the SAME DGP parameters as i=0 (same make_sim_params seed = 1810)
# but a different random draw for the data (controlled by DATA_SEED).
#
# Saved to: _1trt_effect/2stages/datasets/new_i0/s2_k{k}_simple_{flavor}_new_i0.csv
# Info file: _1trt_effect/2stages/datasets/new_i0/_info_simple_new_i0.csv
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from YAX_funs import gen_X, gen_A, gen_A2_simple, gen_X2, gen_Y_simple
from sim_params import make_sim_params

datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
new_i0_dir   = os.path.join(datasets_dir, 'new_i0')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
info_out     = os.path.join(new_i0_dir,   '_info_simple_new_i0.csv')


def gen_new_i0(k, flavor_Y, seed_params, n, p1, data_seed):
    """
    Generate a two-stage simple dataset with the same DGP params as seed_params
    but a fresh random draw seeded by data_seed.

    Parameters
    ----------
    k           : int   Number of treatment levels (k1 = k2 = k)
    flavor_Y    : str   Y distribution flavor
    seed_params : int   Seed passed to make_sim_params (same as i=0 seed)
    n           : int   Sample size
    p1          : int   Number of stage-1 covariates
    data_seed   : int   Seed for the actual data draws (different from params seed)
    """
    p2     = p1 + 1
    params = make_sim_params(p1=p1, p2=p2, k1=k, k2=k, seed=seed_params)

    beta_A1 = params['beta_A1']
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']
    delta2  = params['delta2']
    Delta2  = params['Delta2']

    # Separate seed for data draws — make_sim_params uses default_rng which does
    # NOT affect the legacy np.random state, so this seeding is clean.
    np.random.seed(data_seed)

    # Stage 1
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k)

    # Intermediate outcome Y1
    X2_temp  = gen_X2(X1=X1, A1=A1, p2=1, delta1=delta1, beta_Y1=beta_Y1,
                      flavor_X2=flavor_Y, p_bin=0, Delta1=Delta1)
    Y1_vals  = X2_temp['Y_1'].values

    # Stage 2 covariates: X2 = X1 (time-invariant baseline)
    X2_simple = pd.DataFrame({'Y_1': Y1_vals})
    for j, col in enumerate(X1.columns):
        X2_simple[f'X2_{j+1}'] = X1[col].values

    A2 = gen_A2_simple(A1=A1, Y1_obs=Y1_vals, k2=k)

    # Final outcome
    Y_result = gen_Y_simple(X1=X1, A1=A1, A2=A2, beta_Y1=beta_Y1,
                            delta1=delta1, Delta1=Delta1,
                            delta2_scalar=float(delta2[0]),
                            Delta2_scalar=float(Delta2[0]),
                            flavor_Y=flavor_Y)
    Y = Y_result['Y']

    dat = pd.concat([
        X1,
        pd.Series(A1,      name='A1'),
        X2_simple,
        pd.Series(A2,      name='A2'),
        pd.Series(Y,       name='Y'),
    ], axis=1)

    filename = f"s2_k{k}_simple_{flavor_Y}_new_i0"
    dat_path = os.path.join(new_i0_dir, f'{filename}.csv')
    dat.to_csv(dat_path, index=False)

    row = pd.DataFrame([{
        'k1':       k, 'k2': k,
        's':        2, 'n':  n, 'p1': p1, 'p2': p2,
        'flavor_Y': flavor_Y,
        'seed_params': seed_params,
        'data_seed':   data_seed,
        'filename':    filename,
    }])
    write_header = not os.path.exists(info_out)
    row.to_csv(info_out, mode='a', header=write_header, index=False)

    print(f"  n={n}, A1={np.bincount(A1)}, A2={np.bincount(A2)}, "
          f"Y mean={Y.mean():.3f}")
    print(f"✓ Saved: {filename}.csv")
    return dat


if __name__ == '__main__':
    DATA_SEED = 1810   # change to get a different new_i0 draw
    K_FILTER  = None   # set to 2, 3, or 5 to generate only that k; None = all
    I_SOURCE  = 5      # which i to source DGP parameters from

    os.makedirs(new_i0_dir, exist_ok=True)
    if os.path.exists(info_out):
        os.remove(info_out)

    info = pd.read_csv(info_path)
    i0   = info[info['i'] == I_SOURCE].copy()
    if K_FILTER is not None:
        i0 = i0[i0['k1'] == K_FILTER]

    for _, row in i0.iterrows():
        k       = int(row['k1'])
        flavor  = row['flavor_Y']
        seed_p  = int(row['seed'])
        n       = int(row['n'])
        p1      = int(row['p1'])
        print(f'\n{"="*55}\nk={k}  flavor={flavor}  params_seed={seed_p}\n{"="*55}')
        gen_new_i0(k=k, flavor_Y=flavor, seed_params=seed_p,
                   n=n, p1=p1, data_seed=DATA_SEED)

    print('\nDone.')
