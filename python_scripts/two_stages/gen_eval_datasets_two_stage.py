# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gen_eval_datasets_two_stage.py
# Generate one evaluation dataset per row of _info_simple.csv for the two-stage DGP.
#
# For each row i, uses the SAME DGP parameters (same make_sim_params seed) as the
# training dataset but a fresh random draw (data_seed = row seed + DATA_SEED_OFFSET).
#
# Saved to: _1trt_effect/2stages/datasets/eval_per_i/{filename}_eval.csv
# Info file: _1trt_effect/2stages/datasets/eval_per_i/_info_eval.csv
#
# Optional filters (set in __main__):
#   K_FILTER      : int or None  — only generate for k = K_FILTER
#   FLAVOR_FILTER : str or None  — only generate for a specific flavor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from YAX_funs   import gen_X, gen_A, gen_A2_simple, gen_X2, gen_Y_simple
from sim_params import make_sim_params

datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_per_i')
info_path    = os.path.join(datasets_dir, '_info_simple.csv')
info_out     = os.path.join(eval_dir,     '_info_eval.csv')

DATA_SEED_OFFSET = 99999   # added to the row's params seed to get the data seed


def gen_eval_dataset(row, data_seed_offset=DATA_SEED_OFFSET):
    """
    Generate an evaluation dataset for one row of _info_simple.csv.

    Uses the same DGP parameters as the training dataset (same make_sim_params seed)
    but draws fresh data using (row seed + data_seed_offset).

    Parameters
    ----------
    row             : pd.Series   One row from _info_simple.csv
    data_seed_offset: int         Added to the row seed to produce the data seed

    Returns
    -------
    pd.DataFrame   The generated evaluation dataset
    """
    k         = int(row['k1'])
    flavor_Y  = row['flavor_Y']
    seed_p    = int(row['seed'])
    n         = int(row['n'])
    p1        = int(row['p1'])
    p2        = p1 + 1
    i_val     = int(row['i'])
    filename  = row['filename']
    data_seed = seed_p + data_seed_offset

    params = make_sim_params(p1=p1, p2=p2, k1=k, k2=k, seed=seed_p, flavor_Y=flavor_Y)

    beta_A1 = params['beta_A1']
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']
    delta2  = params['delta2']
    Delta2  = params['Delta2']

    # make_sim_params uses default_rng — does NOT affect legacy np.random state
    np.random.seed(data_seed)

    # Stage 1
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A='logit', k=k)

    # Intermediate outcome Y1
    X2_temp = gen_X2(X1=X1, A1=A1, p2=1, delta1=delta1, beta_Y1=beta_Y1,
                     flavor_X2=flavor_Y, p_bin=0, Delta1=Delta1)
    Y1_vals = X2_temp['Y_1'].values

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
        pd.Series(A1, name='A1'),
        X2_simple,
        pd.Series(A2, name='A2'),
        pd.Series(Y,  name='Y'),
    ], axis=1)

    eval_filename = f'{filename}_eval'
    dat_path = os.path.join(eval_dir, f'{eval_filename}.csv')
    dat.to_csv(dat_path, index=False)

    info_row = pd.DataFrame([{
        'i':           i_val,
        'k1':          k,  'k2':  k,
        's':           2,
        'n':           n,  'p1':  p1, 'p2': p2,
        'flavor_Y':    flavor_Y,
        'seed_params': seed_p,
        'data_seed':   data_seed,
        'filename':    eval_filename,
        'train_file':  filename,
        'delta1':      str([round(float(x), 2) for x in delta1]),
        'Delta1':      str([round(float(x), 2) for x in Delta1]),
        'delta2':      str([round(float(x), 2) for x in delta2]),
        'Delta2':      str([round(float(x), 2) for x in Delta2]),
    }])
    write_header = not os.path.exists(info_out)
    info_row.to_csv(info_out, mode='a', header=write_header, index=False)

    print(f"  i={i_val}  k={k}  flavor={flavor_Y}  "
          f"A1={np.bincount(A1)}  A2={np.bincount(A2)}  Y_mean={Y.mean():.3f}")
    print(f"  ✓ Saved: {eval_filename}.csv")
    return dat


if __name__ == '__main__':
    K_FILTER      = None   # set to 2, 3, or 5; None = all k values
    FLAVOR_FILTER = None   # set to 'expo', 'lognormal', 'sigmoid', 'gamma'; None = all

    os.makedirs(eval_dir, exist_ok=True)

    # Remove old info file so we start fresh
    if os.path.exists(info_out):
        os.remove(info_out)

    info = pd.read_csv(info_path)
    if K_FILTER is not None:
        info = info[info['k1'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k      = int(row['k1'])
        flavor = row['flavor_Y']
        print(f'\n{"="*55}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*55}')
        gen_eval_dataset(row)

    print('\nDone.')
