# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gen_eval_datasets_single_stage.py
# Generate one evaluation dataset per row of _info_single.csv for the single-stage DGP.
#
# For each row i, uses the SAME DGP parameters (same make_sim_params_single seed) as
# the training dataset but a fresh random draw (data_seed = row seed + DATA_SEED_OFFSET).
#
# Saved to: _1trt_effect/1stage/datasets/eval_sets/{filename}_eval.csv
# Info file: _1trt_effect/1stage/datasets/eval_sets/_info_eval.csv
#
# Optional filters (set in __main__):
#   K_FILTER      : int or None  — only generate for k = K_FILTER
#   FLAVOR_FILTER : str or None  — only generate for a specific flavor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from YAX_funs import gen_X, gen_A, gen_Y
from sim_params_single import make_sim_params_single

datasets_dir = os.path.join(script_dir, '../_1trt_effect/1stage/datasets')
eval_dir     = os.path.join(datasets_dir, 'eval_sets')
info_path    = os.path.join(datasets_dir, '_info_single.csv')
info_out     = os.path.join(eval_dir,     '_info_eval.csv')

DATA_SEED_OFFSET = 99999   # added to the row's params seed to get the data seed


def gen_eval_dataset(row, data_seed_offset=DATA_SEED_OFFSET):
    """
    Generate an evaluation dataset for one row of _info_single.csv.

    Uses the same DGP parameters as the training dataset (same make_sim_params_single seed)
    but draws fresh data using (row seed + data_seed_offset).

    Parameters
    ----------
    row              : pd.Series   One row from _info_single.csv
    data_seed_offset : int         Added to the row seed to produce the data seed

    Returns
    -------
    pd.DataFrame   The generated evaluation dataset
    """
    k         = int(row['k'])
    flavor_Y  = row['flavor_Y']
    seed_p    = int(row['seed'])
    n         = int(row['n'])
    p         = int(row['p'])
    i_val     = int(row['i'])
    filename  = row['filename']
    data_seed = seed_p + data_seed_offset

    params = make_sim_params_single(p=p, k=k, seed=seed_p, flavor_Y=flavor_Y)

    # make_sim_params_single uses default_rng — does NOT affect legacy np.random state
    np.random.seed(data_seed)

    # Generate eval data
    X        = gen_X(n=n, p=p, rho=0.5, p_bin=1)
    A        = gen_A(X=X, beta_A=params['beta_A'], flavor_A='logit', k=k)
    Y_result = gen_Y(delta=params['delta'], X=X, A=A,
                     beta_Y=params['beta_Y'], Delta=params['Delta'],
                     flavor_Y=flavor_Y)
    Y = Y_result['Y']

    dat = pd.concat([X,
                     pd.Series(A, name='A'),
                     pd.Series(Y, name='Y')], axis=1)

    eval_filename = f'{filename}_eval'
    dat_path = os.path.join(eval_dir, f'{eval_filename}.csv')
    dat.to_csv(dat_path, index=False)

    # Update _info_eval.csv
    info_row = pd.DataFrame([{
        'i':           i_val,
        'k':           k,
        's':           int(row['s']),
        'n':           n,
        'p':           p,
        'flavor_Y':    flavor_Y,
        'seed_params': seed_p,
        'data_seed':   data_seed,
        'filename':    eval_filename,
        'train_file':  filename,
        'delta':       str([round(float(x), 2) for x in params['delta']]),
        'Delta':       str([round(float(x), 2) for x in params['Delta']]),
    }])
    write_header = not os.path.exists(info_out)
    info_row.to_csv(info_out, mode='a', header=write_header, index=False)

    print(f"  i={i_val}  k={k}  flavor={flavor_Y}  "
          f"A={np.bincount(A)}  Y_mean={Y.mean():.3f}")
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
        info = info[info['k'] == K_FILTER]
    if FLAVOR_FILTER is not None:
        info = info[info['flavor_Y'] == FLAVOR_FILTER]

    for _, row in info.iterrows():
        k      = int(row['k'])
        flavor = row['flavor_Y']
        print(f'\n{"="*55}\ni={row["i"]}  k={k}  flavor={flavor}\n{"="*55}')
        gen_eval_dataset(row)

    print('\nDone.')
