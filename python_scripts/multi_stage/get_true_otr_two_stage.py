# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load a two-stage dataset and compute its true optimal treatment regime (OTR)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params
from get_true_optimal_regime import compute_true_optimal_regime

script_dir   = os.path.dirname(os.path.abspath(__file__))
script_dir = '/Users/raul_torres_aragon/Library/CloudStorage/GoogleDrive-rdtaragon@gmail.com/My Drive/Dissertation/DRE-via-ML/python_scripts/multi_stage'

datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
info_path    = os.path.join(datasets_dir, '_info.csv')


def get_otr(filename, n_samples=1000):
    """
    Load a two-stage dataset, compute the true OTR via backward induction,
    and save results to datasets_dir.

    Parameters
    ----------
    filename  : str   Base filename without extension (e.g. 's2_k2_logit_expo_0')
    n_samples : int   Monte Carlo samples used in Q1 integration (default 1000)
    """
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dat_path = os.path.join(datasets_dir, f'{filename}.csv')
    dat = pd.read_csv(dat_path)

    # ------------------------------------------------------------------
    # Look up params from _info.csv
    # ------------------------------------------------------------------
    info = pd.read_csv(info_path)
    matches = info[info['filename'] == filename]
    if len(matches) == 0:
        raise ValueError(f"No entry found for '{filename}' in _info.csv")
    row = matches.iloc[0]

    p1       = int(row['p1'])
    p2       = int(row['p2'])
    k1       = int(row['k1'])
    k2       = int(row['k2'])
    flavor_Y = row['flavor_Y']
    seed     = int(row['seed'])
    i        = int(row['i'])

    # ------------------------------------------------------------------
    # Reconstruct true params from seed
    # ------------------------------------------------------------------
    params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)

    # ------------------------------------------------------------------
    # Split dataset into model components
    # ------------------------------------------------------------------
    X1_cols = [c for c in dat.columns if c.startswith('X1_')]
    X2_cols = ['Y_1'] + [c for c in dat.columns if c.startswith('X2_')]

    X1 = dat[X1_cols]
    A1 = dat['A1'].values
    X2 = dat[X2_cols] # includes intermediate outcome Y_1

    print(f"\nComputing OTR for: {filename}")
    print(f"  n={len(dat)}, p1={p1}, p2={p2}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}, seed={seed}")

    # ------------------------------------------------------------------
    # Compute true OTR (backward induction)
    # ------------------------------------------------------------------
    result = compute_true_optimal_regime(
        X1=X1, X2=X2, A1=A1,
        k1=k1, k2=k2,
        delta1=params['delta1'], beta_Y1=params['beta_Y1'],
        delta2=params['delta2'], beta_Y2=params['beta_Y2'],
        p2=p2, rho=0.5, flavor_Y=flavor_Y,
        n_samples=n_samples,
        Delta1=params['Delta1'], Delta2=params['Delta2']
    )

    # ------------------------------------------------------------------
    # Assemble output: OTR decisions + Q-values
    # ------------------------------------------------------------------
    otr_dat = pd.DataFrame({
        'd1_star': result['true_optimal_A1'],
        'd2_star': result['true_optimal_A2'],
    })
    for a in range(k1):
        otr_dat[f'Q1_a{a}'] = result['true_Q1_all'][:, a]
    for a in range(k2):
        otr_dat[f'Q2_a{a}'] = result['true_Q2_all'][:, a]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_filename = f'{filename}_OTR'
    out_path = os.path.join(datasets_dir, f'{out_filename}.csv')
    otr_dat.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_filename}.csv")

    return otr_dat


# ============================================================
# Run over all datasets in _info.csv
# ============================================================
info = pd.read_csv(info_path)
for _, row in info.iterrows():
    get_otr(row['filename'], n_samples = 300)
