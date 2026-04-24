# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate and save three-stage DTR datasets (simple DGP)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from YAX_funs import (gen_X, gen_A, gen_A2_simple, gen_A3_simple,
                      gen_X2, gen_Y2_simple, gen_Y_final_simple)
from sim_params import make_sim_params, print_param_shapes
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))


def gen_3stage_data_simple(s, n, p1, k1, k2, k3, flavor_Y, i=1, seed=None):
    """
    Generate simplified three-stage DTR dataset.

    DGP structure
    -------------
    Covariates are time-invariant: X1 = X2 = X3 (baseline covariates repeated).

    Y1 = f(X1, A1):
      eta1 = X1_with_int @ beta_Y1
               + delta1[a1-1]*1   + Delta1[a1-1]*1   * X1_bin   (full effect)

    Y2 = f(X1, A1, A2):
      eta2 = X1_with_int @ beta_Y1
               + delta1[a1-1]*0.5 + Delta1[a1-1]*0.5 * X1_bin   (halved)
               + delta2[0]*A2     + Delta2[0]*A2     * X1_bin    (full)

    Y  = f(X1, A1, A2, A3):
      eta  = X1_with_int @ beta_Y1
               + delta1[a1-1]*0.25 + Delta1[a1-1]*0.25 * X1_bin  (quartered)
               + delta2[0]*0.5*A2  + Delta2[0]*0.5*A2  * X1_bin  (halved)
               + delta3[0]*A3      + Delta3[0]*A3       * X1_bin  (full)

    A2 stay rule: based on Y1 response vs counterfactual E[Y1(a_opp)|X1]
    A3 stay rule: based on Y2 response vs counterfactual E[Y2(a_opp, A1_obs)|X1]
    Both use threshold = 70th percentile of CATE; p_stay = 0.7 above, 0.5 below.

    Dataset columns: X1_1...X1_p1 | A1 | Y_1 | A2 | Y_2 | A3 | Y

    Saves to:  _1trt_effect/3stages/datasets/s{s}_k{k1}_simple_{flavor_Y}_{i}.csv
    Info file: _1trt_effect/3stages/datasets/_info_simple.csv
    """
    if seed is None:
        seed = 1810 + i

    p2 = p1 + 1  # Y_1 column + p1 duplicated X1 columns (mirrors two-stage structure)
    params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, k3=k3, seed=seed)

    beta_A1 = params['beta_A1']
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']
    delta2  = params['delta2']
    Delta2  = params['Delta2']
    delta3  = params['delta3']
    Delta3  = params['Delta3']

    print(f"\nGenerating (simple 3-stage): n={n}, p1={p1}, k1={k1}, k2={k2}, k3={k3}, "
          f"flavor_Y={flavor_Y}, i={i}, seed={seed}")
    print(f"A1 effects on Y1: delta1={delta1}, Delta1={Delta1}")
    print(f"A2 effects on Y2: delta2[0]={delta2[0]:.4f}, Delta2[0]={Delta2[0]:.4f}")
    print(f"A3 effects on Y:  delta3[0]={delta3[0]:.4f}, Delta3[0]={Delta3[0]:.4f}")

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)
    print(f"\nA1 distribution: {np.bincount(A1)}  proportions: {np.bincount(A1)/n}")

    # Intermediate outcome Y1 (via gen_X2 with p2=1 — only Y_1 column)
    X2_temp = gen_X2(X1=X1, A1=A1, p2=1, delta1=delta1, beta_Y1=beta_Y1,
                     flavor_X2=flavor_Y, p_bin=0, Delta1=Delta1)
    Y1_vals = X2_temp['Y_1'].values

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    A2 = gen_A2_simple(A1=A1, Y1_obs=Y1_vals, k2=k2)
    print(f"A2 distribution: {np.bincount(A2)}  proportions: {np.bincount(A2)/n}")

    Y2_result = gen_Y2_simple(X1=X1, A1=A1, A2=A2, beta_Y1=beta_Y1,
                              delta1=delta1, Delta1=Delta1,
                              delta2_scalar=float(delta2[0]),
                              Delta2_scalar=float(Delta2[0]),
                              flavor_Y=flavor_Y)
    Y2_vals = Y2_result['Y']

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------
    A3 = gen_A3_simple(A2=A2, Y2_obs=Y2_vals, k3=k3)
    print(f"A3 distribution: {np.bincount(A3)}  proportions: {np.bincount(A3)/n}")

    Y_result = gen_Y_final_simple(X1=X1, A1=A1, A2=A2, A3=A3,
                                  beta_Y1=beta_Y1, delta1=delta1, Delta1=Delta1,
                                  delta2_scalar=float(delta2[0]),
                                  Delta2_scalar=float(Delta2[0]),
                                  delta3_scalar=float(delta3[0]),
                                  Delta3_scalar=float(Delta3[0]),
                                  flavor_Y=flavor_Y)
    Y = Y_result['Y']

    print(f"\nY1  summary: min={Y1_vals.min():.2f}, mean={Y1_vals.mean():.2f}, max={Y1_vals.max():.2f}")
    print(f"Y2  summary: min={Y2_vals.min():.2f}, mean={Y2_vals.mean():.2f}, max={Y2_vals.max():.2f}")
    print(f"Y   summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")

    print("Y mean by treatment group:")
    for a1 in range(k1):
        for a2 in range(k2):
            for a3 in range(k3):
                mask = (A1 == a1) & (A2 == a2) & (A3 == a3)
                if mask.sum() > 0:
                    print(f"  A1={a1}, A2={a2}, A3={a3}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

    # --------------------------------------------------------
    # Assemble dataset
    # --------------------------------------------------------
    dat = pd.concat([
        X1,
        pd.Series(A1,      name='A1'),
        pd.Series(Y1_vals, name='Y_1'),
        pd.Series(A2,      name='A2'),
        pd.Series(Y2_vals, name='Y_2'),
        pd.Series(A3,      name='A3'),
        pd.Series(Y,       name='Y'),
    ], axis=1)

    filename = f"s{s}_k{k1}_simple_{flavor_Y}_{i}"

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(dat['Y_1'], bins=30, alpha=0.7)
    axes[0].set_title(f'Y_1  (i={i})')
    axes[1].hist(dat['Y_2'], bins=30, alpha=0.7)
    axes[1].set_title(f'Y_2  (i={i})')
    axes[2].hist(dat['Y'],   bins=30, alpha=0.7)
    axes[2].set_title(f'Y  (i={i})')
    plt.suptitle(f"s={s}, k={k1}, 3-stage simple, {flavor_Y}, i={i}")
    plt.tight_layout()
    img_path = os.path.join(script_dir, f'../_1trt_effect/3stages/images/{filename}.jpeg')
    fig.savefig(img_path)
    plt.close(fig)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    dat_path = os.path.join(script_dir, f'../_1trt_effect/3stages/datasets/{filename}.csv')
    dat.to_csv(dat_path, index=False)

    # --------------------------------------------------------
    # Update _info_simple.csv
    # --------------------------------------------------------
    info_path = os.path.join(script_dir, '../_1trt_effect/3stages/datasets/_info_simple.csv')
    row = pd.DataFrame([{
        'i':        i,
        's':        s,
        'n':        n,
        'p1':       p1,
        'p2':       p2,
        'k1':       k1,
        'k2':       k2,
        'k3':       k3,
        'flavor_Y': flavor_Y,
        'seed':     seed,
        'filename': filename,
        'delta1':   str(delta1.tolist()),
        'Delta1':   str(Delta1.tolist()),
        'delta2':   str([float(delta2[0])]),
        'Delta2':   str([float(Delta2[0])]),
        'delta3':   str([float(delta3[0])]),
        'Delta3':   str([float(Delta3[0])]),
    }])
    write_header = not os.path.exists(info_path)
    row.to_csv(info_path, mode='a', header=write_header, index=False)

    print(f"✓ Saved: {filename}")


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':

    info_path = os.path.join(script_dir, '../_1trt_effect/3stages/datasets/_info_simple.csv')
    if os.path.exists(info_path):
        os.remove(info_path)

    s = 3
    for k in [2]:
        p1 = 3
        n  = k * 200
        for fY in ['expo', 'lognormal', 'gamma', 'sigmoid']:
            for i in range(30):
                gen_3stage_data_simple(s=s, n=n, p1=p1, k1=k, k2=k, k3=k,
                                       flavor_Y=fY, i=i)
