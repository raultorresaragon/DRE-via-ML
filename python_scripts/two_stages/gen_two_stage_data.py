# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate and save two-stage DTR datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from YAX_funs import gen_X, gen_A, gen_A2, gen_X2, gen_Y_two_stage, gen_A2_simple, gen_Y_simple
from sim_params import make_sim_params, print_param_shapes
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

def gen_2stage_data(s, n, p1, p2, k1, k2, flavor_Y, i=1, seed=None):
    if seed is None:
        seed = 1810 + i

    params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)

    beta_A1    = params['beta_A1']    # stage1 covariates (+ intercept) effects on pscore stage1
    beta_A2    = params['beta_A2']    # stage1, stage2, A1 covariates effects on pscore stage2
    gamma_stay = params['gamma_stay'] # logodds bump to stay on A if A is working
    delta1     = params['delta1']     # main trt effect on Y_1
    beta_Y1    = params['beta_Y1']    # covariates 1st stage (+ intercept) effect on Y_1
    Delta1     = params['Delta1']     # A1 x X1_bin interaction on Y_1
    delta2     = params['delta2']     # main trt effect on Y
    beta_Y2    = params['beta_Y2']    # [intercept, X1, A1, X2] effects on Y
    Delta2     = params['Delta2']     # A2 x X1_bin interaction on Y

    print(f"\nGenerating: n={n}, p1={p1}, p2={p2}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}, i={i}, seed={seed}")
    print_param_shapes(params, p1=p1, p2=p2, k1=k1, k2=k2)
    print(f"main trt effect(s)    on Y_1: delta1={delta1}")
    print(f"interaction effect(s) on Y_1: Delta1={Delta1}")
    print(f"main trt effect(s)    on Y:   delta2={delta2}")
    print(f"interaction effect(s) on Y:   Delta2={Delta2}")

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)
    print(f"\nA1 distribution: {np.bincount(A1)}  proportions: {np.bincount(A1)/n}")

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    X2 = gen_X2(X1=X1, A1=A1, p2=p2, delta1=delta1, beta_Y1=beta_Y1,
                flavor_X2=flavor_Y, rho=0.5, p_bin=1, Delta1=Delta1)
    A2 = gen_A2(X1=X1, A1=A1, X2=X2, beta_A2=beta_A2, gamma_stay=gamma_stay,
                flavor_A="logit", k2=k2)
    print(f"A2 distribution: {np.bincount(A2)}  proportions: {np.bincount(A2)/n}")

    # --------------------------------------------------------
    # Final outcome
    # --------------------------------------------------------
    Y_result = gen_Y_two_stage(
        delta2=delta2, X1=X1, A1=A1, X2=X2, A2=A2,
        beta_Y2=beta_Y2, flavor_Y=flavor_Y, Delta2=Delta2
    )
    Y = Y_result['Y']
    print(f"\nY   summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")

    print("Y mean by treatment group:")
    for a1 in range(k1):
        for a2 in range(k2):
            mask = (A1 == a1) & (A2 == a2)
            if mask.sum() > 0:
                print(f"  A1={a1}, A2={a2}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

    # --------------------------------------------------------
    # Assemble dataset
    # --------------------------------------------------------
    dat = pd.concat([
        X1,
        pd.Series(A1, name='A1'),
        X2,
        pd.Series(A2, name='A2'),
        pd.Series(Y, name='Y')
    ], axis=1)

    Y1 = dat['Y_1']
    print(f"\nY_1 summary: min={Y1.min():.2f}, mean={Y1.mean():.2f}, max={Y1.max():.2f}")
    print(f"Dataset shape: {dat.shape}  columns: {list(dat.columns)}")

    filename = f"s{s}_k{k1}_logit_{flavor_Y}_{i}"

    # --------------------------------------------------------
    # Histograms — save before closing figure
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(dat['Y'],   bins=30, alpha=0.7)
    ax1.set_title(f'Y  (i={i})')
    ax2.hist(dat['Y_1'], bins=30, alpha=0.7)
    ax2.set_title(f'Y_1  (i={i})')
    plt.suptitle(f"s={s}, k={k1}, {flavor_Y}, i={i}")
    plt.tight_layout()
    img_path = os.path.join(script_dir, f'../_1trt_effect/2stages/images/{filename}.jpeg')
    fig.savefig(img_path)
    plt.close(fig)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    dat_path = os.path.join(script_dir, f'../_1trt_effect/2stages/datasets/{filename}.csv')
    dat.to_csv(dat_path, index=False)

    # --------------------------------------------------------
    # Update _info.csv
    # --------------------------------------------------------
    info_path = os.path.join(script_dir, '../_1trt_effect/2stages/datasets/_info.csv')
    row = pd.DataFrame([{
        'i':        i,
        's':        s,
        'n':        n,
        'p1':       p1,
        'p2':       p2,
        'k1':       k1,
        'k2':       k2,
        'flavor_Y': flavor_Y,
        'seed':     seed,
        'filename': filename,
        'delta1':   str(delta1.tolist()),
        'Delta1':   str(Delta1.tolist()),
        'delta2':   str(delta2.tolist()),
        'Delta2':   str(Delta2.tolist()),
    }])
    write_header = not os.path.exists(info_path)
    row.to_csv(info_path, mode='a', header=write_header, index=False)

    print(f"✓ Saved: {filename}")


def gen_2stage_data_simple(s, n, p1, k1, k2, flavor_Y, i=1, seed=None):
    """
    Generate simplified two-stage DTR dataset.

    Key differences from gen_2stage_data:
    - Covariates are time-invariant: X2 = X1 (baseline covariates repeated at stage 2)
    - A2 assigned via response-based stay rule (see gen_A2_simple)
    - Y model: f(X1 @ beta_Y1 + delta2[0]*A2 + Delta2[0]*A2*X1_bin) + N(0, 0.5)
      using beta_Y1 by default (pass beta_Y_override to gen_Y_simple to override)
      A2 enters as a scalar multiplier (dose interpretation for k>2)

    Dataset column structure matches gen_2stage_data for downstream compatibility:
      X1_1...X1_p1 | A1 | Y_1 | X2_1...X2_p1 | A2 | Y
    where X2_j = X1_j (time-invariant baseline covariates).

    Saves to:  datasets/s{s}_k{k1}_simple_{flavor_Y}_{i}.csv
    Info file: datasets/_info_simple.csv  (separate from _info.csv)
    """
    if seed is None:
        seed = 1810 + i

    p2 = p1 + 1  # Y_1 + p1 duplicated X1 columns (mirrors original structure)
    params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)

    beta_A1 = params['beta_A1']
    beta_Y1 = params['beta_Y1']
    delta1  = params['delta1']
    Delta1  = params['Delta1']
    delta2  = params['delta2']
    Delta2  = params['Delta2']

    print(f"\nGenerating (simple): n={n}, p1={p1}, k1={k1}, k2={k2}, flavor_Y={flavor_Y}, i={i}, seed={seed}")
    print(f"main trt effect(s)    on Y_1: delta1={delta1}")
    print(f"interaction effect(s) on Y_1: Delta1={Delta1}")
    print(f"main trt effect      on Y:   delta2[0]={delta2[0]:.4f}")
    print(f"interaction effect   on Y:   Delta2[0]={Delta2[0]:.4f}")

    # --------------------------------------------------------
    # Stage 1  (identical to gen_2stage_data)
    # --------------------------------------------------------
    X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)
    print(f"\nA1 distribution: {np.bincount(A1)}  proportions: {np.bincount(A1)/n}")

    # --------------------------------------------------------
    # Intermediate outcome Y1  (via gen_X2 with p2=1 — only Y_1 column)
    # --------------------------------------------------------
    X2_temp = gen_X2(X1=X1, A1=A1, p2=1, delta1=delta1, beta_Y1=beta_Y1,
                     flavor_X2=flavor_Y, p_bin=0, Delta1=Delta1)
    Y1_vals = X2_temp['Y_1'].values

    # --------------------------------------------------------
    # Stage 2 covariates: X2 = X1  (time-invariant baseline)
    # Columns: Y_1, X2_1, ..., X2_p1
    # --------------------------------------------------------
    X2_simple = pd.DataFrame({'Y_1': Y1_vals})
    for j, col in enumerate(X1.columns):
        X2_simple[f'X2_{j+1}'] = X1[col].values

    # --------------------------------------------------------
    # Stage 2 treatment  (response-based stay rule)
    # --------------------------------------------------------
    A2 = gen_A2_simple(A1=A1, Y1_obs=Y1_vals, k2=k2)
    print(f"A2 distribution: {np.bincount(A2)}  proportions: {np.bincount(A2)/n}")

    # --------------------------------------------------------
    # Final outcome
    # --------------------------------------------------------
    Y_result = gen_Y_simple(X1=X1, A1=A1, A2=A2, beta_Y1=beta_Y1,
                            delta1=delta1, Delta1=Delta1,
                            delta2_scalar=float(delta2[0]),
                            Delta2_scalar=float(Delta2[0]),
                            flavor_Y=flavor_Y)
    Y = Y_result['Y']
    print(f"\nY   summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")
    print(f"Y_1 summary: min={Y1_vals.min():.2f}, mean={Y1_vals.mean():.2f}, max={Y1_vals.max():.2f}")

    print("Y mean by treatment group:")
    for a1 in range(k1):
        for a2 in range(k2):
            mask = (A1 == a1) & (A2 == a2)
            if mask.sum() > 0:
                print(f"  A1={a1}, A2={a2}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

    # --------------------------------------------------------
    # Assemble dataset  (same column order as gen_2stage_data)
    # --------------------------------------------------------
    dat = pd.concat([
        X1,
        pd.Series(A1, name='A1'),
        X2_simple,
        pd.Series(A2, name='A2'),
        pd.Series(Y,  name='Y')
    ], axis=1)

    filename = f"s{s}_k{k1}_simple_{flavor_Y}_{i}"

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(dat['Y'],   bins=30, alpha=0.7)
    ax1.set_title(f'Y  (i={i})')
    ax2.hist(dat['Y_1'], bins=30, alpha=0.7)
    ax2.set_title(f'Y_1  (i={i})')
    plt.suptitle(f"s={s}, k={k1}, simple, {flavor_Y}, i={i}")
    plt.tight_layout()
    img_path = os.path.join(script_dir, f'../_1trt_effect/2stages/images/{filename}.jpeg')
    fig.savefig(img_path)
    plt.close(fig)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    dat_path = os.path.join(script_dir, f'../_1trt_effect/2stages/datasets/{filename}.csv')
    dat.to_csv(dat_path, index=False)

    # --------------------------------------------------------
    # Update _info_simple.csv  (separate from _info.csv)
    # --------------------------------------------------------
    info_path = os.path.join(script_dir, '../_1trt_effect/2stages/datasets/_info_simple.csv')
    row = pd.DataFrame([{
        'i':        i,
        's':        s,
        'n':        n,
        'p1':       p1,
        'p2':       p2,
        'k1':       k1,
        'k2':       k2,
        'flavor_Y': flavor_Y,
        'seed':     seed,
        'filename': filename,
        'delta1':   str(delta1.tolist()),
        'Delta1':   str(Delta1.tolist()),
        'delta2':   str([float(delta2[0])]),
        'Delta2':   str([float(Delta2[0])]),
    }])
    write_header = not os.path.exists(info_path)
    row.to_csv(info_path, mode='a', header=write_header, index=False)

    print(f"✓ Saved: {filename}")


# ============================================================
# Run
# ============================================================
if __name__ == '__main__': # <- so this doesn't run when importing it

    info_path = os.path.join(script_dir, '../_1trt_effect/2stages/datasets/_info_simple.csv')                                                                  
    if os.path.exists(info_path):                                                                                                                       
        os.remove(info_path) 
    s = 2
    for k in [2]:
        if k == 2:
            p1 = 3
        else:
            p1 = 8
        n = k * 200
        p2 = p1 + 1
        for fY in ['expo', 'lognormal', 'gamma', 'sigmoid']:
            for i in range(30):
                #gen_2stage_data(s=s, n=n, p1=p1, p2=p2, k1=k, k2=k, flavor_Y=fY, i=i)
                gen_2stage_data_simple(s=s, n=n, p1=p1, k1=k, k2=k, flavor_Y=fY, i=i)
