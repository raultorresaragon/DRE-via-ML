# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test script for two-stage data generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
# when running interactively:
# os.chdir('/Users/raulta/Desktop/Tesina/DRE-via-ML/python_scripts/multi_stage')
from YAX_funs import gen_X, gen_A, gen_A2, gen_X2, gen_Y_two_stage
from sim_params import make_sim_params, print_param_shapes
import matplotlib.pyplot as plt

# ============================================================
# CUSTOMIZABLE PARAMETERS - Change these as needed
# ============================================================
n        = 100
p1       = 3
p2       = p1 + 1
k1       = 2   # <-- change freely; all arrays auto-size
k2       = 2   # <-- change freely; all arrays auto-size
flavor_Y = 'lognormal'  # expo, lognormal, sigmoid, gamma

params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=1810)

beta_A1    = params['beta_A1']
beta_A2    = params['beta_A2']
gamma_stay = params['gamma_stay']
delta1     = params['delta1']
beta_Y1    = params['beta_Y1']
Delta1     = params['Delta1']
delta2     = params['delta2']
beta_Y2    = params['beta_Y2']
Delta2     = params['Delta2']

print(f"Testing with: n={n}, p1={p1}, p2={p2}, k1={k1}, k2={k2}\n")
print_param_shapes(params, p1=p1, p2=p2, k1=k1, k2=k2)

# ============================================================
# Stage 1: Generate baseline data
# ============================================================
print("\n" + "=" * 60)
print("STAGE 1: Generating baseline data")
print("=" * 60)

X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
print(f"X1 shape: {X1.shape}")
print(f"X1 head:\n{X1.head()}\n")

A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)
print(f"A1 distribution: {np.bincount(A1)}")
print(f"A1 proportions: {np.bincount(A1) / n}\n")

# ============================================================
# Stage 2: Generate intermediate covariates
# ============================================================
print("=" * 60)
print("STAGE 2: Generating intermediate covariates")
print("=" * 60)

X2 = gen_X2(X1=X1, A1=A1, p2=p2, delta1=delta1, beta_Y1=beta_Y1,
            flavor_X2=flavor_Y, rho=0.5, p_bin=1, Delta1=Delta1)
print(f"X2 shape: {X2.shape}")
print(f"X2 head:\n{X2.head()}\n")

# Check that X2 differs by A1 group
print("X2 mean by A1 group:")
for a in range(k1):
    mask = A1 == a
    print(f"  A1={a}: {X2[mask].mean().values}")
print()

A2 = gen_A2(X1=X1, A1=A1, X2=X2, beta_A2=beta_A2, gamma_stay=gamma_stay,
            flavor_A="logit", k2=k2)
print(f"A2 distribution: {np.bincount(A2)}")
print(f"A2 proportions: {np.bincount(A2) / n}\n")

# ============================================================
# Generate final outcome
# ============================================================
print("=" * 60)
print("FINAL OUTCOME: Generating Y")
print("=" * 60)

Y_result = gen_Y_two_stage(
    delta2=delta2, X1=X1, A1=A1, X2=X2, A2=A2,
    beta_Y2=beta_Y2, flavor_Y=flavor_Y, Delta2=Delta2
)

Y = Y_result['Y']
print(f"Y shape: {Y.shape}")
print(f"Y summary: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}\n")

# Check Y differs by treatment groups
print("Y mean by treatment groups:")
for a1 in range(k1):
    for a2 in range(k2):
        mask = (A1 == a1) & (A2 == a2)
        if mask.sum() > 0:
            print(f"  A1={a1}, A2={a2}: n={mask.sum()}, Y_mean={Y[mask].mean():.2f}")

# Create final dataset
print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

dat = pd.concat([
    X1,
    pd.Series(A1, name='A1'),
    X2,
    pd.Series(A2, name='A2'),
    pd.Series(Y, name='Y')
], axis=1)

print(f"Dataset shape: {dat.shape}")
print(f"Dataset columns: {list(dat.columns)}")
print(f"\nFirst 10 rows:\n{dat.head(10)}")

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
ax1.hist(dat['Y'], alpha=0.5, label='Y')
ax1.set_title('Y')
ax2.hist(dat['Y_1'], alpha=0.5, label='Y_1')
ax2.set_title('Y_1')
plt.tight_layout()
plt.show()

print("\n✓ Two-stage data generation successful!")
