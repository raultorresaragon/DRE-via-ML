# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test script for two-stage data generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from YAX_funs import gen_X, gen_A, gen_X2, gen_Y_two_stage

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n = 100
p1 = 3  # Stage 1 covariates
p2 = 2  # Stage 2 covariates
k1 = 3  # Stage 1 treatment levels
k2 = 3  # Stage 2 treatment levels

# Stage 1: Generate baseline data
print("=" * 60)
print("STAGE 1: Generating baseline data")
print("=" * 60)

X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)
print(f"X1 shape: {X1.shape}")
print(f"X1 head:\n{X1.head()}\n")

# Stage 1 treatment model
# beta_A1 should be (p1+1) x (k1-1) = 4 x 2
beta_A1 = np.array([
    [0.5, 0.3],   # Intercept
    [-0.3, 0.4],  # X1 effect
    [0.2, -0.1],  # X2 effect
    [0.1, 0.2]    # X3 effect
])

A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)
print(f"A1 distribution: {np.bincount(A1)}")
print(f"A1 proportions: {np.bincount(A1) / n}\n")

# Stage 2: Generate intermediate covariates
print("=" * 60)
print("STAGE 2: Generating intermediate covariates")
print("=" * 60)

# X2 depends on X1 and A1
gamma1_X2 = np.array([0.5, 1.0])  # Effect of A1=1 and A1=2 on X2
beta_X2 = np.array([0.0, 0.3, 0.2, 0.1])  # Intercept + effects of X1 on X2

X2 = gen_X2(X1=X1, A1=A1, p2=p2, gamma1_X2=gamma1_X2, beta_X2=beta_X2, rho=0.5, p_bin=1)
print(f"X2 shape: {X2.shape}")
print(f"X2 head:\n{X2.head()}\n")

# Check that X2 differs by A1 group
print("X2 mean by A1 group:")
for a in range(k1):
    mask = A1 == a
    print(f"  A1={a}: {X2[mask].mean().values}")
print()

# Stage 2 treatment model (depends on full history)
# beta_A2 should be (p1 + 1 + p2 + 1) x (k2-1) = 7 x 2
X_history = pd.concat([X1, pd.Series(A1, name='A1'), X2], axis=1)
beta_A2 = np.array([
    [0.3, 0.2],    # Intercept
    [-0.2, 0.3],   # X1 effect
    [0.1, -0.1],   # X2 effect
    [0.2, 0.1],    # X3 effect
    [0.4, -0.2],   # A1 effect
    [-0.3, 0.5],   # X2_1 effect
    [0.1, -0.2]    # X2_2 effect
])

A2 = gen_A(X=X_history, beta_A=beta_A2, flavor_A="logit", k=k2)
print(f"A2 distribution: {np.bincount(A2)}")
print(f"A2 proportions: {np.bincount(A2) / n}\n")

# Generate final outcome
print("=" * 60)
print("FINAL OUTCOME: Generating Y")
print("=" * 60)

gamma1_Y = np.array([1.0, 2.0])  # Effect of A1=1 and A1=2 on Y
gamma2_Y = np.array([1.5, 3.0])  # Effect of A2=1 and A2=2 on Y
beta_Y = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.3])  # Intercept + X1 + X2 effects

Y_result = gen_Y_two_stage(
    gamma1_Y=gamma1_Y,
    gamma2_Y=gamma2_Y,
    X1=X1,
    A1=A1,
    X2=X2,
    A2=A2,
    beta_Y=beta_Y,
    flavor_Y="expo"
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

print("\n✓ Two-stage data generation successful!")
