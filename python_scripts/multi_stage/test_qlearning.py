# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test script for two-stage Q-learning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from YAX_funs import gen_X, gen_A, gen_A2, gen_X2, gen_Y_two_stage
from pscores_models import estimate_A_logit
from outcome_models import estimate_optimal_regime_two_stage

# Set seed
np.random.seed(42)

# Parameters
n = 200
p1 = 3
p2 = 2
k1 = 3
k2 = 3

print("=" * 70)
print("TWO-STAGE Q-LEARNING TEST")
print("=" * 70)

# Generate data
print("\nGenerating two-stage data...")
X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)

beta_A1 = np.array([[0.5, 0.3], [-0.3, 0.4], [0.2, -0.1], [0.1, 0.2]])
A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)

gamma1_X2 = np.array([0.5, 1.0])
beta_X2 = np.array([0.0, 0.3, 0.2, 0.1])
X2 = gen_X2(X1=X1, A1=A1, p2=p2, gamma1_X2=gamma1_X2, beta_X2=beta_X2, rho=0.5, p_bin=1)

X_history = pd.concat([X1, pd.Series(A1, name='A1'), X2], axis=1)
beta_A2 = np.array([[0.3, 0.2], [-0.2, 0.3], [0.1, -0.1], [0.2, 0.1], [0.4, -0.2], [-0.3, 0.5], [0.1, -0.2]])
gamma_stay = 0.5  # stay-probability: higher X2 -> more likely to stay on A1
A2 = gen_A2(X1=X1, A1=A1, X2=X2, beta_A2=beta_A2, gamma_stay=gamma_stay,
            flavor_A="logit", k2=k2)

gamma1_Y = np.array([1.0, 2.0])
gamma2_Y = np.array([1.5, 3.0])
beta_Y = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.3])

Y_result = gen_Y_two_stage(
    gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y,
    X1=X1, A1=A1, X2=X2, A2=A2,
    beta_Y=beta_Y, flavor_Y="expo"
)
Y = Y_result['Y']

# Create dataset
dat = pd.concat([X1, pd.Series(A1, name='A1'), X2, pd.Series(A2, name='A2'), pd.Series(Y, name='Y')], axis=1)
print(f"Dataset shape: {dat.shape}")
print(f"Columns: {list(dat.columns)}")

# Estimate propensity scores
print("\nEstimating propensity scores...")
print("  Stage 1 propensity scores...")
dat_stage1 = pd.concat([pd.Series(A1, name='A'), X1], axis=1)
fit_A1 = estimate_A_logit(X=None, dat=dat_stage1, k=k1, verbose=False)
pscores_A1 = fit_A1['pscores']

print("  Stage 2 propensity scores...")
dat_stage2 = pd.concat([pd.Series(A2, name='A'), X_history], axis=1)
fit_A2 = estimate_A_logit(X=None, dat=dat_stage2, k=k2, verbose=False)
pscores_A2 = fit_A2['pscores']

# Q-learning with OLS
print("\n" + "=" * 70)
print("Q-LEARNING WITH OLS")
print("=" * 70)

result_ols = estimate_optimal_regime_two_stage(
    dat=dat,
    pscores_A1=pscores_A1,
    pscores_A2=pscores_A2,
    k1=k1,
    k2=k2,
    model_type='ols'
)

print(f"\nOptimal A1 distribution: {np.bincount(result_ols['optimal_A1'])}")
print(f"Optimal A2 distribution: {np.bincount(result_ols['optimal_A2'])}")

# Compare observed vs optimal
print(f"\nObserved A1 distribution: {np.bincount(A1)}")
print(f"Observed A2 distribution: {np.bincount(A2)}")

# Expected value under optimal regime
optimal_value = np.mean([
    result_ols['Q2_predictions'][i, result_ols['optimal_A2'][i]]
    for i in range(n)
])
print(f"\nExpected value under optimal regime: {optimal_value:.2f}")
print(f"Observed mean outcome: {Y.mean():.2f}")

# Show Q-values for first 5 individuals
print("\n" + "=" * 70)
print("Q-VALUES FOR FIRST 5 INDIVIDUALS")
print("=" * 70)

for i in range(5):
    print(f"\nIndividual {i}:")
    print(f"  Observed: A1={A1[i]}, A2={A2[i]}, Y={Y[i]:.2f}")
    print(f"  Optimal:  A1={result_ols['optimal_A1'][i]}, A2={result_ols['optimal_A2'][i]}")
    print(f"  Q1 values: {result_ols['Q1_predictions'][i, :]}")
    print(f"  Q2 values: {result_ols['Q2_predictions'][i, :]}")
    print(f"  Pseudo-outcome (Ỹ): {result_ols['Y_tilde'][i]:.2f}")

print("\n✓ Two-stage Q-learning test successful!")
