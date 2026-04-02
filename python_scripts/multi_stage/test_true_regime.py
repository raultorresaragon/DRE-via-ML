# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test: Compare estimated vs. true optimal regime
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from YAX_funs import gen_X, gen_A, gen_A2, gen_X2, gen_Y_two_stage
from pscores_models import estimate_A_logit
from outcome_models import estimate_optimal_regime_two_stage
from get_true_optimal_regime import compute_true_optimal_regime, evaluate_regime_accuracy
from sim_params import make_sim_params

# ============================================================
# CUSTOMIZABLE PARAMETERS - Change these as needed
# ============================================================
n   = 200
p1  = 3
p2  = 2
k1  = 3   # <-- change freely; all arrays auto-size
k2  = 3   # <-- change freely; all arrays auto-size

np.random.seed(42)
params = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=42)

beta_A1    = params['beta_A1']
beta_A2    = params['beta_A2']
gamma_stay = params['gamma_stay']
delta1     = params['delta1']
beta_Y1    = params['beta_Y1']
Delta1     = params['Delta1']
delta2     = params['delta2']
beta_Y2    = params['beta_Y2']
Delta2     = params['Delta2']

print("=" * 80)
print("COMPARING ESTIMATED VS. TRUE OPTIMAL REGIME")
print("=" * 80)

# Generate data
print("\nGenerating data...")
X1 = gen_X(n=n, p=p1, rho=0.5, p_bin=1)

A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A="logit", k=k1)

X2 = gen_X2(X1=X1, A1=A1, p2=p2, delta1=delta1, beta_Y1=beta_Y1,
            flavor_X2="expo", rho=0.5, p_bin=1, Delta1=Delta1)

X_history = pd.concat([X1, pd.Series(A1, name='A1'), X2], axis=1)
A2 = gen_A2(X1=X1, A1=A1, X2=X2, beta_A2=beta_A2, gamma_stay=gamma_stay,
            flavor_A="logit", k2=k2)

Y_result = gen_Y_two_stage(
    delta2=delta2, X1=X1, A1=A1, X2=X2, A2=A2,
    beta_Y2=beta_Y2, flavor_Y="expo", Delta2=Delta2
)
Y = Y_result['Y']

dat = pd.concat([X1, pd.Series(A1, name='A1'), X2, pd.Series(A2, name='A2'), pd.Series(Y, name='Y')], axis=1)

# Compute TRUE optimal regime
print("\n" + "=" * 80)
print("COMPUTING TRUE OPTIMAL REGIME")
print("=" * 80)

true_regime = compute_true_optimal_regime(
    X1=X1, X2=X2, A1=A1, k1=k1, k2=k2,
    delta1=delta1, beta_Y1=beta_Y1,
    delta2=delta2, beta_Y2=beta_Y2,
    p2=p2, rho=0.5, flavor_Y="expo",
    n_samples=500,
    Delta2=Delta2, Delta1=Delta1
)

print(f"\nTrue optimal regime distribution:")
print(f"  A1: {np.bincount(true_regime['true_optimal_A1'])}")
print(f"  A2: {np.bincount(true_regime['true_optimal_A2'])}")

# Estimate propensity scores
print("\n" + "=" * 80)
print("ESTIMATING REGIMES WITH DIFFERENT MODELS")
print("=" * 80)

dat_stage1 = pd.concat([pd.Series(A1, name='A'), X1], axis=1)
fit_A1 = estimate_A_logit(X=None, dat=dat_stage1, k=k1, verbose=False)

dat_stage2 = pd.concat([pd.Series(A2, name='A'), X_history], axis=1)
fit_A2 = estimate_A_logit(X=None, dat=dat_stage2, k=k2, verbose=False)

# Estimate with OLS
regime_ols = estimate_optimal_regime_two_stage(
    dat=dat, pscores_A1=fit_A1['pscores'], pscores_A2=fit_A2['pscores'],
    k1=k1, k2=k2, model_type='ols'
)

# Estimate with Expo
regime_expo = estimate_optimal_regime_two_stage(
    dat=dat, pscores_A1=fit_A1['pscores'], pscores_A2=fit_A2['pscores'],
    k1=k1, k2=k2, model_type='expo'
)

# Evaluate accuracy
print("\n" + "=" * 80)
print("REGIME ACCURACY COMPARISON")
print("=" * 80)

print("\nOLS Model:")
ols_accuracy = evaluate_regime_accuracy(
    regime_ols['optimal_A1'], regime_ols['optimal_A2'],
    true_regime['true_optimal_A1'], true_regime['true_optimal_A2']
)
print(f"  A1 accuracy: {ols_accuracy['A1_accuracy']:.2%} ({ols_accuracy['A1_agreement']}/{ols_accuracy['n']})")
print(f"  A2 accuracy: {ols_accuracy['A2_accuracy']:.2%} ({ols_accuracy['A2_agreement']}/{ols_accuracy['n']})")
print(f"  Joint accuracy: {ols_accuracy['joint_accuracy']:.2%} ({ols_accuracy['joint_agreement']}/{ols_accuracy['n']})")

print("\nExpo Model:")
expo_accuracy = evaluate_regime_accuracy(
    regime_expo['optimal_A1'], regime_expo['optimal_A2'],
    true_regime['true_optimal_A1'], true_regime['true_optimal_A2']
)
print(f"  A1 accuracy: {expo_accuracy['A1_accuracy']:.2%} ({expo_accuracy['A1_agreement']}/{expo_accuracy['n']})")
print(f"  A2 accuracy: {expo_accuracy['A2_accuracy']:.2%} ({expo_accuracy['A2_agreement']}/{expo_accuracy['n']})")
print(f"  Joint accuracy: {expo_accuracy['joint_accuracy']:.2%} ({expo_accuracy['joint_agreement']}/{expo_accuracy['n']})")

# Compare Q-value correlations
print("\n" + "=" * 80)
print("Q-VALUE CORRELATIONS WITH TRUE Q-VALUES")
print("=" * 80)

# Stage 2 Q-values
for a2 in range(k2):
    ols_corr = np.corrcoef(regime_ols['Q2_predictions'][:, a2], true_regime['true_Q2_all'][:, a2])[0, 1]
    expo_corr = np.corrcoef(regime_expo['Q2_predictions'][:, a2], true_regime['true_Q2_all'][:, a2])[0, 1]
    print(f"\nQ2(A2={a2}):")
    print(f"  OLS correlation:  {ols_corr:.3f}")
    print(f"  Expo correlation: {expo_corr:.3f}")

# Stage 1 Q-values
for a1 in range(k1):
    ols_corr = np.corrcoef(regime_ols['Q1_predictions'][:, a1], true_regime['true_Q1_all'][:, a1])[0, 1]
    expo_corr = np.corrcoef(regime_expo['Q1_predictions'][:, a1], true_regime['true_Q1_all'][:, a1])[0, 1]
    print(f"\nQ1(A1={a1}):")
    print(f"  OLS correlation:  {ols_corr:.3f}")
    print(f"  Expo correlation: {expo_corr:.3f}")

print("\n" + "=" * 80)
print("✓ Comparison complete!")
print("=" * 80)
print("\nInterpretation:")
print("- Higher accuracy = better regime estimation")
print("- Higher Q-value correlation = better value function approximation")
print("- Expo should perform better since data is generated with expo link")
