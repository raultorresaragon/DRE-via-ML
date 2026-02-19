# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: one_sim_interactive.py
# Date: 2026-01-08
# Note: Interactive version of one_sim.py for testing single iterations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import time


# Import custom functions
from YAX_funs import gen_X, gen_A, gen_Y
from pscores_models import estimate_A_nn, estimate_A_logit
from outcome_models import estimate_Y_nn, estimate_Y_ols, estimate_Y_expo, estimate_Y_lognormal
from get_true_diff import get_true_diff
from compute_Vn import get_Vn
from Y_Yhat_sorted_plots import plot_predicted_A_Y

# Run one simulation iteration for k>=3 treatments
# Parameters:
zero_effect = False
k = 5
n = 300 * k
if k == 2:
    p = 3
elif k == 3:
    p = 8
elif k == 5:
    p = 12

# Start timer for a single run
total_start_time = time.time()

Xmu = np.round(np.random.uniform(-1, 1, p), 1)
beta_A = np.vstack([
            np.full((1, k-1), -0.1),
            np.round(np.random.uniform(-1.5, 1.5, (p, k-1)), 1)
        ])
beta_Y = np.concatenate([
            [1],
            np.round(np.random.uniform(-1, 1, p), 1)
        ]) * 1
gamma = np.array([0.6, 0.4, 0.75, 0.17])[:(k-1)] * (1 if not zero_effect else 0)
A_flavor = "logit"  # "logit" "tanh"
Y_flavor = "lognormal"  # "expo", "sigmoid", "gamma", "lognormal"
hidunits = [5, 20]
eps = [100, 250]
penals = [0.001, 0.01]
verbose = False
iter = 1
export_images = False
root = f"./_{'1' if not zero_effect else '0'}trt_effect/"
rho = round(np.random.uniform(0.4, 0.6), 1)


# Generate data
X = gen_X(n=n, p=p, rho=rho, mu=Xmu, p_bin=1)
A = gen_A(X=X, beta_A=beta_A, flavor_A=A_flavor, k=k)
Y_result = gen_Y(gamma=gamma, X=X, A=A, beta_Y=beta_Y, flavor_Y=Y_flavor)
Y = Y_result['Y']

# Create dataset
dat = pd.concat([
        pd.Series(Y, name='Y'),
        pd.Series(A, name='A'),
        X
    ], axis=1)
dat['A'].value_counts(normalize=True) * 100

# Ensure positive outcomes
assert np.all(Y >= 0), "All Y values must be non-negative"

# Save dataset (commented out for interactive use)
# os.makedirs(f"{root}/datasets", exist_ok=True)
# dat.to_csv(f"{root}/datasets/df_k{k}{A_flavor}{Y_flavor}_dset{iter}.csv", index=False)

# Plot generated Y
X_with_intercept = np.column_stack([np.ones(n), X.values])
xb_Y = X_with_intercept @ beta_Y

colors = ['black', 'darkred', 'green', 'blue', 'skyblue']

plt.figure(figsize=(10, 6))
for a_val in range(k):
    mask = A == a_val
    if np.any(mask):
        plt.scatter(xb_Y[mask], Y[mask], c=colors[a_val], label=f'A={a_val}', alpha=0.6)

plt.xlabel('xb_Y')
plt.ylabel('Y')
plt.title(f'k={k} flavor:{A_flavor}-{Y_flavor}\nN={n} dim(X)={p}')
plt.legend()

if export_images:
    os.makedirs(f"{root}/images/genYplots", exist_ok=True)
    plt.savefig(f"{root}/images/genYplots/genY_k{k}{A_flavor}{Y_flavor}_dset{iter}.png", dpi=150, bbox_inches='tight')
    plt.show()

# Print treatment probabilities
for i in range(k):
    prob = np.mean(A == i)
    print(f"  P(A={i})= {prob:.1f}")

# Calculate true differences
true_diffs = []
for comparison in combinations(range(k), 2):
    true_diff = get_true_diff(comparison, xb_Y, gamma, Y_flavor)
    true_diffs.append(true_diff)

# Estimate propensity scores
print("\nEstimating propensity scores...")
start_time = time.time()
fit_A_nn = estimate_A_nn(X=None, dat=dat, k=k,
                         hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
fit_A_logit = estimate_A_logit(X=None, dat=dat, k=k, verbose=verbose)
print(f"A model time: {time.time() - start_time:.2f}s")

# Estimate outcome models
print("Estimating outcome models...")
start_time = time.time()

# Fit NN model (only once)
fit_Y_nn = estimate_Y_nn(dat, pscores_df=fit_A_nn['pscores'], k=k,
                         hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)

# Fit all parametric models
fit_Y_ols = estimate_Y_ols(dat, pscores_df=fit_A_logit['pscores'], k=k)
fit_Y_expo = estimate_Y_expo(dat, pscores_df=fit_A_logit['pscores'], k=k)

# Fit lognormal only if Y_flavor is lognormal
fit_Y_lognormal = None
if Y_flavor == "lognormal":
    fit_Y_lognormal = estimate_Y_lognormal(dat, pscores_df=fit_A_logit['pscores'], k=k)

print(f"Y model time: {time.time() - start_time:.2f}s")

# Plot predictions for each parametric model
param_models = [('ols', fit_Y_ols), ('expo', fit_Y_expo)]
if Y_flavor == "lognormal":
    param_models.append(('lognormal', fit_Y_lognormal))

for Y_param, fit_Y_param in param_models:
    plot_predicted_A_Y(beta_A, beta_Y, dat, fit_Y_nn, fit_Y_param, gamma,
                      fit_A_nn, fit_A_logit, A_flavor, Y_flavor, iter, k, Y_param=Y_param,
                      save=export_images, root=root)

# Extract results
def get_naive_est(comparison):
    j, i = comparison
    d = np.mean(Y[A == i]) - np.mean(Y[A == j])
    print(f"  Naive diff means_{{{j},{i}}} = {d:.3f}")
    return d

naive_est = [get_naive_est(comp) for comp in combinations(range(k), 2)]

# Extract NN estimates
nn_estimates = []
nn_pvals = []
for key in fit_Y_nn.keys():
    nn_estimates.append(fit_Y_nn[key][0]['diff_means_' + key[2:]])
    nn_pvals.append(fit_Y_nn[key][0]['pval_' + key[2:]])

# Helper function to extract estimates from a fitted model
def extract_param_estimates(fit_Y_model):
    estimates = []
    pvals = []
    for key in fit_Y_model.keys():
        estimates.append(fit_Y_model[key][0]['diff_means_' + key[2:]])
        pvals.append(fit_Y_model[key][0]['pval_' + key[2:]])
    return estimates, pvals

# Extract estimates from all parametric models
ols_estimates, ols_pvals = extract_param_estimates(fit_Y_ols)
expo_estimates, expo_pvals = extract_param_estimates(fit_Y_expo)

lognormal_estimates, lognormal_pvals = None, None
if Y_flavor == "lognormal":
    lognormal_estimates, lognormal_pvals = extract_param_estimates(fit_Y_lognormal)

# Create results DataFrames for each model type
comparison_names = [f"A_{j}{i}" for j, i in combinations(range(k), 2)]

# Helper function to create results dataframe for a specific parametric model
def create_results_df(Y_param_name, param_estimates, param_pvals):
    results_data = {
        'dataset': [iter] * 4,
        'estimate': ['True_diff', 'NN_est', f'Logit{Y_param_name}_est', 'Naive_est'],
    }
    for idx, name in enumerate(comparison_names):
        results_data[name] = [
            true_diffs[idx],
            nn_estimates[idx],
            param_estimates[idx],
            naive_est[idx]
        ]
        results_data[f"{name}_pval"] = [
            np.nan,
            nn_pvals[idx],
            param_pvals[idx],
            np.nan
        ]
    return pd.DataFrame(results_data)

my_k_rows_ols = create_results_df('ols', ols_estimates, ols_pvals)
my_k_rows_expo = create_results_df('expo', expo_estimates, expo_pvals)
my_k_rows_lognormal = None
if Y_flavor == "lognormal":
    my_k_rows_lognormal = create_results_df('lognormal', lognormal_estimates, lognormal_pvals)

print("\nResults (OLS):")
print(my_k_rows_ols)
print("\nResults (Expo):")
print(my_k_rows_expo)
if Y_flavor == "lognormal":
    print("\nResults (Lognormal):")
    print(my_k_rows_lognormal)

# Compute OTR
X_new = pd.DataFrame(np.random.uniform(-8, 8, (5, p)),
                     columns=[f'X{i+1}' for i in range(p)])

Vn_df = get_Vn(fit_Y_nn, X_new)
Vn_df['dataset'] = iter
Vn_df = Vn_df[['dataset'] + [col for col in Vn_df.columns if col != 'dataset']]
print("\nOTR:")
print(Vn_df)

# Helper function to extract muhat pooled from a fitted model
def extract_muhat_pooled(fit_Y_model, iter_num):
    muhat_dict = {'dataset': iter_num}
    for key in fit_Y_model.keys():
        result_dict = fit_Y_model[key][0]
        j, i = key[2], key[3]  # Extract treatment indices from key like 'A_01'
        muhat_dict[f'pooled_A{j}'] = result_dict[f'muhat_{j}']
        muhat_dict[f'pooled_A{i}'] = result_dict[f'muhat_{i}']
    return pd.DataFrame(muhat_dict)

# Extract muhat vectors by treatment level from all models
muhat_pooled_nn = extract_muhat_pooled(fit_Y_nn, iter)
muhat_pooled_ols = extract_muhat_pooled(fit_Y_ols, iter)
muhat_pooled_expo = extract_muhat_pooled(fit_Y_expo, iter)

muhat_pooled_lognormal = None
if Y_flavor == "lognormal":
    muhat_pooled_lognormal = extract_muhat_pooled(fit_Y_lognormal, iter)

print("\nPooled muhat by treatment (NN):")
print(muhat_pooled_nn.head())
print("\nPooled muhat by treatment (OLS):")
print(muhat_pooled_ols.head())
print("\nPooled muhat by treatment (Expo):")
print(muhat_pooled_expo.head())
if Y_flavor == "lognormal":
    print("\nPooled muhat by treatment (Lognormal):")
    print(muhat_pooled_lognormal.head())


print(f"\nThis run took: {time.time() - total_start_time:.2f} seconds")
