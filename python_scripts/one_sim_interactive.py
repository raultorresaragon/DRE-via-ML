# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: one_sim_k3plus.py
# Date: 2026-01-08
# Note: This script creates a function
#       to run one iteration of k3 plus sims
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
k = 2
n = 300 * k
if k == 2:
    p = 3
elif k == 3:
    p = 8
elif k == 5:
    p = 12

Xmu = np.round(np.random.uniform(-1, 1, p), 1)
beta_A = np.vstack([
            np.full((1, k-1), 0.1),
            np.round(np.random.uniform(-2, 2, (p, k-1)), 1)
        ])
beta_Y = np.concatenate([
            [1], 
            np.round(np.random.uniform(-1, 1, p), 1)
        ]) * 1
gamma = np.array([0.6, 0.4, 0.75, 0.17])[:(k-1)] * (1 if not zero_effect else 0)
A_flavor = "logit" #"logit" "tanh"
Y_flavor = "lognormal" #"expo", "sigmoid", "gamma", "lognormal"
Y_param = "lognormal" #"ols", "expo", "lognormal"
hidunits = [5, 20]
eps = [100, 250]
penals = [0.001, 0.01]
verbose = True
iter = 1
export_images = True
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
    
# Save dataset
os.makedirs(f"{root}/datasets", exist_ok=True)
dat.to_csv(f"{root}/datasets/df_k{k}{A_flavor}{Y_flavor}_dset{iter}.csv", index=False)
    
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
fit_Y_nn = estimate_Y_nn(dat, pscores_df=fit_A_nn['pscores'], k=k,
                         hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
if Y_param == "expo":
    fit_Y_param = estimate_Y_expo(dat, pscores_df=fit_A_logit['pscores'], k=k)
elif Y_param == "lognormal":
    fit_Y_param = estimate_Y_lognormal(dat, pscores_df=fit_A_logit['pscores'], k=k, sigma2=0.25)
else:
    fit_Y_param = estimate_Y_ols(dat, pscores_df=fit_A_logit['pscores'], k=k)
print(f"Y model time: {time.time() - start_time:.2f}s")
    
# Plot predictions
plot_predicted_A_Y(beta_A, beta_Y, dat, fit_Y_nn, fit_Y_param, gamma,
                   fit_A_nn, fit_A_logit, A_flavor, Y_flavor, iter, k, expo_or_ols=Y_param,
                   save=export_images, root=root)
    
# Extract results
def get_naive_est(comparison):
    j, i = comparison
    d = np.mean(Y[A == i]) - np.mean(Y[A == j])
    print(f"  Naive diff means_{{{j},{i}}} = {d:.3f}")
    return d
    
naive_est = [get_naive_est(comp) for comp in combinations(range(k), 2)]
    
# Extract NN model estimates
nn_estimates = []
nn_pvals = []
for key in fit_Y_nn.keys():
    nn_estimates.append(fit_Y_nn[key][0]['diff_means_' + key[2:]])
    nn_pvals.append(fit_Y_nn[key][0]['pval_' + key[2:]])
    
# Extract parametric model estimates
param_estimates = []
param_pvals = []
for key in fit_Y_param.keys():
    param_estimates.append(fit_Y_param[key][0]['diff_means_' + key[2:]])
    param_pvals.append(fit_Y_param[key][0]['pval_' + key[2:]])
    
# Create results DataFrame
comparison_names = [f"A_{j}{i}" for j, i in combinations(range(k), 2)]
    
results_data = {
    'dataset': [iter] * 4,
    'estimate': ['True_diff', 'NN_est', f'Logit{Y_param}_est', 'Naive_est'],
}
    
# Add estimates for each comparison
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
    
my_k_rows = pd.DataFrame(results_data)
print(my_k_rows)
    
# Compute OTR
X_new = pd.DataFrame(np.random.uniform(-8, 8, (5, p)), 
                     columns=[f'X{i+1}' for i in range(p)])
    
Vn_df = get_Vn(fit_Y_nn, X_new)
Vn_df['dataset'] = iter
Vn_df = Vn_df[['dataset'] + [col for col in Vn_df.columns if col != 'dataset']]   
print(Vn_df)
