# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: get_type1error.py
# Date: 2026-02-02
# Note: This script takes a muhat_pooled_simk${k}_${pmodel}_${omodel} and computes the
#       type_error_rate across all ${M} datasets using the ${M} pooled variance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations

#k = 3
#A_flavor = 'logit'
#Y_flavor = 'expo'
#zero_effect = True

# generalized to K=k
def get_type1error(k, A_flavor, Y_flavor, zero_effect):
    root = f"./_{'1' if not zero_effect else '0'}trt_effect"

    # Define model types to process
    # Include lognormal only if Y_flavor is lognormal
    model_types = ['nn', 'ols', 'expo']
    if Y_flavor == 'lognormal':
        model_types.append('lognormal')

    for model in model_types:
        # Build file paths based on model type
        if model == 'nn':
            suffix = 'nn'
        else:
            suffix = f'{model}_param'

        csv_path = f"{root}/tables/muhat_pooled_simk{k}_{A_flavor}_{Y_flavor}_est_with_{suffix}.csv"
        pooled_df = pd.read_csv(csv_path)

        # Get all treatment comparisons
        comparisons = list(combinations(range(k), 2))

        results_list = []
        for j, i in comparisons:
            comp_name = f'{j}{i}'

            # Compute delta for this comparison
            pooled_df[f'delta_{comp_name}'] = pooled_df[f'pooled_A{j}'] - pooled_df[f'pooled_A{i}']

            # Get parameter count
            n_params = pooled_df['n_params'].iloc[0] if 'n_params' in pooled_df.columns else 0

            # P-values with dataset-specific variance (adjusted for n-p)
            pvals = pooled_df.groupby('dataset')[f'delta_{comp_name}'].apply(
                lambda x: 2*(1 - norm.cdf(np.abs(np.mean(x))/np.sqrt(np.var(x, ddof=0) * len(x) / (len(x) - n_params) / len(x))))
            )

            # P-values with pooled variance (adjusted for n-p)
            n_total = len(pooled_df[f'delta_{comp_name}'])
            pooled_var = np.var(pooled_df[f'delta_{comp_name}'], ddof=0) * n_total / (n_total - n_params)
            pvals_pooled_var = pooled_df.groupby('dataset')[f'delta_{comp_name}'].apply(
                lambda x: 2*(1 - norm.cdf(np.abs(np.mean(x))/np.sqrt(pooled_var/len(x))))
            )

            # Type I error rates
            type_I_error_pvals = np.mean(pvals <= 0.05)
            type_I_error_pvals_pooled = np.mean(pvals_pooled_var <= 0.05)

            # Store results
            comp_res = pd.DataFrame({
                f'pvals_pooled_var_{comp_name}': pvals_pooled_var,
                f'pvals_dataset_var_{comp_name}': pvals
            })
            results_list.append(comp_res)

        # Combine all comparisons
        res = pd.concat(results_list, axis=1)
        output_path = f"{root}/tables/Results/Type I error rates/type_1_error_simk{k}_{A_flavor}_{Y_flavor}_est_with_{suffix}.csv"
        res.to_csv(output_path)
