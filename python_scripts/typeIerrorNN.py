# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: get_typeI_error_rate_NN.py
# Date: 2026-01-24
# Note: Script to calculate the proportion of simulated datasets where the 
#       NN estimator has a significant p-value (p <= 0.05).
#       Since True_diff = 0, this measures the Type I error rate 
#       (false positive rate).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
from typing import Dict

# All possible pairwise comparison suffixes
A_SUFFIXES = ['01', '02', '12', '03', '04', '13', '14', '23', '24', '34']


def calc_nn_rejection_rate(file_path: str) -> Dict[str, float]:
    """
    Calculate the proportion of simulated datasets where NN has p-value <= 0.05.

    Parameters
    ----------
    file_path : str
        Path to a CSV file containing simulation results.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each A_XX suffix to the proportion of datasets where
        NN_est has A_XX_pval <= 0.05.
    """
    df = pd.read_csv(file_path)

    # Filter to only NN_est rows
    nn_df = df[df['estimate'] == 'NN_est'].copy()

    n_sims = len(nn_df)
    results = {}

    for suffix in A_SUFFIXES:
        col_pval = f'A_{suffix}_pval'

        # Check if this column exists in the file
        if col_pval not in nn_df.columns:
            continue

        # Count rows where A_XX_pval <= 0.05
        count = (nn_df[col_pval] <= 0.05).sum()
        results[suffix] = count / n_sims

    return results
