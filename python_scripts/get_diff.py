# --------------------------------------------
# Author: Raul
# Date: 2025-01-08
# Script: get_diff.py
# Note: This script creates the function that outputs
#       the estimated difference in means
#       given a fit model
# --------------------------------------------

import numpy as np
from scipy.stats import norm

def get_diff(ghat_B, delta_B, ghat_A, delta_A, pi_hat, Y):
    """
    Estimate difference in means using doubly robust estimation
    
    Parameters:
    - ghat_B: predicted outcomes for treatment B
    - delta_B: indicator for treatment B
    - ghat_A: predicted outcomes for treatment A  
    - delta_A: indicator for treatment A
    - pi_hat: propensity scores
    - Y: observed outcomes
    
    Returns:
    - Dictionary with difference estimates and statistics
    """
    
    # Avoid division by zero
    pi_hat = np.clip(pi_hat, 1e-6, 1-1e-6)
    
    # Doubly robust estimators
    muhat_B = ghat_B + (delta_B * (Y - ghat_B) / pi_hat) / np.mean(delta_B / pi_hat)
    muhat_A = ghat_A + (delta_A * (Y - ghat_A) / (1 - pi_hat)) / np.mean(delta_A / (1 - pi_hat))
    
    # Difference in means
    diff_means = np.mean(muhat_B - muhat_A)
    
    # Variance estimation
    diff_var = np.var(muhat_B - muhat_A) / len(muhat_B)
    
    # P-value (two-sided test)
    pval = 2 * (1 - norm.cdf(abs(diff_means) / np.sqrt(diff_var)))
    
    return {
        'diff_means': diff_means,
        'muhat_B': muhat_B,
        'muhat_A': muhat_A,
        'diff_var': diff_var,
        'pval': pval
    }