# --------------------------------------------
# Author: Raul
# Date: 2025-01-08
# Script: get_true_diff.py
# Note: This script outputs the true diff
#       in means between two groups
#       given the true data generating process
# --------------------------------------------

import numpy as np
from scipy.special import gamma

def get_true_diff(a_lvl, xb_Y, gamma_vals, Y_flavor):
    """
    Calculate true difference in means between treatment levels
    
    Parameters:
    - a_lvl: tuple of treatment levels (i, j)
    - xb_Y: linear predictor values
    - gamma_vals: treatment effect parameters
    - Y_flavor: functional form of outcome
    
    Returns:
    - True difference in expected outcomes
    """
    
    gamma_allvals = np.concatenate([[0], gamma_vals])
    i, j = a_lvl
    gamma_i = gamma_allvals[i]
    gamma_j = gamma_allvals[j]
    
    if Y_flavor == "expo":
        EY_j = np.mean(np.exp(xb_Y + gamma_j))
        EY_i = np.mean(np.exp(xb_Y + gamma_i))
        
    elif Y_flavor == "sigmoid":
        logistic = lambda x: 1/(1 + np.exp(-x))
        EY_j = np.mean(10 * logistic(xb_Y + gamma_j))
        EY_i = np.mean(10 * logistic(xb_Y + gamma_i))
        
    elif Y_flavor == "lognormal":
        fun_Y = lambda x: (1 / (np.exp(x) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2) * 10
        EY_j = np.mean(fun_Y(xb_Y + gamma_j))
        EY_i = np.mean(fun_Y(xb_Y + gamma_i))
        
    elif Y_flavor == "gamma":
        shape = 2
        scale = 3
        fun_Y = lambda x: (np.exp(shape * x) * np.exp(-np.exp(x)/scale)) / \
                         (gamma(shape) * scale**shape) * 10
        EY_j = np.mean(fun_Y(xb_Y + gamma_j))
        EY_i = np.mean(fun_Y(xb_Y + gamma_i))
    
    d = EY_j - EY_i
    round_d = round(d, 3)
    print(f"  True diff means_{{{j},{i}}} = {round_d}")
    
    return d