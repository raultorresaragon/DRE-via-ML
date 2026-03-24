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

def get_true_diff(a_lvl, xb_Y, delta_vals, Y_flavor, Delta=None, X_bin=None):
    """
    Calculate true difference in means between treatment levels

    Parameters:
    - a_lvl: tuple of treatment levels (i, j)
    - xb_Y: linear predictor values (Xbeta_Y, no treatment terms)
    - delta_vals: main treatment effect parameters (shape k-1)
    - Y_flavor: functional form of outcome
    - Delta: interaction (effect modification) coefficients (shape k-1), optional
    - X_bin: binary effect modifier values (shape n), optional

    Returns:
    - True difference in expected outcomes
    """

    delta_allvals = np.concatenate([[0], delta_vals])
    Delta_allvals = np.concatenate([[0], Delta]) if Delta is not None else None
    i, j = a_lvl

    # Effective shift per patient: main effect + interaction term
    def eta(trt_idx):
        shift = delta_allvals[trt_idx]
        if Delta_allvals is not None and X_bin is not None:
            shift = shift + X_bin * Delta_allvals[trt_idx]
        return xb_Y + shift

    if Y_flavor == "expo":
        EY_j = np.mean(np.exp(eta(j)))
        EY_i = np.mean(np.exp(eta(i)))

    elif Y_flavor == "sigmoid":
        logistic = lambda x: 1/(1 + np.exp(-x))
        EY_j = np.mean(10 * logistic(eta(j)))
        EY_i = np.mean(10 * logistic(eta(i)))

    elif Y_flavor == "lognormal":
        #fun_Y = lambda x: (1 / (np.exp(x) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2) * 10
        #EY_j = np.mean(fun_Y(eta(j)))
        #EY_i = np.mean(fun_Y(eta(i)))
        # For lognormal: if log(Y) ~ N(μ, σ²), then E[Y] = exp(μ + σ²/2)
        sigma = 0.5
        EY_j = np.mean(np.exp(eta(j) + sigma**2 / 2))
        EY_i = np.mean(np.exp(eta(i) + sigma**2 / 2))

    elif Y_flavor == "gamma":
        shape = 2
        scale = 3
        fun_Y = lambda x: (np.exp(shape * x) * np.exp(-np.exp(x)/scale)) / \
                         (gamma(shape) * scale**shape) * 10
        EY_j = np.mean(fun_Y(eta(j)))
        EY_i = np.mean(fun_Y(eta(i)))
    
    d = EY_j - EY_i
    round_d = round(d, 3)
    print(f"  True diff means_{{{j},{i}}} = {round_d}")
    
    return d