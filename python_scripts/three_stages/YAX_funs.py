# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: YAX_functions.py
# Date: 2025-01-08
# Note: This script creates functions needed for 
#       simulating DRE for k=3+ where the differences
#       in means are compared pairwise: combn(k,2)
# Modified: 2026-02-19 - Added two-stage functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from itertools import combinations

# --------------------------
# Generate X (design matrix)
# --------------------------
def gen_X(p, rho=0.6, mu=None, n=1000, p_bin=1):
    """
    Generate correlated design matrix X
    
    Parameters:
    - p: number of covariates
    - rho: correlation parameter
    - mu: mean vector
    - n: sample size
    - p_bin: number of binary covariates
    """
    if mu is None:
        mu = np.zeros(p)
    
    # Create correlation matrix
    Sigma = np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])
    
    # Generate multivariate normal
    X = multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    X_df = pd.DataFrame(X, columns=[f'X1_{i+1}' for i in range(p)])
    
    # Convert last p_bin columns to binary
    if p_bin > 0:
        for j in range(p-p_bin, p):
            col_name = f'X1_{j+1}'
            X_df[col_name] = (X_df[col_name] > X_df[col_name].mean()).astype(int)
    
    return X_df

# -----------
# Generate A 
# -----------
def gen_A(X, beta_A, flavor_A="logit", k=None):
    """
    Generate treatment assignments
    
    Parameters:
    - X: design matrix
    - beta_A: coefficient matrix for treatment model
    - flavor_A: "logit" or "tanh"
    - k: number of treatment levels
    """
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X.values])
    
    if beta_A.ndim == 1:
        beta_A = beta_A.reshape(-1, 1)
    
    xb = X_with_intercept @ beta_A
    
    if flavor_A == "logit":
        exp_xb = np.exp(xb)
        denom = 1 + np.sum(exp_xb, axis=1, keepdims=True)
        probs = np.column_stack([1/denom.flatten(), (exp_xb/denom)])
        
    elif flavor_A == "tanh":
        raw_scores = 0.5 * (np.tanh(xb) + 1)
    
        if k == 2:
        # Binary case: only need 2 classes
            prob_1 = raw_scores[:, 0]
            prob_0 = 1 - prob_1
            probs = np.column_stack([prob_0, prob_1])
        else:
        # Multi-class case
            probs = np.zeros((n, k))
            for i in range(min(raw_scores.shape[1], k-1)):
                other_classes = np.sum(raw_scores[:, [j for j in range(raw_scores.shape[1]) if j != i]], axis=1)
                probs[:, i] = raw_scores[:, i] / (1 + other_classes)
            probs[:, -1] = 1 - np.sum(probs[:, :-1], axis=1)
    
    # Sample treatment assignments
    if probs.shape[1] > 2:  # Multi-class
        A = np.array([np.random.multinomial(1, prob).argmax() for prob in probs])
    else:  # Binary
        A = np.random.binomial(1, probs[:, 1])
    
    return A



# ------------
# Generate Y 
# ------------
def gen_Y(delta, X, A, beta_Y, flavor_Y="expo"):
    """
    Generate outcomes

    Parameters:
    - delta: treatment effects
    - X: design matrix
    - A: treatment assignments
    - beta_Y: outcome model coefficients
    - flavor_Y: functional form for outcome
    """
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X.values])

    # Create treatment indicators (excluding baseline A=0)
    unique_A = np.unique(A)
    A_mat = np.column_stack([np.where(A == a, 1, 0) for a in unique_A[1:]])

    # Linear predictor
    xb_delta_a = X_with_intercept @ beta_Y
    if A_mat.shape[1] > 0 and len(delta) > 0:
        xb_delta_a += A_mat @ delta

    # Apply functional form
    if flavor_Y == "expo":
        Y = np.exp(xb_delta_a) + np.random.normal(0, 0.5, n)

    elif flavor_Y == "sigmoid":
        Y = 1/(1 + np.exp(-xb_delta_a)) * 10 + np.random.normal(0, 0.5, n)

    elif flavor_Y == "gamma":
        shape = 2
        scale = 3
        Y = (np.exp(shape * xb_delta_a) * np.exp(-np.exp(xb_delta_a)/scale)) / \
            (math.gamma(shape) * scale**shape) * 10 + np.random.normal(0, 0.5, n) + 0.1

    elif flavor_Y == "lognormal":
        # If log(Y) ~ N(μ, σ²), then Y ~ Lognormal
        sigma = 0.5
        Y = np.exp(xb_delta_a + np.random.normal(0, sigma, n))

        # bell-curve transform
        #Y = (1 / (np.exp(xb_delta_a) * np.sqrt(2 * np.pi))) * \
        #    np.exp(-0.5 * xb_delta_a**2) * 10 + np.random.normal(0, 0.5, n)

    # Ensure positive outcomes
    Y = np.abs(Y)

    # Handle extreme values for exponential
    if flavor_Y == "expo":
        threshold = np.percentile(Y, 99.95)
        Y = np.minimum(Y, threshold)

    return {'Y': Y, 'xb_delta_a': xb_delta_a}


# ========================================
# TWO-STAGE FUNCTIONS (ORIGINAL)
# ========================================

# --------------------------
# Generate A2 (stage 2 treatment with stay-probability)
# --------------------------
def gen_A2(X1, A1, X2, beta_A2, gamma_stay, flavor_A="logit", k2=None):
    """
    Generate stage 2 treatment assignments with stay-probability mechanism.

    If X2 is high (patient responding well to A1), increase P(A2 = A1).

    Parameters:
    - X1: stage 1 covariates (DataFrame)
    - A1: stage 1 treatment (array)
    - X2: stage 2 covariates (DataFrame)
    - beta_A2: coefficient matrix for A2 model, shape (1 + p1 + 1 + p2, k2-1)
              rows: intercept, X1 cols, A1, X2 cols
    - gamma_stay: scalar controlling how strongly X2 influences staying on A1
                  higher gamma_stay -> stronger tendency to stay when X2 is high
    - flavor_A: "logit" or "tanh"
    - k2: number of stage 2 treatment levels

    Returns:
    - A2: array of stage 2 treatment assignments
    """
    n = X1.shape[0]

    # Build history matrix: [1, X1, A1, X2]
    X_history = np.column_stack([
        np.ones(n),
        X1.values,
        A1,
        X2.values
    ])

    if beta_A2.ndim == 1:
        beta_A2 = beta_A2.reshape(-1, 1)

    # Linear predictor for each non-reference treatment
    xb = X_history @ beta_A2  # shape (n, k2-1)

    # Compute X2 summary for each patient (mean across X2 columns)
    X2_mean = X2.mean(axis=1).values  # shape (n,)

    # Add stay bonus: for each treatment level, add gamma_stay * X2_mean if A1 == that level
    # xb has k2-1 columns for treatments 1, 2, ..., k2-1 (treatment 0 is reference)
    for j in range(1, k2):
        # Patients whose A1 == j get a bonus in the j-th column (index j-1)
        stay_bonus = gamma_stay * X2_mean * (A1 == j)
        xb[:, j-1] += stay_bonus

    # For A1 == 0 patients, add bonus to reference category (handled via normalization)
    # We need to subtract from all non-reference categories to effectively boost P(A2=0)
    stay_bonus_ref = gamma_stay * X2_mean * (A1 == 0)
    xb -= stay_bonus_ref.reshape(-1, 1)  # subtract from all non-reference

    if flavor_A == "logit":
        exp_xb = np.exp(xb)
        denom = 1 + np.sum(exp_xb, axis=1, keepdims=True)
        probs = np.column_stack([1/denom.flatten(), (exp_xb/denom)])

    elif flavor_A == "tanh":
        raw_scores = 0.5 * (np.tanh(xb) + 1)

        if k2 == 2:
            prob_1 = raw_scores[:, 0]
            prob_0 = 1 - prob_1
            probs = np.column_stack([prob_0, prob_1])
        else:
            probs = np.zeros((n, k2))
            for i in range(min(raw_scores.shape[1], k2-1)):
                other_classes = np.sum(raw_scores[:, [j for j in range(raw_scores.shape[1]) if j != i]], axis=1)
                probs[:, i] = raw_scores[:, i] / (1 + other_classes)
            probs[:, -1] = 1 - np.sum(probs[:, :-1], axis=1)

    # Clip probabilities to valid range
    probs = np.clip(probs, 0.001, 0.999)
    probs = probs / probs.sum(axis=1, keepdims=True)  # renormalize

    # Sample treatment assignments
    if probs.shape[1] > 2:
        A2 = np.array([np.random.multinomial(1, prob).argmax() for prob in probs])
    else:
        A2 = np.random.binomial(1, probs[:, 1])

    return A2


# --------------------------
# Generate X2 (stage 2 covariates)
# --------------------------
def gen_X2(X1, A1, p2, delta1, beta_Y1, flavor_X2="expo", rho=0.5, p_bin=1,
           Delta1=None):
    """
    Generate stage 2 covariates that depend on stage 1 history
    X2_1 is an intermediate outcome matching the Y flavor
    Remaining X2 covariates are correlated normal variables

    Parameters:
    - X1: stage 1 covariates (DataFrame)
    - A1: stage 1 treatment (array)
    - p2: number of stage 2 covariates
    - delta1: A1 main effects on Y_1 (array of length k1-1)
    - beta_Y1: coefficients for X1 effect on Y_1 (array of length p1+1)
    - flavor_X2: functional form for Y_1 ("expo", "sigmoid", "gamma", "lognormal")
    - rho: correlation parameter for remaining X2 covariates
    - p_bin: number of binary covariates in X2
    - Delta1: A1 × binary modifier interaction coefficients for Y_1 (array of length k1-1)
              Binary modifier is the last column of X1.

    Returns:
    - X2: DataFrame with stage 2 covariates
    """
    n = X1.shape[0]
    X1_with_intercept = np.column_stack([np.ones(n), X1.values])

    # Binary effect modifier: last column of X1
    X1_bin = X1.iloc[:, -1].values

    # Create treatment indicators for A1 (excluding baseline A1=0)
    unique_A1 = np.unique(A1)
    A1_mat = np.column_stack([np.where(A1 == a, 1, 0) for a in unique_A1[1:]])

    # Linear predictor for Y_1
    X2_linear = X1_with_intercept @ beta_Y1
    if A1_mat.shape[1] > 0 and len(delta1) > 0:
        X2_linear += A1_mat @ delta1
    if Delta1 is not None and len(Delta1) > 0:
        X2_linear += (A1_mat * X1_bin.reshape(-1, 1)) @ Delta1
    
    # X2_1: Intermediate outcome matching Y flavor
    if flavor_X2 == "expo":
        X2_1 = np.exp(X2_linear) + np.random.normal(0, 0.5, n)
        
    elif flavor_X2 == "sigmoid":
        X2_1 = 1/(1 + np.exp(-X2_linear)) * 10 + np.random.normal(0, 0.5, n)
        
    elif flavor_X2 == "gamma":
        shape = 2
        scale = 3
        X2_1 = (np.exp(shape * X2_linear) * np.exp(-np.exp(X2_linear)/scale)) / \
               (math.gamma(shape) * scale**shape) * 10 + np.random.normal(0, 0.5, n) + 0.1
               
    elif flavor_X2 == "lognormal":
        sigma = 0.5
        X2_1 = np.exp(X2_linear + np.random.normal(0, sigma, n))
    
    # Ensure X2_1 is non-negative
    X2_1 = np.maximum(X2_1, 0.01)
    
    # Initialize X2 matrix
    X2 = np.zeros((n, p2))
    X2[:, 0] = X2_1  # First column is the intermediate outcome
    
    # Remaining X2 covariates: correlated normal variables (if p2 > 1)
    if p2 > 1:
        Sigma = np.array([[rho**abs(i-j) for j in range(p2-1)] for i in range(p2-1)])
        for i in range(n):
            X2[i, 1:] = multivariate_normal.rvs(mean=np.full(p2-1, X2_linear[i]), cov=Sigma)
    
    # Column naming: Y_1 = intermediate outcome, X2_1..X2_{p2-1} = additional covariates
    X2_df = pd.DataFrame(X2, columns=['Y_1'] + [f'X2_{i+1}' for i in range(p2-1)])

    # Convert last p_bin of the X2 covariates to binary
    if p_bin > 0 and p2 > 1:
        for j in range(p2-p_bin, p2):
            col_name = f'X2_{j}'
            X2_df[col_name] = (X2_df[col_name] > X2_df[col_name].median()).astype(int)
    
    return X2_df


# ------------
# Generate Y (two-stage version)
# ------------
def gen_Y_two_stage(delta2, X1, A1, X2, A2, beta_Y2, flavor_Y="expo",
                    Delta2=None):
    """
    Generate outcomes for two-stage setting

    Parameters:
    - delta2: stage 2 treatment main effects on Y (array of length k2-1)
    - X1: stage 1 covariates
    - A1: stage 1 treatment (included as raw covariate in beta_Y2)
    - X2: stage 2 covariates
    - A2: stage 2 treatment
    - beta_Y2: outcome model coefficients for [1, X1, A1, X2] (length 1+p1+1+p2)
    - flavor_Y: functional form for outcome
    - Delta2: stage 2 treatment × binary modifier interaction coefficients (array of length k2-1)
              Binary modifier is the last column of X1.

    Returns:
    - Dictionary with Y and linear predictor
    """
    n = X1.shape[0]

    # Feature vector: [1, X1, A1, X2]  — A1 enters as a raw covariate
    X_combined = np.column_stack([np.ones(n), X1.values, A1, X2.values])

    # Create treatment indicators for A2 (excluding baseline)
    unique_A2 = np.unique(A2)
    A2_mat = np.column_stack([np.where(A2 == a, 1, 0) for a in unique_A2[1:]])

    # Binary effect modifier for Delta2: I(Y_1 > median(Y_1))
    Y1_vals = X2.iloc[:, 0].values
    Y1_threshold = np.median(Y1_vals)
    Y1_bin = (Y1_vals > Y1_threshold).astype(int)

    # Linear predictor
    xb_delta_a = X_combined @ beta_Y2

    # Add stage 2 treatment main effects
    if A2_mat.shape[1] > 0 and len(delta2) > 0:
        xb_delta_a += A2_mat @ delta2

    # Add stage 2 treatment × Y_1 response interaction
    if Delta2 is not None and len(Delta2) > 0:
        xb_delta_a += (A2_mat * Y1_bin.reshape(-1, 1)) @ Delta2

    # Apply functional form (same as single-stage)
    if flavor_Y == "expo":
        Y = np.exp(xb_delta_a) + np.random.normal(0, 0.5, n)

    elif flavor_Y == "sigmoid":
        Y = 1/(1 + np.exp(-xb_delta_a)) * 10 + np.random.normal(0, 0.5, n)

    elif flavor_Y == "gamma":
        shape = 2
        scale = 3
        Y = (np.exp(shape * xb_delta_a) * np.exp(-np.exp(xb_delta_a)/scale)) / \
            (math.gamma(shape) * scale**shape) * 10 + np.random.normal(0, 0.5, n) + 0.1

    elif flavor_Y == "lognormal":
        sigma = 0.5
        Y = np.exp(xb_delta_a + np.random.normal(0, sigma, n))

    # Ensure positive outcomes
    Y = np.abs(Y)

    # Handle extreme values for exponential
    if flavor_Y == "expo":
        threshold = np.percentile(Y, 99.95)
        Y = np.minimum(Y, threshold)

    return {'Y': Y, 'xb_delta_a': xb_delta_a}


# ========================================
# SIMPLE TWO-STAGE FUNCTIONS
# ========================================

def _mean_outcome_simple(eta, flavor_Y):
    """
    Analytic conditional mean E[Y | eta, flavor_Y], dropping mean-zero noise.

    Used to compute counterfactual potential outcomes directly from DGP parameters
    without simulation (noise terms are mean-zero so E[Y|eta] = f(eta)).

    expo     : E[Y] = exp(eta)
    sigmoid  : E[Y] = 10 / (1 + exp(-eta))
    gamma    : E[Y] = f_gamma(eta) * 10 + 0.1
    lognormal: E[Y] = exp(eta + sigma^2/2),  sigma = 0.5
    """
    if flavor_Y == 'expo':
        return np.exp(eta)
    elif flavor_Y == 'sigmoid':
        return 10.0 / (1.0 + np.exp(-eta))
    elif flavor_Y == 'gamma':
        shape, scale = 2, 3
        return (np.exp(shape * eta) * np.exp(-np.exp(eta) / scale) /
                (math.gamma(shape) * scale**shape)) * 10 + 0.1
    elif flavor_Y == 'lognormal':
        sigma = 0.5
        return np.exp(eta + sigma**2 / 2.0)
    else:
        raise ValueError(f'Unknown flavor_Y: {flavor_Y}')


def gen_A2_simple(A1, Y1_obs, k2):
    """
    Generate stage 2 treatment for simplified two-stage DGP.

    Assignment rule based on observed Y1 value:
      threshold = 70th percentile of Y1_obs across the sample
      p_stay_i  = 0.7  if Y1_obs_i > threshold  (high outcome, stay on A1)
                = 0.5  otherwise
      P(A2 = A1_i)        = p_stay_i
      P(A2 = other arm j) = (1 - p_stay_i) / (k2 - 1)  for each j != A1_i

    Parameters
    ----------
    A1      : array (n,)   -- stage 1 treatment
    Y1_obs  : array (n,)   -- observed intermediate outcome
    k2      : int          -- number of stage 2 treatment levels

    Returns
    -------
    A2 : array (n,)
    """
    n         = len(A1)
    threshold = np.percentile(Y1_obs, 70)
    p_stay    = np.where(Y1_obs > threshold, 0.7, 0.5)

    A2 = np.empty(n, dtype=int)
    for idx in range(n):
        probs          = np.full(k2, (1 - p_stay[idx]) / (k2 - 1))
        probs[A1[idx]] = p_stay[idx]
        A2[idx]        = np.random.choice(k2, p=probs)

    return A2


def gen_Y_simple(X1, A1, A2, beta_Y1, delta1, Delta1, delta2_scalar, Delta2_scalar,
                 flavor_Y, beta_Y_override=None):
    """
    Generate final outcome Y for simplified two-stage DGP.

    Model:
      eta = X1_with_int @ beta_Y
              + sum_{a1>0} I(A1=a1) * (delta1[a1-1]*0.5 + Delta1[a1-1]*0.5 * X1_bin)
              + delta2_scalar * A2  +  Delta2_scalar * A2 * X1_bin
      Y   = f(eta) + epsilon,   epsilon ~ N(0, 0.5)
      f   = flavor_Y link function (expo / sigmoid / gamma / lognormal)

    A1 enters with halved effects (delta1*0.5, Delta1*0.5) — same structure as Y1
    but diluted by half. A2 enters as a scalar multiplier (dose interpretation for k>2).
    X1_bin is the last column of X1 (binary effect modifier).

    Parameters
    ----------
    X1              : DataFrame (n, p1)
    A1              : array (n,)           -- stage 1 treatment
    A2              : array (n,)           -- stage 2 treatment
    beta_Y1         : array (p1+1,)        -- default outcome coefficients (with intercept)
    delta1          : array (k1-1,)        -- stage 1 main effects (halved in eta)
    Delta1          : array (k1-1,)        -- stage 1 x X1_bin interaction (halved in eta)
    delta2_scalar   : float                -- stage 2 main treatment effect
    Delta2_scalar   : float                -- stage 2 treatment x X1_bin interaction
    flavor_Y        : str
    beta_Y_override : array or None        -- if provided, used instead of beta_Y1

    Returns
    -------
    dict with keys 'Y' and 'xb_delta_a'
    """
    beta_Y = beta_Y_override if beta_Y_override is not None else beta_Y1
    n           = X1.shape[0]
    X1_with_int = np.column_stack([np.ones(n), X1.values])
    X1_bin      = X1.iloc[:, -1].values

    eta = X1_with_int @ beta_Y + delta2_scalar * A2 + Delta2_scalar * A2 * X1_bin

    # A1 direct effect on Y: same structure as Y1 but halved
    for a1 in range(1, len(delta1) + 1):
        mask     = (A1 == a1)
        eta[mask] += delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin[mask]

    if flavor_Y == 'expo':
        Y = np.exp(eta) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'sigmoid':
        Y = 10.0 / (1.0 + np.exp(-eta)) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'gamma':
        shape, scale = 2, 3
        Y = ((np.exp(shape * eta) * np.exp(-np.exp(eta) / scale)) /
             (math.gamma(shape) * scale**shape)) * 10 + np.random.normal(0, 0.5, n) + 0.1
    elif flavor_Y == 'lognormal':
        sigma = 0.5
        Y = np.exp(eta + np.random.normal(0, sigma, n))
    else:
        raise ValueError(f'Unknown flavor_Y: {flavor_Y}')

    Y = np.abs(Y)
    if flavor_Y == 'expo':
        Y = np.minimum(Y, np.percentile(Y, 99.95))

    return {'Y': Y, 'xb_delta_a': eta}


# ========================================
# SIMPLE THREE-STAGE FUNCTIONS
# ========================================

def gen_Y2_simple(X1, A1, A2, beta_Y1, delta1, Delta1, delta2_scalar, Delta2_scalar,
                  flavor_Y):
    """
    Generate intermediate outcome Y2 for simplified three-stage DGP.

    Model (attenuation: A1 at 0.5 of full effect, A2 at full effect):
      eta = X1_with_int @ beta_Y1
              + sum_{a1>0} I(A1=a1) * (delta1[a1-1]*0.5 + Delta1[a1-1]*0.5 * X1_bin)
              + delta2_scalar * A2  +  Delta2_scalar * A2 * X1_bin
      Y2  = f(eta) + epsilon,   epsilon ~ N(0, 0.5)

    Parameters
    ----------
    X1             : DataFrame (n, p1)
    A1             : array (n,)
    A2             : array (n,)
    beta_Y1        : array (p1+1,)
    delta1         : array (k1-1,)
    Delta1         : array (k1-1,)
    delta2_scalar  : float           -- A2 main effect at full strength
    Delta2_scalar  : float           -- A2 x X1_bin interaction at full strength
    flavor_Y       : str

    Returns
    -------
    dict with keys 'Y' and 'xb_delta_a'
    """
    n           = X1.shape[0]
    X1_with_int = np.column_stack([np.ones(n), X1.values])
    X1_bin      = X1.iloc[:, -1].values

    eta = X1_with_int @ beta_Y1 + delta2_scalar * A2 + Delta2_scalar * A2 * X1_bin

    # A1 direct effect on Y2: halved relative to Y1
    for a1 in range(1, len(delta1) + 1):
        mask      = (A1 == a1)
        eta[mask] += delta1[a1-1] * 0.5 + Delta1[a1-1] * 0.5 * X1_bin[mask]

    if flavor_Y == 'expo':
        Y = np.exp(eta) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'sigmoid':
        Y = 10.0 / (1.0 + np.exp(-eta)) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'gamma':
        shape, scale = 2, 3
        Y = ((np.exp(shape * eta) * np.exp(-np.exp(eta) / scale)) /
             (math.gamma(shape) * scale**shape)) * 10 + np.random.normal(0, 0.5, n) + 0.1
    elif flavor_Y == 'lognormal':
        sigma = 0.5
        Y = np.exp(eta + np.random.normal(0, sigma, n))
    else:
        raise ValueError(f'Unknown flavor_Y: {flavor_Y}')

    Y = np.abs(Y)
    if flavor_Y == 'expo':
        Y = np.minimum(Y, np.percentile(Y, 99.95))

    return {'Y': Y, 'xb_delta_a': eta}


def gen_A3_simple(A2, Y2_obs, k3):
    """
    Generate stage 3 treatment for simplified three-stage DGP.

    Assignment rule based on observed Y2 value:
      threshold = 70th percentile of Y2_obs across the sample
      p_stay_i  = 0.7  if Y2_obs_i > threshold  (high outcome, stay on A2)
                = 0.5  otherwise
      P(A3 = A2_i)        = p_stay_i
      P(A3 = other arm j) = (1 - p_stay_i) / (k3 - 1)  for each j != A2_i

    Parameters
    ----------
    A2      : array (n,)   -- stage 2 treatment
    Y2_obs  : array (n,)   -- observed intermediate outcome Y2
    k3      : int          -- number of stage 3 treatment levels

    Returns
    -------
    A3 : array (n,)
    """
    n         = len(A2)
    threshold = np.percentile(Y2_obs, 70)
    p_stay    = np.where(Y2_obs > threshold, 0.7, 0.5)

    A3 = np.empty(n, dtype=int)
    for idx in range(n):
        probs          = np.full(k3, (1 - p_stay[idx]) / (k3 - 1))
        probs[A2[idx]] = p_stay[idx]
        A3[idx]        = np.random.choice(k3, p=probs)

    return A3


def gen_Y_final_simple(X1, A1, A2, A3, beta_Y1, delta1, Delta1,
                       delta2_scalar, Delta2_scalar,
                       delta3_scalar, Delta3_scalar,
                       flavor_Y):
    """
    Generate final outcome Y for simplified three-stage DGP.

    Model (attenuation: A1 at 0.25, A2 at 0.5, A3 at full):
      eta = X1_with_int @ beta_Y1
              + sum_{a1>0} I(A1=a1) * (delta1[a1-1]*0.25 + Delta1[a1-1]*0.25 * X1_bin)
              + delta2_scalar*0.5 * A2  +  Delta2_scalar*0.5 * A2 * X1_bin
              + delta3_scalar * A3  +  Delta3_scalar * A3 * X1_bin
      Y   = f(eta) + epsilon,   epsilon ~ N(0, 0.5)

    Attenuation rule: effects decay by 0.5 for each additional stage of distance.
      A1 introduced at stage 1 → appears in Y (stage 3 outcome) at factor 0.5^2 = 0.25
      A2 introduced at stage 2 → appears in Y (stage 3 outcome) at factor 0.5^1 = 0.5
      A3 introduced at stage 3 → appears in Y (stage 3 outcome) at factor 0.5^0 = 1.0

    Parameters
    ----------
    X1             : DataFrame (n, p1)
    A1             : array (n,)
    A2             : array (n,)
    A3             : array (n,)
    beta_Y1        : array (p1+1,)
    delta1         : array (k1-1,)
    Delta1         : array (k1-1,)
    delta2_scalar  : float           -- A2 main effect at full strength (halved internally)
    Delta2_scalar  : float           -- A2 x X1_bin at full strength (halved internally)
    delta3_scalar  : float           -- A3 main effect at full strength
    Delta3_scalar  : float           -- A3 x X1_bin at full strength
    flavor_Y       : str

    Returns
    -------
    dict with keys 'Y' and 'xb_delta_a'
    """
    n           = X1.shape[0]
    X1_with_int = np.column_stack([np.ones(n), X1.values])
    X1_bin      = X1.iloc[:, -1].values

    eta = (X1_with_int @ beta_Y1
           + delta2_scalar * 0.5 * A2 + Delta2_scalar * 0.5 * A2 * X1_bin
           + delta3_scalar * A3        + Delta3_scalar * A3 * X1_bin)

    # A1 direct effect on Y: quartered (0.5^2) relative to Y1
    for a1 in range(1, len(delta1) + 1):
        mask      = (A1 == a1)
        eta[mask] += delta1[a1-1] * 0.25 + Delta1[a1-1] * 0.25 * X1_bin[mask]

    if flavor_Y == 'expo':
        Y = np.exp(eta) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'sigmoid':
        Y = 10.0 / (1.0 + np.exp(-eta)) + np.random.normal(0, 0.5, n)
    elif flavor_Y == 'gamma':
        shape, scale = 2, 3
        Y = ((np.exp(shape * eta) * np.exp(-np.exp(eta) / scale)) /
             (math.gamma(shape) * scale**shape)) * 10 + np.random.normal(0, 0.5, n) + 0.1
    elif flavor_Y == 'lognormal':
        sigma = 0.5
        Y = np.exp(eta + np.random.normal(0, sigma, n))
    else:
        raise ValueError(f'Unknown flavor_Y: {flavor_Y}')

    Y = np.abs(Y)
    if flavor_Y == 'expo':
        Y = np.minimum(Y, np.percentile(Y, 99.95))

    return {'Y': Y, 'xb_delta_a': eta}