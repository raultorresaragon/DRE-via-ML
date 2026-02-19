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
    
    X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
    
    # Convert last p_bin columns to binary
    if p_bin > 0:
        for j in range(p-p_bin, p):
            col_name = f'X{j+1}'
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
def gen_Y(gamma, X, A, beta_Y, flavor_Y="expo"):
    """
    Generate outcomes
    
    Parameters:
    - gamma: treatment effects
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
    xb_gamma_a = X_with_intercept @ beta_Y
    if A_mat.shape[1] > 0 and len(gamma) > 0:
        xb_gamma_a += A_mat @ gamma
    
    # Apply functional form
    if flavor_Y == "expo":
        Y = np.exp(xb_gamma_a) + np.random.normal(0, 0.5, n)
        
    elif flavor_Y == "sigmoid":
        Y = 1/(1 + np.exp(-xb_gamma_a)) * 10 + np.random.normal(0, 0.5, n)
        
    elif flavor_Y == "gamma":
        shape = 2
        scale = 3
        Y = (np.exp(shape * xb_gamma_a) * np.exp(-np.exp(xb_gamma_a)/scale)) / \
            (math.gamma(shape) * scale**shape) * 10 + np.random.normal(0, 0.5, n) + 0.1
            
    elif flavor_Y == "lognormal":
        # If log(Y) ~ N(μ, σ²), then Y ~ Lognormal
        sigma = 0.5
        Y = np.exp(xb_gamma_a + np.random.normal(0, sigma, n))

        # bell-curve transform 
        #Y = (1 / (np.exp(xb_gamma_a) * np.sqrt(2 * np.pi))) * \
        #    np.exp(-0.5 * xb_gamma_a**2) * 10 + np.random.normal(0, 0.5, n) 
    
    # Ensure positive outcomes
    Y = np.abs(Y)
    
    # Handle extreme values for exponential
    if flavor_Y == "expo":
        threshold = np.percentile(Y, 99.95)
        Y = np.minimum(Y, threshold)
    
    return {'Y': Y, 'xb_gamma_a': xb_gamma_a}


# ========================================
# TWO-STAGE FUNCTIONS
# ========================================

# --------------------------
# Generate X2 (stage 2 covariates)
# --------------------------
def gen_X2(X1, A1, p2, gamma1_X2, beta_X2, rho=0.5, p_bin=1):
    """
    Generate stage 2 covariates that depend on stage 1 history
    
    Parameters:
    - X1: stage 1 covariates (DataFrame)
    - A1: stage 1 treatment (array)
    - p2: number of stage 2 covariates
    - gamma1_X2: treatment effects of A1 on X2 (array of length k1-1)
    - beta_X2: coefficients for X1 effect on X2 (array of length p1+1)
    - rho: correlation parameter for X2
    - p_bin: number of binary covariates in X2
    
    Returns:
    - X2: DataFrame with stage 2 covariates
    """
    n = X1.shape[0]
    X1_with_intercept = np.column_stack([np.ones(n), X1.values])
    
    # Create treatment indicators for A1 (excluding baseline A1=0)
    unique_A1 = np.unique(A1)
    A1_mat = np.column_stack([np.where(A1 == a, 1, 0) for a in unique_A1[1:]])
    
    # Mean of X2 depends on X1 and A1
    X2_mean = X1_with_intercept @ beta_X2
    if A1_mat.shape[1] > 0 and len(gamma1_X2) > 0:
        X2_mean += A1_mat @ gamma1_X2
    
    # Generate correlated X2 around this mean
    Sigma = np.array([[rho**abs(i-j) for j in range(p2)] for i in range(p2)])
    
    # Generate X2 with mean shifted by treatment effect
    X2 = np.zeros((n, p2))
    for i in range(n):
        X2[i, :] = multivariate_normal.rvs(mean=np.full(p2, X2_mean[i]), cov=Sigma)
    
    X2_df = pd.DataFrame(X2, columns=[f'X2_{i+1}' for i in range(p2)])
    
    # Convert last p_bin columns to binary
    if p_bin > 0:
        for j in range(p2-p_bin, p2):
            col_name = f'X2_{j+1}'
            X2_df[col_name] = (X2_df[col_name] > X2_df[col_name].median()).astype(int)
    
    return X2_df


# ------------
# Generate Y (two-stage version)
# ------------
def gen_Y_two_stage(gamma1_Y, gamma2_Y, X1, A1, X2, A2, beta_Y, flavor_Y="expo"):
    """
    Generate outcomes for two-stage setting
    
    Parameters:
    - gamma1_Y: stage 1 treatment effects on Y (array of length k1-1)
    - gamma2_Y: stage 2 treatment effects on Y (array of length k2-1)
    - X1: stage 1 covariates
    - A1: stage 1 treatment
    - X2: stage 2 covariates
    - A2: stage 2 treatment
    - beta_Y: outcome model coefficients [beta_X1, beta_X2]
    - flavor_Y: functional form for outcome
    
    Returns:
    - Dictionary with Y and linear predictor
    """
    n = X1.shape[0]
    p1 = X1.shape[1]
    
    # Combine X1 and X2
    X_combined = np.column_stack([np.ones(n), X1.values, X2.values])
    
    # Create treatment indicators for A1 (excluding baseline)
    unique_A1 = np.unique(A1)
    A1_mat = np.column_stack([np.where(A1 == a, 1, 0) for a in unique_A1[1:]])
    
    # Create treatment indicators for A2 (excluding baseline)
    unique_A2 = np.unique(A2)
    A2_mat = np.column_stack([np.where(A2 == a, 1, 0) for a in unique_A2[1:]])
    
    # Linear predictor
    xb_gamma_a = X_combined @ beta_Y
    
    # Add stage 1 treatment effects
    if A1_mat.shape[1] > 0 and len(gamma1_Y) > 0:
        xb_gamma_a += A1_mat @ gamma1_Y
    
    # Add stage 2 treatment effects
    if A2_mat.shape[1] > 0 and len(gamma2_Y) > 0:
        xb_gamma_a += A2_mat @ gamma2_Y
    
    # Apply functional form (same as single-stage)
    if flavor_Y == "expo":
        Y = np.exp(xb_gamma_a) + np.random.normal(0, 0.5, n)
        
    elif flavor_Y == "sigmoid":
        Y = 1/(1 + np.exp(-xb_gamma_a)) * 10 + np.random.normal(0, 0.5, n)
        
    elif flavor_Y == "gamma":
        shape = 2
        scale = 3
        Y = (np.exp(shape * xb_gamma_a) * np.exp(-np.exp(xb_gamma_a)/scale)) / \
            (math.gamma(shape) * scale**shape) * 10 + np.random.normal(0, 0.5, n) + 0.1
            
    elif flavor_Y == "lognormal":
        sigma = 0.5
        Y = np.exp(xb_gamma_a + np.random.normal(0, sigma, n))
    
    # Ensure positive outcomes
    Y = np.abs(Y)
    
    # Handle extreme values for exponential
    if flavor_Y == "expo":
        threshold = np.percentile(Y, 99.95)
        Y = np.minimum(Y, threshold)
    
    return {'Y': Y, 'xb_gamma_a': xb_gamma_a}