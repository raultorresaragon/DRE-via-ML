# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: get_true_optimal_regime.py
# Date: 2026-02-19
# Note: Compute true optimal two-stage regime given data-generating parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def compute_true_Q2(X1, A1, X2, a2, gamma1_Y, gamma2_Y, beta_Y, flavor_Y="expo"):
    """
    Compute true Q2(X1, A1, X2, a2) = E[Y | X1, A1, X2, A2=a2]
    
    Parameters:
    - X1: stage 1 covariates (DataFrame or array)
    - A1: stage 1 treatment (array)
    - X2: stage 2 covariates (DataFrame or array)
    - a2: stage 2 treatment level (scalar)
    - gamma1_Y: stage 1 treatment effects
    - gamma2_Y: stage 2 treatment effects
    - beta_Y: covariate effects
    - flavor_Y: outcome functional form
    
    Returns:
    - True Q2 value for each individual
    """
    n = len(A1)
    
    # Combine covariates
    if isinstance(X1, pd.DataFrame):
        X1_vals = X1.values
    else:
        X1_vals = X1
    
    if isinstance(X2, pd.DataFrame):
        X2_vals = X2.values
    else:
        X2_vals = X2
    
    X_combined = np.column_stack([np.ones(n), X1_vals, X2_vals])
    
    # Linear predictor
    eta = X_combined @ beta_Y
    
    # Add stage 1 treatment effect
    for i, a1 in enumerate(np.unique(A1)):
        if a1 > 0:  # Exclude baseline
            mask = (A1 == a1)
            eta[mask] += gamma1_Y[a1 - 1]
    
    # Add stage 2 treatment effect
    if a2 > 0:  # Exclude baseline
        eta += gamma2_Y[a2 - 1]
    
    # Apply functional form (expected value)
    if flavor_Y == "expo":
        # E[exp(eta) + noise] = exp(eta) since E[noise] = 0
        Q2 = np.exp(eta)
        
    elif flavor_Y == "sigmoid":
        Q2 = 1/(1 + np.exp(-eta)) * 10
        
    elif flavor_Y == "lognormal":
        # E[exp(eta + sigma*Z)] = exp(eta + sigma^2/2) where Z ~ N(0,1)
        sigma = 0.5
        Q2 = np.exp(eta + sigma**2 / 2)
    
    else:
        # Default to linear
        Q2 = eta
    
    return Q2


def compute_true_optimal_A2(X1, A1, X2, k2, gamma1_Y, gamma2_Y, beta_Y, flavor_Y="expo"):
    """
    Compute true optimal stage 2 treatment for each individual
    
    Returns:
    - optimal_A2: optimal stage 2 treatment (array)
    - Q2_all: Q2 values for all treatments (n x k2 array)
    """
    n = len(A1)
    Q2_all = np.zeros((n, k2))
    
    # Compute Q2 for each possible A2
    for a2 in range(k2):
        Q2_all[:, a2] = compute_true_Q2(X1, A1, X2, a2, gamma1_Y, gamma2_Y, beta_Y, flavor_Y)
    
    # Optimal A2 is argmax Q2
    optimal_A2 = np.argmax(Q2_all, axis=1)
    
    return optimal_A2, Q2_all


def compute_true_Q1(X1, a1, k2, gamma1_X2, beta_X2, gamma1_Y, gamma2_Y, beta_Y, 
                    p2, rho, flavor_Y="expo", n_samples=1000):
    """
    Compute true Q1(X1, a1) = E[max_{a2} Q2(X1, a1, X2, a2) | X1, A1=a1]
    
    This requires integrating over the distribution of X2 | X1, A1=a1
    We approximate this with Monte Carlo sampling
    
    Parameters:
    - X1: stage 1 covariates (DataFrame or array)
    - a1: stage 1 treatment level (scalar)
    - k2: number of stage 2 treatments
    - gamma1_X2: effect of A1 on X2
    - beta_X2: effect of X1 on X2
    - gamma1_Y, gamma2_Y, beta_Y: outcome model parameters
    - p2: number of stage 2 covariates
    - rho: correlation for X2
    - flavor_Y: outcome functional form
    - n_samples: number of Monte Carlo samples
    
    Returns:
    - True Q1 value for each individual
    """
    n = len(X1) if isinstance(X1, pd.DataFrame) else X1.shape[0]
    
    if isinstance(X1, pd.DataFrame):
        X1_vals = X1.values
    else:
        X1_vals = X1
    
    X1_with_intercept = np.column_stack([np.ones(n), X1_vals])
    
    # Mean of X2 given X1 and A1=a1
    X2_mean = X1_with_intercept @ beta_X2
    if a1 > 0:  # Add treatment effect
        X2_mean += gamma1_X2[a1 - 1]
    
    # Covariance matrix for X2
    Sigma = np.array([[rho**abs(i-j) for j in range(p2)] for i in range(p2)])
    
    Q1_values = np.zeros(n)
    
    # For each individual, sample X2 and compute expected max Q2
    for i in range(n):
        max_Q2_samples = []
        
        for _ in range(n_samples):
            # Sample X2 | X1, A1=a1
            X2_sample = multivariate_normal.rvs(
                mean=np.full(p2, X2_mean[i]), 
                cov=Sigma
            ).reshape(1, -1)
            
            # Compute Q2 for all A2 values
            A1_sample = np.array([a1])
            Q2_values = np.zeros(k2)
            
            for a2 in range(k2):
                Q2_values[a2] = compute_true_Q2(
                    X1_vals[i:i+1], A1_sample, X2_sample, a2,
                    gamma1_Y, gamma2_Y, beta_Y, flavor_Y
                )[0]
            
            # Take max over A2
            max_Q2_samples.append(np.max(Q2_values))
        
        # Average over samples
        Q1_values[i] = np.mean(max_Q2_samples)
    
    return Q1_values


def compute_true_optimal_regime(X1, X2, A1, k1, k2, 
                                gamma1_X2, beta_X2,
                                gamma1_Y, gamma2_Y, beta_Y,
                                p2, rho, flavor_Y="expo",
                                n_samples=1000):
    """
    Compute true optimal two-stage regime
    
    Parameters:
    - X1: stage 1 covariates
    - X2: stage 2 covariates (observed)
    - A1: stage 1 treatment (observed)
    - k1, k2: number of treatments at each stage
    - gamma1_X2, beta_X2: X2 model parameters
    - gamma1_Y, gamma2_Y, beta_Y: outcome model parameters
    - p2: number of stage 2 covariates
    - rho: correlation for X2
    - flavor_Y: outcome functional form
    - n_samples: Monte Carlo samples for Q1 computation
    
    Returns:
    - Dictionary with true optimal regimes and Q-values
    """
    n = len(A1)
    
    print("\nComputing true optimal regime...")
    
    # Stage 2: Compute true optimal A2 given observed (X1, A1, X2)
    print("  Computing true optimal A2...")
    true_optimal_A2, true_Q2_all = compute_true_optimal_A2(
        X1, A1, X2, k2, gamma1_Y, gamma2_Y, beta_Y, flavor_Y
    )
    
    # Stage 1: Compute true Q1 for each possible A1
    print("  Computing true Q1 values (this may take a moment)...")
    true_Q1_all = np.zeros((n, k1))
    
    for a1 in range(k1):
        print(f"    A1={a1}...")
        true_Q1_all[:, a1] = compute_true_Q1(
            X1, a1, k2, gamma1_X2, beta_X2, gamma1_Y, gamma2_Y, beta_Y,
            p2, rho, flavor_Y, n_samples
        )
    
    # Optimal A1 is argmax Q1
    true_optimal_A1 = np.argmax(true_Q1_all, axis=1)
    
    print("  ✓ True optimal regime computed")
    
    return {
        'true_optimal_A1': true_optimal_A1,
        'true_optimal_A2': true_optimal_A2,
        'true_Q1_all': true_Q1_all,
        'true_Q2_all': true_Q2_all
    }


def evaluate_regime_accuracy(estimated_A1, estimated_A2, true_A1, true_A2):
    """
    Evaluate how well estimated regime matches true optimal regime
    
    Returns:
    - Dictionary with accuracy metrics
    """
    n = len(true_A1)
    
    # Stage-specific accuracy
    A1_accuracy = np.mean(estimated_A1 == true_A1)
    A2_accuracy = np.mean(estimated_A2 == true_A2)
    
    # Joint accuracy (both stages correct)
    joint_accuracy = np.mean((estimated_A1 == true_A1) & (estimated_A2 == true_A2))
    
    return {
        'A1_accuracy': A1_accuracy,
        'A2_accuracy': A2_accuracy,
        'joint_accuracy': joint_accuracy,
        'A1_agreement': np.sum(estimated_A1 == true_A1),
        'A2_agreement': np.sum(estimated_A2 == true_A2),
        'joint_agreement': np.sum((estimated_A1 == true_A1) & (estimated_A2 == true_A2)),
        'n': n
    }
