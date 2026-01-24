# --------------------------------------------
# Author: Raul
# Date: 2025-01-08
# Script: outcome_models.py
# Note: This script fits models for estimating 
#       outcome Y
# --------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
import statsmodels.api as sm
from Y_nn_tuning import Y_model_nn
from get_diff import get_diff

def estimate_Y_ols(dat, pscores_df, k):
    """
    Estimate outcome models using OLS regression
    
    Parameters:
    - dat: full dataset
    - pscores_df: propensity scores DataFrame
    - k: number of treatment levels
    
    Returns:
    - Dictionary of results for each treatment comparison
    """
    Y = dat['Y'].values
    
    # Compute pairwise comparisons
    m = list(combinations(range(k), 2))
    
    def get_d_ij(comparison):
        j, i = comparison  # j < i
        
        # Get propensity scores for treatment i
        pi_hat_i = pscores_df.iloc[:, i].values if k > 2 else pscores_df.iloc[:, 0].values
        
        # Create binary treatment indicator
        A_binary = np.where(dat['A'] == i, 1, 
                           np.where(dat['A'] == j, 0, 99))
        
        delta_i = (A_binary == 1).astype(int)
        delta_j = (A_binary == 0).astype(int)
        
        # Fit outcome models
        # Model for treatment i
        dat_i = dat[A_binary == 1].drop('A', axis=1)
        g_i = LinearRegression()
        g_i.fit(dat_i.drop('Y', axis=1), dat_i['Y'])
        ghat_i = g_i.predict(dat.drop(['Y', 'A'], axis=1))
        
        # Model for treatment j
        dat_j = dat[A_binary == 0].drop('A', axis=1)
        g_j = LinearRegression()
        g_j.fit(dat_j.drop('Y', axis=1), dat_j['Y'])
        ghat_j = g_j.predict(dat.drop(['Y', 'A'], axis=1))
        
        # Compute difference
        d_ij = get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
        
        print(f"  logit-ols est diff means [a={j} vs. a={i}]={d_ij['diff_means']:.3f}")
        
        # Rename results
        result_names = {
            'diff_means': f'diff_means_{j}{i}',
            'muhat_B': f'muhat_{i}',
            'muhat_A': f'muhat_{j}',
            'diff_var': f'diff_var_{j}{i}',
            'pval': f'pval_{j}{i}'
        }
        
        renamed_result = {result_names[k]: v for k, v in d_ij.items()}
        
        return [renamed_result, g_i, g_j, ghat_j, ghat_i]
    
    # Apply to all comparisons
    results = {}
    for comp in m:
        j, i = comp
        results[f'A_{j}{i}'] = get_d_ij(comp)
    
    return results

def estimate_Y_expo(dat, pscores_df, k, link="log"):
    """
    Estimate outcome models using Gaussian GLM with log link

    Parameters:
    - dat: full dataset
    - pscores_df: propensity scores DataFrame
    - k: number of treatment levels
    - link: link function ("log" or "identity")

    Returns:
    - Dictionary of results for each treatment comparison
    """
    Y = dat['Y'].values

    # Select link function
    if link == "log":
        link_func = sm.families.links.Log()
    else:
        link_func = sm.families.links.Identity()

    # Compute pairwise comparisons
    m = list(combinations(range(k), 2))

    def get_d_ij(comparison):
        j, i = comparison  # j < i

        # Get propensity scores for treatment i
        pi_hat_i = pscores_df.iloc[:, i].values if k > 2 else pscores_df.iloc[:, 0].values

        # Create binary treatment indicator
        A_binary = np.where(dat['A'] == i, 1,
                           np.where(dat['A'] == j, 0, 99))

        delta_i = (A_binary == 1).astype(int)
        delta_j = (A_binary == 0).astype(int)

        # Prepare data for statsmodels (add constant for intercept)
        X_full = sm.add_constant(dat.drop(['Y', 'A'], axis=1))

        # Model for treatment i
        dat_i = dat[A_binary == 1]
        X_i = sm.add_constant(dat_i.drop(['Y', 'A'], axis=1))
        y_i = dat_i['Y']
        g_i = sm.GLM(y_i, X_i, family=sm.families.Gaussian(link_func)).fit()
        ghat_i = g_i.predict(X_full)

        # Model for treatment j
        dat_j = dat[A_binary == 0]
        X_j = sm.add_constant(dat_j.drop(['Y', 'A'], axis=1))
        y_j = dat_j['Y']
        g_j = sm.GLM(y_j, X_j, family=sm.families.Gaussian(link_func)).fit()
        ghat_j = g_j.predict(X_full)

        # Compute difference
        d_ij = get_diff(ghat_i.values, delta_i, ghat_j.values, delta_j, pi_hat_i, Y)

        print(f"  logit-expo est diff means [a={j} vs. a={i}]={d_ij['diff_means']:.3f}")

        # Rename results
        result_names = {
            'diff_means': f'diff_means_{j}{i}',
            'muhat_B': f'muhat_{i}',
            'muhat_A': f'muhat_{j}',
            'diff_var': f'diff_var_{j}{i}',
            'pval': f'pval_{j}{i}'
        }

        renamed_result = {result_names[k]: v for k, v in d_ij.items()}

        return [renamed_result, g_i, g_j, ghat_j, ghat_i]

    # Apply to all comparisons
    results = {}
    for comp in m:
        j, i = comp
        results[f'A_{j}{i}'] = get_d_ij(comp)

    return results

def estimate_Y_lognormal(dat, pscores_df, k, bias_correction=True, sigma2=None):
    """
    Estimate outcome models using lognormal regression

    Fits OLS on log(Y), then transforms predictions back via exp().
    Since Y ~ Lognormal(μ, σ²) implies log(Y) ~ Normal(μ, σ²),
    we model log(Y) = Xβ + ε where ε ~ Normal(0, σ²).

    Parameters:
    - dat: full dataset (Y must be positive)
    - pscores_df: propensity scores DataFrame
    - k: number of treatment levels
    - bias_correction: if True, applies exp(σ²/2) correction for E[Y]
                       (otherwise predicts median)
    - sigma2: if provided, uses this value for σ² instead of estimating
              (useful for simulation studies where true σ² is known)

    Returns:
    - Dictionary of results for each treatment comparison
    """
    Y = dat['Y'].values

    # Estimate pooled σ² from full data (log(Y) ~ X + A)
    if sigma2 is None and bias_correction:
        X_pooled = sm.add_constant(dat.drop(['Y', 'A'], axis=1).copy())
       #X_pooled['A'] = dat['A'].values
        log_y_pooled = np.log(dat['Y'])
        pooled_model = sm.OLS(log_y_pooled, X_pooled).fit()
        sigma2_pooled = pooled_model.mse_resid
    else:
        sigma2_pooled = sigma2

    # Compute pairwise comparisons
    m = list(combinations(range(k), 2))

    def get_d_ij(comparison):
        j, i = comparison  # j < i

        # Get propensity scores for treatment i
        pi_hat_i = pscores_df.iloc[:, i].values if k > 2 else pscores_df.iloc[:, 0].values

        # Create binary treatment indicator
        A_binary = np.where(dat['A'] == i, 1,
                           np.where(dat['A'] == j, 0, 99))

        delta_i = (A_binary == 1).astype(int)
        delta_j = (A_binary == 0).astype(int)

        # Prepare data for statsmodels (add constant for intercept)
        X_full = sm.add_constant(dat.drop(['Y', 'A'], axis=1))

        # Model for treatment i: fit on log(Y)
        dat_i = dat[A_binary == 1]
        X_i = sm.add_constant(dat_i.drop(['Y', 'A'], axis=1))
        log_y_i = np.log(dat_i['Y'])
        g_i = sm.OLS(log_y_i, X_i).fit()
        log_ghat_i = g_i.predict(X_full)
        # Transform back to original scale using pooled σ²
        if bias_correction:
            ghat_i = np.exp(log_ghat_i + sigma2_pooled / 2)
        else:
            ghat_i = np.exp(log_ghat_i)

        # Model for treatment j: fit on log(Y)
        dat_j = dat[A_binary == 0]
        X_j = sm.add_constant(dat_j.drop(['Y', 'A'], axis=1))
        log_y_j = np.log(dat_j['Y'])
        g_j = sm.OLS(log_y_j, X_j).fit()
        log_ghat_j = g_j.predict(X_full)
        # Transform back to original scale using pooled σ²
        if bias_correction:
            ghat_j = np.exp(log_ghat_j + sigma2_pooled / 2)
        else:
            ghat_j = np.exp(log_ghat_j)

        # Compute difference
        d_ij = get_diff(ghat_i.values, delta_i, ghat_j.values, delta_j, pi_hat_i, Y)

        print(f"  logit-lognormal est diff means [a={j} vs. a={i}]={d_ij['diff_means']:.3f}")

        # Rename results
        result_names = {
            'diff_means': f'diff_means_{j}{i}',
            'muhat_B': f'muhat_{i}',
            'muhat_A': f'muhat_{j}',
            'diff_var': f'diff_var_{j}{i}',
            'pval': f'pval_{j}{i}'
        }

        renamed_result = {result_names[k]: v for k, v in d_ij.items()}

        return [renamed_result, g_i, g_j, ghat_j, ghat_i]

    # Apply to all comparisons
    results = {}
    for comp in m:
        j, i = comp
        results[f'A_{j}{i}'] = get_d_ij(comp)

    return results

def estimate_Y_nn(dat, pscores_df, hidunits, eps, penals, k, verbose=False):
    """
    Estimate outcome models using neural networks
    
    Parameters:
    - dat: full dataset
    - pscores_df: propensity scores DataFrame
    - hidunits: hidden units for NN
    - eps: epochs for NN
    - penals: regularization parameters
    - k: number of treatment levels
    - verbose: print details
    
    Returns:
    - Dictionary of results for each treatment comparison
    """
    Y = dat['Y'].values
    
    # Compute pairwise comparisons
    m = list(combinations(range(k), 2))
    
    def get_d_ij(comparison):
        j, i = comparison  # j < i
        
        # Get propensity scores for treatment i
        pi_hat_i = pscores_df.iloc[:, i].values if k > 2 else pscores_df.iloc[:, 0].values
        
        # Create binary treatment indicator
        A_binary = np.where(dat['A'] == i, 1, 
                           np.where(dat['A'] == j, 0, 99))
        
        delta_i = (A_binary == 1).astype(int)
        delta_j = (A_binary == 0).astype(int)
        
        # Fit neural network models
        # Model for treatment i
        dat_i = dat[A_binary == 1].drop('A', axis=1)
        g_i = Y_model_nn(dat=dat_i, hidunits=hidunits, eps=eps, 
                        penals=penals, verbose=verbose)
        ghat_i = g_i.predict(dat.drop(['Y', 'A'], axis=1))
        
        # Model for treatment j
        dat_j = dat[A_binary == 0].drop('A', axis=1)
        g_j = Y_model_nn(dat=dat_j, hidunits=hidunits, eps=eps, 
                        penals=penals, verbose=verbose)
        ghat_j = g_j.predict(dat.drop(['Y', 'A'], axis=1))
        
        # Compute difference
        d_ij = get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
        
        print(f"  NN est diff means [a={j} vs. a={i}]={d_ij['diff_means']:.3f}")
        
        # Rename results
        result_names = {
            'diff_means': f'diff_means_{j}{i}',
            'muhat_B': f'muhat_{i}',
            'muhat_A': f'muhat_{j}',
            'diff_var': f'diff_var_{j}{i}',
            'pval': f'pval_{j}{i}'
        }
        
        renamed_result = {result_names[k]: v for k, v in d_ij.items()}
        
        return [renamed_result, g_i, g_j, ghat_j, ghat_i]
    
    # Apply to all comparisons
    results = {}
    for comp in m:
        j, i = comp
        results[f'A_{j}{i}'] = get_d_ij(comp)
    
    return results