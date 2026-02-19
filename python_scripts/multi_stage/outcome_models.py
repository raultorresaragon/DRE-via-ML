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
        n_params_i = g_i.coef_.size + 1  # coefficients + intercept
        
        # Model for treatment j
        dat_j = dat[A_binary == 0].drop('A', axis=1)
        g_j = LinearRegression()
        g_j.fit(dat_j.drop('Y', axis=1), dat_j['Y'])
        ghat_j = g_j.predict(dat.drop(['Y', 'A'], axis=1))
        n_params_j = g_j.coef_.size + 1  # coefficients + intercept
        
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
        renamed_result['n_params'] = n_params_i + n_params_j
        
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
        n_params_i = len(g_i.params)

        # Model for treatment j
        dat_j = dat[A_binary == 0]
        X_j = sm.add_constant(dat_j.drop(['Y', 'A'], axis=1))
        y_j = dat_j['Y']
        g_j = sm.GLM(y_j, X_j, family=sm.families.Gaussian(link_func)).fit()
        ghat_j = g_j.predict(X_full)
        n_params_j = len(g_j.params)

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
        renamed_result['n_params'] = n_params_i + n_params_j

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
        n_params_i = len(g_i.params)
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
        n_params_j = len(g_j.params)
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
        renamed_result['n_params'] = n_params_i + n_params_j

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
        mlp_i = g_i.named_steps['mlp']
        n_params_i = sum(coef.size for coef in mlp_i.coefs_) + sum(intercept.size for intercept in mlp_i.intercepts_)
        
        # Model for treatment j
        dat_j = dat[A_binary == 0].drop('A', axis=1)
        g_j = Y_model_nn(dat=dat_j, hidunits=hidunits, eps=eps, 
                        penals=penals, verbose=verbose)
        ghat_j = g_j.predict(dat.drop(['Y', 'A'], axis=1))
        mlp_j = g_j.named_steps['mlp']
        n_params_j = sum(coef.size for coef in mlp_j.coefs_) + sum(intercept.size for intercept in mlp_j.intercepts_)
        
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
        renamed_result['n_params'] = n_params_i + n_params_j
        
        return [renamed_result, g_i, g_j, ghat_j, ghat_i]
    
    # Apply to all comparisons
    results = {}
    for comp in m:
        j, i = comp
        results[f'A_{j}{i}'] = get_d_ij(comp)
    
    return results


# ========================================
# TWO-STAGE Q-LEARNING FUNCTIONS
# ========================================

def estimate_Q2_models(dat, pscores_df, k2, model_type='ols', **kwargs):
    """
    Estimate stage 2 Q-functions: Q2(X1, A1, X2, A2) = E[Y | X1, A1, X2, A2]
    
    Parameters:
    - dat: full dataset with columns [X1, A1, X2, A2, Y]
    - pscores_df: propensity scores for A2
    - k2: number of stage 2 treatment levels
    - model_type: 'ols', 'expo', 'lognormal', or 'nn'
    - **kwargs: additional arguments for model fitting
    
    Returns:
    - Dictionary with Q2 models for each A2 level
    """
    # Identify column names
    X1_cols = [c for c in dat.columns if c.startswith('X') and not c.startswith('X2_')]
    X2_cols = [c for c in dat.columns if c.startswith('X2_')]
    
    # Create feature matrix: [X1, A1, X2]
    features = X1_cols + ['A1'] + X2_cols
    X_full = dat[features]
    Y = dat['Y'].values
    A2 = dat['A2'].values
    
    # Fit Q2 model for each treatment level
    Q2_models = {}
    Q2_predictions = np.zeros((len(dat), k2))
    
    for a2 in range(k2):
        # Subset to individuals who received A2 = a2
        mask = (A2 == a2)
        X_a2 = X_full[mask]
        Y_a2 = Y[mask]
        
        # Fit model
        if model_type == 'ols':
            model = LinearRegression()
            model.fit(X_a2, Y_a2)
            Q2_predictions[:, a2] = model.predict(X_full)
            n_params = X_a2.shape[1] + 1
            
        elif model_type == 'expo':
            X_a2_sm = sm.add_constant(X_a2)
            X_full_sm = sm.add_constant(X_full)
            model = sm.GLM(Y_a2, X_a2_sm, family=sm.families.Gaussian(sm.families.links.Log())).fit()
            Q2_predictions[:, a2] = model.predict(X_full_sm)
            n_params = len(model.params)
            
        elif model_type == 'lognormal':
            X_a2_sm = sm.add_constant(X_a2)
            X_full_sm = sm.add_constant(X_full)
            log_Y_a2 = np.log(Y_a2)
            model = sm.OLS(log_Y_a2, X_a2_sm).fit()
            log_pred = model.predict(X_full_sm)
            sigma2 = model.mse_resid
            Q2_predictions[:, a2] = np.exp(log_pred + sigma2 / 2)
            n_params = len(model.params)
            
        elif model_type == 'nn':
            # Create temporary dataset for NN
            dat_a2 = pd.concat([pd.DataFrame(X_a2).reset_index(drop=True), 
                               pd.Series(Y_a2, name='Y').reset_index(drop=True)], axis=1)
            model = Y_model_nn(dat=dat_a2, **kwargs)
            Q2_predictions[:, a2] = model.predict(X_full)
            mlp = model.named_steps['mlp']
            n_params = sum(coef.size for coef in mlp.coefs_) + sum(intercept.size for intercept in mlp.intercepts_)
        
        Q2_models[a2] = {'model': model, 'n_params': n_params}
    
    return {'Q2_models': Q2_models, 'Q2_predictions': Q2_predictions}


def compute_pseudo_outcome(Q2_predictions):
    """
    Compute pseudo-outcome for stage 1: Ỹ = max_{a2} Q2(X1, A1, X2, a2)
    
    Parameters:
    - Q2_predictions: array of shape (n, k2) with Q2 predictions for each A2
    
    Returns:
    - Y_tilde: pseudo-outcome (max Q2 value for each individual)
    - optimal_A2: optimal stage 2 treatment for each individual
    """
    Y_tilde = np.max(Q2_predictions, axis=1)
    optimal_A2 = np.argmax(Q2_predictions, axis=1)
    
    return {'Y_tilde': Y_tilde, 'optimal_A2': optimal_A2}


def estimate_Q1_models(dat, Y_tilde, pscores_df, k1, model_type='ols', **kwargs):
    """
    Estimate stage 1 Q-functions: Q1(X1, A1) = E[Ỹ | X1, A1]
    
    Parameters:
    - dat: dataset with X1 and A1
    - Y_tilde: pseudo-outcome from stage 2
    - pscores_df: propensity scores for A1
    - k1: number of stage 1 treatment levels
    - model_type: 'ols', 'expo', 'lognormal', or 'nn'
    - **kwargs: additional arguments for model fitting
    
    Returns:
    - Dictionary with Q1 models for each A1 level
    """
    # Identify X1 columns
    X1_cols = [c for c in dat.columns if c.startswith('X') and not c.startswith('X2_')]
    X1 = dat[X1_cols]
    A1 = dat['A1'].values
    
    # Fit Q1 model for each treatment level
    Q1_models = {}
    Q1_predictions = np.zeros((len(dat), k1))
    
    for a1 in range(k1):
        # Subset to individuals who received A1 = a1
        mask = (A1 == a1)
        X1_a1 = X1[mask]
        Y_tilde_a1 = Y_tilde[mask]
        
        # Fit model
        if model_type == 'ols':
            model = LinearRegression()
            model.fit(X1_a1, Y_tilde_a1)
            Q1_predictions[:, a1] = model.predict(X1)
            n_params = X1_a1.shape[1] + 1
            
        elif model_type == 'expo':
            X1_a1_sm = sm.add_constant(X1_a1)
            X1_sm = sm.add_constant(X1)
            model = sm.GLM(Y_tilde_a1, X1_a1_sm, family=sm.families.Gaussian(sm.families.links.Log())).fit()
            Q1_predictions[:, a1] = model.predict(X1_sm)
            n_params = len(model.params)
            
        elif model_type == 'lognormal':
            X1_a1_sm = sm.add_constant(X1_a1)
            X1_sm = sm.add_constant(X1)
            log_Y_tilde_a1 = np.log(Y_tilde_a1)
            model = sm.OLS(log_Y_tilde_a1, X1_a1_sm).fit()
            log_pred = model.predict(X1_sm)
            sigma2 = model.mse_resid
            Q1_predictions[:, a1] = np.exp(log_pred + sigma2 / 2)
            n_params = len(model.params)
            
        elif model_type == 'nn':
            # Create temporary dataset for NN
            dat_a1 = pd.concat([pd.DataFrame(X1_a1).reset_index(drop=True), 
                               pd.Series(Y_tilde_a1, name='Y').reset_index(drop=True)], axis=1)
            model = Y_model_nn(dat=dat_a1, **kwargs)
            Q1_predictions[:, a1] = model.predict(X1)
            mlp = model.named_steps['mlp']
            n_params = sum(coef.size for coef in mlp.coefs_) + sum(intercept.size for intercept in mlp.intercepts_)
        
        Q1_models[a1] = {'model': model, 'n_params': n_params}
    
    return {'Q1_models': Q1_models, 'Q1_predictions': Q1_predictions}


def estimate_optimal_regime_two_stage(dat, pscores_A1, pscores_A2, k1, k2, 
                                      model_type='ols', **kwargs):
    """
    Estimate optimal two-stage regime using Q-learning
    
    Parameters:
    - dat: full dataset with [X1, A1, X2, A2, Y]
    - pscores_A1: propensity scores for stage 1
    - pscores_A2: propensity scores for stage 2
    - k1: number of stage 1 treatment levels
    - k2: number of stage 2 treatment levels
    - model_type: 'ols', 'expo', 'lognormal', or 'nn'
    - **kwargs: additional arguments for model fitting
    
    Returns:
    - Dictionary with Q1, Q2 models and optimal regimes
    """
    print(f"\n  Estimating Stage 2 Q-functions ({model_type})...")
    Q2_result = estimate_Q2_models(dat, pscores_A2, k2, model_type, **kwargs)
    
    print(f"  Computing pseudo-outcome...")
    pseudo_result = compute_pseudo_outcome(Q2_result['Q2_predictions'])
    
    print(f"  Estimating Stage 1 Q-functions ({model_type})...")
    Q1_result = estimate_Q1_models(dat, pseudo_result['Y_tilde'], pscores_A1, k1, 
                                   model_type, **kwargs)
    
    # Optimal stage 1 treatment
    optimal_A1 = np.argmax(Q1_result['Q1_predictions'], axis=1)
    
    return {
        'Q2_models': Q2_result['Q2_models'],
        'Q2_predictions': Q2_result['Q2_predictions'],
        'Q1_models': Q1_result['Q1_models'],
        'Q1_predictions': Q1_result['Q1_predictions'],
        'optimal_A1': optimal_A1,
        'optimal_A2': pseudo_result['optimal_A2'],
        'Y_tilde': pseudo_result['Y_tilde']
    }
