# --------------------------------------------
# Author: Raul
# Date: 2025-01-08
# Script: pscores_models.py
# Note: This script fits models for estimating 
#       propensity score for k>2
# --------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from A_nn_tuning import A_model_nn

def estimate_A_nn(X, dat, k, hidunits, eps, penals, verbose=False):
    """
    Estimate propensity scores using neural network
    
    Parameters:
    - X: covariate matrix
    - dat: full dataset
    - k: number of treatment levels
    - hidunits: hidden units for NN
    - eps: epochs for NN
    - penals: regularization parameters
    - verbose: print details
    
    Returns:
    - Dictionary with propensity scores and fitted model
    """
    print("   ...fitting 1 hidden-layer neural networks")
    
    # Prepare data (exclude Y if present)
    dat_A = dat.drop('Y', axis=1, errors='ignore')
    
    # Fit neural network
    H_nn = A_model_nn(dat=dat_A, hidunits=hidunits, eps=eps, 
                      penals=penals, verbose=verbose)
    
    # Get propensity scores
    X_pred = dat_A.drop('A', axis=1)
    pscores = H_nn.predict_proba(X_pred)
    
    # Create column names
    if k == 2:
        pscores_names = ['pscores_1']
        pscores = pscores[:, 1].reshape(-1, 1)  # Only probability of class 1
    else:
        pscores_names = [f'pscores_{i}' for i in range(k)]
    
    pscores_df = pd.DataFrame(pscores, columns=pscores_names)
    
    return {'pscores': pscores_df, 'H_nn': H_nn}

def estimate_A_logit(X, dat, k, verbose=False):
    """
    Estimate propensity scores using logistic regression
    
    Parameters:
    - X: covariate matrix
    - dat: full dataset
    - k: number of treatment levels
    - verbose: print details
    
    Returns:
    - Dictionary with propensity scores and fitted model
    """
    print("   ...fitting logistic model")
    
    # Prepare data
    dat_A = dat.drop('Y', axis=1, errors='ignore')
    X_pred = dat_A.drop('A', axis=1)
    y = dat_A['A']
    
    # Fit logistic regression
    if k == 2:
        H_logit = LogisticRegression(random_state=42, max_iter=1000)
    else:
        H_logit = LogisticRegression(random_state=42, max_iter=1000, 
                                   multi_class='multinomial', solver='lbfgs')
    
    H_logit.fit(X_pred, y)
    
    # Get propensity scores
    pscores = H_logit.predict_proba(X_pred)
    
    # Create column names
    if k == 2:
        pscores_names = ['pscores_1']
        pscores = pscores[:, 1].reshape(-1, 1)  # Only probability of class 1
    else:
        pscores_names = [f'pscores_{i}' for i in range(k)]
    
    pscores_df = pd.DataFrame(pscores, columns=pscores_names)
    
    return {'pscores': pscores_df, 'H_logits': H_logit}