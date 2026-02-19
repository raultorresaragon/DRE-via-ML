# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: A_nn_tuning.py
# Date: 2025-01-08
# Note: This script deploys a function for fitting a Neural Net for 
#       a propensity score (A) model with hyperparameter tuning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def A_model_nn(dat, a_func="A~.", hidunits=[5, 20], eps=[100, 250], 
               penals=[0.001, 0.01], cvs=6, verbose=False):
    """
    Fit neural network for propensity score model with hyperparameter tuning
    
    Parameters:
    - dat: DataFrame with treatment and covariates
    - a_func: formula (not used in Python version, assumes A is target)
    - hidunits: range of hidden units to try
    - eps: range of epochs (max_iter) to try
    - penals: range of regularization parameters to try
    - cvs: number of cross-validation folds
    - verbose: whether to print tuning results
    
    Returns:
    - Fitted pipeline with best hyperparameters
    """
    
    # Prepare data
    X = dat.drop('A', axis=1)
    y = dat['A'].astype(int)
    
    # Create pipeline with scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, early_stopping=True, 
                             validation_fraction=0.2, n_iter_no_change=10))
    ])
    
    # Parameter grid
    param_grid = {
        'mlp__hidden_layer_sizes': [(h,) for h in hidunits],
        'mlp__max_iter': eps,
        'mlp__alpha': penals,
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__activation': ['relu', 'tanh']
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cvs, shuffle=True, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        pipe, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit
    grid_search.fit(X, y)
    
    if verbose:
        print("Best parameters:", grid_search.best_params_)
        print("Best CV score:", round(grid_search.best_score_, 4))
    
    return grid_search.best_estimator_