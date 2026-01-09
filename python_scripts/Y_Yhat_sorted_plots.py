# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: Y_Yhat_sorted_plots.py
# Date: 2025-01-08
# Note: This script creates plots comparing predicted vs actual outcomes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_predicted_A_Y(beta_A, beta_Y, dat, fit_Y_nn, fit_Y_param, gamma, 
                      fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k, 
                      save=False, root="./"):
    """
    Plot predicted vs actual outcomes and treatment assignments
    
    Parameters:
    - beta_A: treatment model coefficients
    - beta_Y: outcome model coefficients  
    - dat: dataset
    - fit_Y_nn: neural network outcome models
    - fit_Y_param: parametric outcome models
    - gamma: treatment effects
    - fit_A_nn: neural network propensity models
    - fit_A_logit: logistic propensity models
    - A_flavor: treatment model type
    - Y_flavor: outcome model type
    - ds: dataset number
    - k: number of treatments
    - save: whether to save plots
    - root: root directory for saving
    """
    
    # Colors for different treatments
    colors = ['black', 'darkred', 'green', 'blue', 'skyblue']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'k={k} flavor:{A_flavor}-{Y_flavor} N={len(dat)} Dataset {ds}', 
                fontsize=14)
    
    # Plot 1: Actual Y vs predicted Y (NN)
    if fit_Y_nn:
        # Get predictions from first comparison
        first_key = list(fit_Y_nn.keys())[0]
        if len(fit_Y_nn[first_key]) > 4:  # Check if ghat_i exists
            y_pred_nn = fit_Y_nn[first_key][4]  # ghat_i
            axes[0,0].scatter(dat['Y'], y_pred_nn, c=[colors[a] for a in dat['A']], alpha=0.6)
            axes[0,0].plot([dat['Y'].min(), dat['Y'].max()], 
                          [dat['Y'].min(), dat['Y'].max()], 'r--', alpha=0.8)
            axes[0,0].set_xlabel('Actual Y')
            axes[0,0].set_ylabel('Predicted Y (NN)')
            axes[0,0].set_title('Neural Network Predictions')
    
    # Plot 2: Actual Y vs predicted Y (Parametric)
    if fit_Y_param:
        first_key = list(fit_Y_param.keys())[0]
        if len(fit_Y_param[first_key]) > 4:
            y_pred_param = fit_Y_param[first_key][4]  # ghat_i
            axes[0,1].scatter(dat['Y'], y_pred_param, c=[colors[a] for a in dat['A']], alpha=0.6)
            axes[0,1].plot([dat['Y'].min(), dat['Y'].max()], 
                          [dat['Y'].min(), dat['Y'].max()], 'r--', alpha=0.8)
            axes[0,1].set_xlabel('Actual Y')
            axes[0,1].set_ylabel('Predicted Y (Parametric)')
            axes[0,1].set_title('Parametric Model Predictions')
    
    # Plot 3: Propensity scores (NN)
    if fit_A_nn and 'pscores' in fit_A_nn:
        pscores_nn = fit_A_nn['pscores']
        for i in range(min(k, pscores_nn.shape[1])):
            if i < len(colors):
                axes[1,0].hist(pscores_nn.iloc[:, i], alpha=0.5, 
                              color=colors[i], label=f'P(A={i})', bins=20)
        axes[1,0].set_xlabel('Propensity Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Propensity Scores (NN)')
        axes[1,0].legend()
    
    # Plot 4: Propensity scores (Logistic)
    if fit_A_logit and 'pscores' in fit_A_logit:
        pscores_logit = fit_A_logit['pscores']
        for i in range(min(k, pscores_logit.shape[1])):
            if i < len(colors):
                axes[1,1].hist(pscores_logit.iloc[:, i], alpha=0.5, 
                              color=colors[i], label=f'P(A={i})', bins=20)
        axes[1,1].set_xlabel('Propensity Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Propensity Scores (Logistic)')
        axes[1,1].legend()
    
    plt.tight_layout()
    
    if save:
        os.makedirs(f"{root}/images/predicted_plots", exist_ok=True)
        plt.savefig(f"{root}/images/predicted_plots/pred_k{k}_{A_flavor}{Y_flavor}_dset{ds}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()