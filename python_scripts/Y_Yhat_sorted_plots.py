# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: Y_Yhat_sorted_plots.py
# Date: 2025-01-08
# Note: This script creates plots comparing predicted vs actual outcomes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def plot_predicted_A_Y(beta_A, beta_Y, dat, fit_Y_nn, fit_Y_expo, gamma, 
                      fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k, expo_or_ols,
                      save=True, blue=True, root=""):
    
    if save:
        plt.figure(figsize=(10, 5.1))
    
    Y = dat['Y']
    X = dat.filter(regex='^X')
    A = dat['A']
    
    # Colors
    if blue:
        mycols = [to_rgba(c, alpha=0.9) for c in ["black", "darkgray", "blue"]]
    else:
        mycols = [to_rgba(c, alpha=0.9) for c in ["black", "darkgray", "darkgray"]]
    
    xb_A = np.column_stack([np.ones(len(X)), X]) @ beta_A
    sample_idx = np.random.choice(len(Y), k*100, replace=False)
    
    # Calculate true propensity scores
    if A_flavor == "tanh":
        xb = np.column_stack([np.ones(len(X)), X]) @ beta_A
        raw_scores = pd.DataFrame(0.5 * (np.tanh(xb) + 1))
        raw_scores = pd.concat([raw_scores, pd.DataFrame(np.zeros((len(raw_scores), 2)))], axis=1)
        
        probs = pd.DataFrame()
        for i in range(xb.shape[1]):
            probs[f'class{i+1}'] = raw_scores.iloc[:, i] / (1 + raw_scores.drop(columns=raw_scores.columns[i]).sum(axis=1))
        
        if k == 2:
            sum_other = 1 - probs['class1']
        else:
            sum_other = 1 - probs.sum(axis=1)
        probs[f'class{k}'] = sum_other
        
        probs.columns = [f'true_pscores{i}' for i in (range(1, k+1) if k == 2 else range(k))]
        legposY = "lower right"
    
    elif A_flavor == "logit":
        exp_xb_A = np.exp(xb_A)
        denom = 1 + exp_xb_A.sum(axis=1)
        probs = pd.DataFrame(1/denom, columns=['true_pscores0'])
        for i in range(beta_A.shape[1]):
            probs[f'true_pscores{i+1}'] = exp_xb_A[:, i] / denom
        legposY = "upper left"
    
    dat = pd.concat([dat, probs], axis=1)
    
    # Add predicted propensity scores
    fit_A_logit_df = pd.DataFrame(fit_A_logit['pscores'])
    fit_A_logit_df.columns = [f"{col}_logit" for col in fit_A_logit_df.columns]
    fit_A_nn_df = pd.DataFrame(fit_A_nn['pscores'])
    fit_A_nn_df.columns = [f"{col}_nn" for col in fit_A_nn_df.columns]
    
    dat = pd.concat([dat, fit_A_logit_df, fit_A_nn_df], axis=1)
    
    if k == 2:
        dat['pscores_0_logit'] = 1 - dat['pscores_1_logit']
        dat['pscores_0_nn'] = 1 - dat['pscores_1_nn']
    
    # Add predicted Y values
    Yhat_nn = np.full(len(Y), np.nan)
    Yhat_expo = np.full(len(Y), np.nan)
    
    Yhat_nn[A == 0] = fit_Y_nn['A_01'][3][A == 0]
    Yhat_expo[A == 0] = fit_Y_expo['A_01'][3][A == 0]
    
    for i in range(1, k):
        mask = A == i
        Yhat_nn[mask] = fit_Y_nn[f'A_0{i}'][4][mask]
        Yhat_expo[mask] = fit_Y_expo[f'A_0{i}'][4][mask]
    
    dat['Yhat_nn'] = Yhat_nn
    dat['Yhat_expo'] = Yhat_expo
    
    dat_sample = dat.iloc[sample_idx]
    dat_pA = dat_sample.sample(100)
    
    # Create subplots
    fig, axes = plt.subplots(2, k, figsize=(10, 5.1))
    
    # Propensity score plots
    for i in range(k):
        ax = axes[0, i] if k > 1 else axes[0]
        true_col = f'true_pscores{i}'
        
        sorted_true = np.sort(dat_pA[true_col])
        ax.plot(sorted_true, color=mycols[0], linewidth=2, label='true' if i == 0 else "")
        
        order_idx = dat_pA[true_col].argsort()
        ax.scatter(range(len(order_idx)), dat_pA.iloc[order_idx][f'pscores_{i}_logit'], 
                  color=mycols[1], marker='^', label='logit' if i == 0 else "")
        ax.scatter(range(len(order_idx)), dat_pA.iloc[order_idx][f'pscores_{i}_nn'], 
                  color=mycols[2], marker='x', label='nn' if i == 0 else "")
        
        ax.set_ylabel(f'true pscore for P(A={i})')
        ax.set_xlabel(f'predicted pscore for P(A={i})')
        
        if i == 0:
            ax.legend(loc='upper left')
    
    # Outcome Y plots
    for d in range(k):
        ax = axes[1, d] if k > 1 else axes[1]
        mask = A == d
        
        sorted_Y = np.sort(dat.loc[mask, 'Y'])
        ax.plot(sorted_Y, color=mycols[0], linewidth=2, label='observed' if d == 0 else "")
        
        Y_order = dat['Y'].argsort()
        ax.scatter(range(sum(mask)), dat.iloc[Y_order].loc[mask, 'Yhat_expo'], 
                  color=mycols[1], marker='^', label=expo_or_ols if d == 0 else "")
        ax.scatter(range(sum(mask)), dat.iloc[Y_order].loc[mask, 'Yhat_nn'], 
                  color=mycols[2], marker='x', label='nn' if d == 0 else "")
        
        ax.set_ylabel(f'observed Y[A={d}]')
        ax.set_xlabel(f'predicted Y[A={d}]')
        
        if d == 0:
            ax.legend(loc=legposY)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{root}images/YYhat_sorted/YYhat_sorted_k{k}{A_flavor}{Y_flavor}_dset{ds}.jpeg", 
                   dpi=100, bbox_inches='tight')
    
    plt.show()