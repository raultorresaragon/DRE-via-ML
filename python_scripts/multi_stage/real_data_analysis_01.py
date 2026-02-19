# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: real_data_analysis_01.py
# Date: 2025-01-08
# Note: This script computes the OTR based on real data provided
#       by Dr. Ahn
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import os

# Import custom functions
from pscores_models import estimate_A_nn, estimate_A_logit
from outcome_models import estimate_Y_nn, estimate_Y_ols
from compute_Vn import get_Vn

# Set random seed
np.random.seed(1810)

# Neural network parameter space for tuning
hidunits = [5, 20]
eps = [100, 250]
penals = [0.001, 0.01]

# Load data
df = pd.read_csv("real_data/recoded_ASTR.csv")

# Prepare variables
Y = df['gh'].values
X = df[['gender', 'age', 'partial_or_total_removal']].copy()
X.columns = ['X1', 'X2', 'X3']
A = df['chemo'].values

# Create complete dataset
dat = pd.concat([pd.Series(Y, name='Y'), 
                pd.Series(A, name='A'), 
                X], axis=1).dropna()

print(f"Dataset shape after removing missing values: {dat.shape}")

Y = dat['Y'].values
A = dat['A'].values

# Estimate propensity scores
print("Estimating propensity scores...")

# Logistic regression
fit_A_logit = estimate_A_logit(X=None, dat=dat, k=2, verbose=False)
pscores_logit = fit_A_logit['pscores'].iloc[:, 0].values

# Neural network
fit_A_nn = estimate_A_nn(X=None, dat=dat, k=2, 
                        hidunits=hidunits, eps=eps, penals=penals, verbose=False)
pscores_nn = fit_A_nn['pscores'].iloc[:, 0].values

# Estimate outcome models
print("Estimating outcome models...")

delta_1 = (A == 1).astype(int)
delta_0 = (A == 0).astype(int)

# Parametric models (OLS)
dat_1 = dat[dat['A'] == 1]
dat_0 = dat[dat['A'] == 0]

g_1 = LinearRegression()
g_1.fit(dat_1[['X1', 'X2', 'X3']], dat_1['Y'])
ghat_1 = g_1.predict(dat[['X1', 'X2', 'X3']])

g_0 = LinearRegression()
g_0.fit(dat_0[['X1', 'X2', 'X3']], dat_0['Y'])
ghat_0 = g_0.predict(dat[['X1', 'X2', 'X3']])

# Doubly robust estimates (parametric)
muhat_1_param = np.mean(ghat_1 + (delta_1 * (Y - ghat_1) / pscores_logit) / 
                       np.mean(delta_1 / pscores_logit))
muhat_0_param = np.mean(ghat_0 + (delta_0 * (Y - ghat_0) / (1 - pscores_logit)) / 
                       np.mean(delta_0 / (1 - pscores_logit)))
diff_means_param = muhat_1_param - muhat_0_param

print(f"Parametric difference in means: {diff_means_param:.3f}")

# Neural network models
fit_Y_nn = estimate_Y_nn(dat, pscores_df=fit_A_nn['pscores'], k=2,
                        hidunits=hidunits, eps=eps, penals=penals, verbose=False)

# Extract NN predictions (assuming binary treatment)
if 'A_01' in fit_Y_nn:
    ghat_1_nn = fit_Y_nn['A_01'][4]  # ghat_i (treatment 1)
    ghat_0_nn = fit_Y_nn['A_01'][3]  # ghat_j (treatment 0)
    
    # Doubly robust estimates (NN)
    muhat_1_nn = np.mean(ghat_1_nn + (delta_1 * (Y - ghat_1_nn) / pscores_nn) / 
                        np.mean(delta_1 / pscores_nn))
    muhat_0_nn = np.mean(ghat_0_nn + (delta_0 * (Y - ghat_0_nn) / (1 - pscores_nn)) / 
                        np.mean(delta_0 / (1 - pscores_nn)))
    diff_means_nn = muhat_1_nn - muhat_0_nn
    
    print(f"Neural network difference in means: {diff_means_nn:.3f}")

# Compute OTR for new patient
X_new = pd.DataFrame({'X1': [1], 'X2': [75], 'X3': [1]})

def get_Vn_simple(g_1, g_0, X_new, from_model="parametric"):
    """Simple OTR computation for binary treatment"""
    if from_model == "nn" and hasattr(g_1, 'predict'):
        V_1 = g_1.predict(X_new)
        V_0 = g_0.predict(X_new)
    else:
        V_1 = g_1.predict(X_new)
        V_0 = g_0.predict(X_new)
    
    result = X_new.copy()
    result['V_1'] = V_1
    result['V_0'] = V_0
    result['Optimal_A'] = (V_1 > V_0).astype(int)
    
    return result

# Compute OTR
Vn_df_param = get_Vn_simple(g_1, g_0, X_new, from_model="parametric")
print("OTR (Parametric):")
print(Vn_df_param)

if 'A_01' in fit_Y_nn:
    # For NN, we need to extract the actual models
    g_1_nn = fit_Y_nn['A_01'][1]  # g_i model
    g_0_nn = fit_Y_nn['A_01'][2]  # g_j model
    
    Vn_df_nn = get_Vn_simple(g_1_nn, g_0_nn, X_new, from_model="nn")
    print("OTR (Neural Network):")
    print(Vn_df_nn)

# Add predicted values to dataset for plotting
dat['Yhat_param'] = np.where(dat['A'] == 1, ghat_1, ghat_0)
if 'A_01' in fit_Y_nn:
    dat['Yhat_nn'] = np.where(dat['A'] == 1, ghat_1_nn, ghat_0_nn)

# Compute RMSE
def RMSE(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

rmse_param = RMSE(dat['Y'], dat['Yhat_param'])
print(f"RMSE (Parametric): {rmse_param:.1f}")

if 'Yhat_nn' in dat.columns:
    rmse_nn = RMSE(dat['Y'], dat['Yhat_nn'])
    print(f"RMSE (Neural Network): {rmse_nn:.1f}")

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plots
sample_idx = np.random.choice(len(dat), size=min(150, len(dat)), replace=False)
plot_dat = dat.iloc[sample_idx].sort_values('Y')

# Parametric model
axes[0].scatter(plot_dat['Yhat_param'], plot_dat['Y'], alpha=0.6, color='black')
axes[0].plot([dat['Y'].min(), dat['Y'].max()], [dat['Y'].min(), dat['Y'].max()], 
            'b--', linewidth=2)
axes[0].set_xlabel(f'GH predicted parametric (RMSE={rmse_param:.1f})')
axes[0].set_ylabel('GH observed')
axes[0].set_title('Parametric Model')

# Neural network model
if 'Yhat_nn' in dat.columns:
    axes[1].scatter(plot_dat['Yhat_nn'], plot_dat['Y'], alpha=0.6, color='black')
    axes[1].plot([dat['Y'].min(), dat['Y'].max()], [dat['Y'].min(), dat['Y'].max()], 
                'b--', linewidth=2)
    axes[1].set_xlabel(f'GH predicted NN (RMSE={rmse_nn:.1f})')
    axes[1].set_ylabel('GH observed')
    axes[1].set_title('Neural Network Model')

plt.tight_layout()
plt.savefig("images/rwd_gh_ghhat_scatters.png", dpi=150, bbox_inches='tight')
plt.show()

# Sorted predictions plot
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(dat['Y'])
plt.plot(dat['Y'].iloc[sorted_idx], 'o', color='blue', label='Observed', markersize=4)
plt.plot(dat['Yhat_param'].iloc[sorted_idx], '^', color='darkgrey', 
         label='Predicted (Parametric)', markersize=4)
if 'Yhat_nn' in dat.columns:
    plt.plot(dat['Yhat_nn'].iloc[sorted_idx], 's', color='black', 
             label='Predicted (NN)', markersize=4)

plt.xlabel('Sorted observations')
plt.ylabel('GH')
plt.title('GH and Predicted GH (in sample) by model')
plt.legend()
plt.tight_layout()
plt.savefig("images/rwd_gh_ghhat.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nReal data analysis complete!")