# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: combine_plots.py
# Date: 2025-01-08
# Note: This script creates combined visualization plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

def combine_boxplots_k2plus(results_dir, k_values=[2, 3, 5], flavors=['le', 'lg'], 
                           Y_param='ols', save_path=None):
    """
    Create combined boxplots for multiple k values and flavors
    
    Parameters:
    - results_dir: directory containing results CSV files
    - k_values: list of k values to include
    - flavors: list of flavor combinations to include
    - Y_param: parameter estimation method
    - save_path: path to save the combined plot
    """
    
    fig, axes = plt.subplots(len(k_values), len(flavors), 
                            figsize=(4*len(flavors), 4*len(k_values)))
    
    if len(k_values) == 1:
        axes = axes.reshape(1, -1)
    if len(flavors) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, k in enumerate(k_values):
        for j, flav in enumerate(flavors):
            
            # Determine A_flavor and Y_flavor from flav
            A_flavor = "logit" if flav[0] == 'l' else "tanh"
            Y_flavor_map = {'e': 'expo', 's': 'sigmoid', 'l': 'lognormal', 'g': 'gamma'}
            Y_flavor = Y_flavor_map[flav[1]]
            
            # Load results
            file_path = Path(results_dir) / f"simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Prepare data for boxplot
                estimates = ['True_diff', 'NN_est', f'Logit{Y_param}_est', 'Naive_est']
                
                # Get comparison columns (exclude dataset and estimate columns)
                comparison_cols = [col for col in df.columns 
                                 if col.startswith('A_') and not col.endswith('_pval')]
                
                if comparison_cols:
                    # Melt data for plotting
                    plot_data = []
                    for est in estimates:
                        est_data = df[df['estimate'] == est]
                        if not est_data.empty:
                            for col in comparison_cols:
                                values = est_data[col].dropna()
                                for val in values:
                                    plot_data.append({
                                        'Estimate': est,
                                        'Comparison': col,
                                        'Value': val
                                    })
                    
                    if plot_data:
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Create boxplot
                        sns.boxplot(data=plot_df, x='Estimate', y='Value', 
                                   ax=axes[i, j])
                        axes[i, j].set_title(f'k={k}, {A_flavor}-{Y_flavor}')
                        axes[i, j].tick_params(axis='x', rotation=45)
                        
                        if j == 0:
                            axes[i, j].set_ylabel('Difference in Means')
                        else:
                            axes[i, j].set_ylabel('')
                        
                        if i == len(k_values) - 1:
                            axes[i, j].set_xlabel('Estimation Method')
                        else:
                            axes[i, j].set_xlabel('')
            else:
                axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center',
                               transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'k={k}, {A_flavor}-{Y_flavor}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def compare_methods_plot(results_dir, k=3, flavor='le', Y_param='ols', save_path=None):
    """
    Create comparison plot between different estimation methods
    
    Parameters:
    - results_dir: directory containing results CSV files
    - k: number of treatment levels
    - flavor: flavor combination
    - Y_param: parameter estimation method
    - save_path: path to save the plot
    """
    
    # Determine A_flavor and Y_flavor from flavor
    A_flavor = "logit" if flavor[0] == 'l' else "tanh"
    Y_flavor_map = {'e': 'expo', 's': 'sigmoid', 'l': 'lognormal', 'g': 'gamma'}
    Y_flavor = Y_flavor_map[flavor[1]]
    
    # Load results
    file_path = Path(results_dir) / f"simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}.csv"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    # Get comparison columns
    comparison_cols = [col for col in df.columns 
                      if col.startswith('A_') and not col.endswith('_pval')]
    
    if not comparison_cols:
        print("No comparison columns found")
        return
    
    # Create subplots for each comparison
    n_comparisons = len(comparison_cols)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(5*n_comparisons, 5))
    
    if n_comparisons == 1:
        axes = [axes]
    
    for idx, col in enumerate(comparison_cols):
        # Extract estimates for this comparison
        true_vals = df[df['estimate'] == 'True_diff'][col].values
        nn_vals = df[df['estimate'] == 'NN_est'][col].values
        param_vals = df[df['estimate'] == f'Logit{Y_param}_est'][col].values
        naive_vals = df[df['estimate'] == 'Naive_est'][col].values
        
        # Create scatter plot
        if len(true_vals) > 0:
            true_val = true_vals[0]  # True value should be constant
            
            axes[idx].axhline(y=true_val, color='red', linestyle='--', 
                             label='True Value', linewidth=2)
            
            if len(nn_vals) > 0:
                axes[idx].scatter(range(len(nn_vals)), nn_vals, 
                                 alpha=0.7, label='Neural Network', s=50)
            
            if len(param_vals) > 0:
                axes[idx].scatter(range(len(param_vals)), param_vals, 
                                 alpha=0.7, label=f'Logit-{Y_param}', s=50)
            
            if len(naive_vals) > 0:
                axes[idx].scatter(range(len(naive_vals)), naive_vals, 
                                 alpha=0.7, label='Naive', s=50)
        
        axes[idx].set_title(f'Comparison {col}')
        axes[idx].set_xlabel('Simulation')
        axes[idx].set_ylabel('Estimated Difference')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Method Comparison: k={k}, {A_flavor}-{Y_flavor}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: create combined boxplots
    # combine_boxplots_k2plus("./results/tables", k_values=[3, 5], 
    #                        flavors=['le', 'lg'], save_path="combined_boxplots.png")
    
    # Example: compare methods for specific case
    # compare_methods_plot("./results/tables", k=3, flavor='le', 
    #                     save_path="method_comparison.png")
    
    print("Plot functions defined. Use combine_boxplots_k2plus() and compare_methods_plot() to create visualizations.")