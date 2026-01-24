# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: boxplots.py
# Date: 2026-01-24
# Note: This script creates boxplot visualizations for simulation results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from itertools import combinations


def parse_filename(filepath):
    """
    Parse the CSV filename to extract k, A_flavor, Y_flavor, and Y_param.

    Example: simk2_logit_lognormal_est_with_lognormal.csv
    Returns: {'k': 2, 'A_flavor': 'logit', 'Y_flavor': 'lognormal', 'Y_param': 'lognormal'}
    """
    filename = os.path.basename(filepath)

    # Extract k
    k_match = re.search(r'simk(\d+)', filename)
    k = int(k_match.group(1)) if k_match else None

    # Extract A_flavor and Y_flavor (between simkX_ and _est_with)
    flavor_match = re.search(r'simk\d+_(\w+)_(\w+)_est_with', filename)
    if flavor_match:
        A_flavor = flavor_match.group(1)
        Y_flavor = flavor_match.group(2)
    else:
        A_flavor, Y_flavor = None, None

    # Extract Y_param (after "with_" and before ".csv")
    param_match = re.search(r'est_with_(\w+)\.csv', filename)
    Y_param = param_match.group(1) if param_match else None

    return {
        'k': k,
        'A_flavor': A_flavor,
        'Y_flavor': Y_flavor,
        'Y_param': Y_param
    }


def get_comparison_columns(df):
    """
    Extract the A_XX column names (excluding pval columns).
    """
    return [col for col in df.columns if col.startswith('A_') and not col.endswith('_pval')]


def create_boxplot_data(df, A_columns, Y_param):
    """
    Transform the data for boxplot visualization.
    Computes errors: True_diff - Estimate for each model.

    Returns a DataFrame with columns: dataset, comparison, model, error
    """
    rows = []

    for A_col in A_columns:
        # Get data for each dataset
        for dataset in df['dataset'].unique():
            df_dataset = df[df['dataset'] == dataset]

            # Get true difference
            true_diff = df_dataset[df_dataset['estimate'] == 'True_diff'][A_col].values
            if len(true_diff) == 0:
                continue
            true_diff = true_diff[0]

            # Get estimates for each model
            for _, row in df_dataset.iterrows():
                estimate_type = row['estimate']

                if estimate_type == 'True_diff':
                    continue

                # Determine model name
                if estimate_type == 'Naive_est':
                    model = 'Naive'
                elif estimate_type == 'NN_est':
                    model = 'NN'
                elif Y_param.lower() in estimate_type.lower():
                    model = Y_param.upper()
                else:
                    # Handle other parametric models (e.g., Logitols_est, Logitexpo_est)
                    model = estimate_type.replace('_est', '').replace('Logit', '')

                estimate_val = row[A_col]
                if pd.notna(estimate_val):
                    error = true_diff - estimate_val
                    rows.append({
                        'dataset': dataset,
                        'comparison': A_col,
                        'model': model,
                        'error': error
                    })

    return pd.DataFrame(rows)


def save_boxplots(filepath, save=True, show=True):
    """
    Create and save boxplots from a simulation results CSV file.

    Parameters:
    - filepath: path to the CSV file
    - save: whether to save the plot
    - show: whether to display the plot

    Returns:
    - fig: the matplotlib figure object
    - parsed: dictionary with parsed filename info
    """
    # Parse filename
    parsed = parse_filename(filepath)
    k = parsed['k']
    A_flavor = parsed['A_flavor']
    Y_flavor = parsed['Y_flavor']
    Y_param = parsed['Y_param']

    print(f"Parsed: k={k}, A_flavor={A_flavor}, Y_flavor={Y_flavor}, Y_param={Y_param}")

    # Read data
    df = pd.read_csv(filepath)

    # Get comparison columns
    A_columns = get_comparison_columns(df)
    print(f"Found comparisons: {A_columns}")

    # Create boxplot data
    plot_df = create_boxplot_data(df, A_columns, Y_param)

    # Define model order for consistent plotting
    model_order = ['Naive', Y_param.upper(), 'NN']
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)

    # Define colors for each model
    colors = {'Naive': '#E57373', Y_param.upper(): '#64B5F6', 'NN': '#81C784'}

    # Create figure
    n_comparisons = len(A_columns)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(4 * n_comparisons, 5), squeeze=False)
    axes = axes.flatten()

    for idx, A_col in enumerate(A_columns):
        ax = axes[idx]
        subset = plot_df[plot_df['comparison'] == A_col].copy()

        # Create grouped boxplot
        positions = []
        box_data = []
        box_colors = []

        for i, model in enumerate(model_order):
            model_data = subset[subset['model'] == model]['error'].values
            if len(model_data) > 0:
                box_data.append(model_data)
                positions.append(i)
                box_colors.append(colors.get(model, 'gray'))

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Labels
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, fontsize=10)
        ax.set_title(f'{A_col}', fontsize=12)
        ax.set_ylabel('Error (True - Estimate)' if idx == 0 else '', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Overall title
    fig.suptitle(f'Estimation Errors: DGP = {A_flavor}-{Y_flavor}, Estimator = {Y_param}\n(k={k})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], alpha=0.7, label=m) for m in model_order]
    #fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95))

    # Save
    if save:
        # Determine output directory
        dir_path = os.path.dirname(filepath)
        images_dir = dir_path.replace('/tables', '/images')
        os.makedirs(images_dir, exist_ok=True)

        # Create output filename
        out_filename = f"boxplot_k{k}_{A_flavor}_{Y_flavor}_with_{Y_param}.png"
        out_path = os.path.join(images_dir, out_filename)

        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig, parsed


def save_pooled_boxplot(filepath, save=True, show=True):
    """
    Create a single pooled boxplot (all comparisons combined) from a simulation results CSV file.

    Parameters:
    - filepath: path to the CSV file
    - save: whether to save the plot
    - show: whether to display the plot

    Returns:
    - fig: the matplotlib figure object
    """
    # Parse filename
    parsed = parse_filename(filepath)
    k = parsed['k']
    A_flavor = parsed['A_flavor']
    Y_flavor = parsed['Y_flavor']
    Y_param = parsed['Y_param']

    # Read data
    df = pd.read_csv(filepath)

    # Get comparison columns
    A_columns = get_comparison_columns(df)

    # Create boxplot data
    plot_df = create_boxplot_data(df, A_columns, Y_param)

    # Define model order
    model_order = ['Naive', Y_param.upper(), 'NN']
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)

    # Define colors
    colors = {'Naive': '#E57373', Y_param.upper(): '#64B5F6', 'NN': '#81C784'}

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create boxplot
    box_data = []
    box_colors = []

    for model in model_order:
        model_data = plot_df[plot_df['model'] == model]['error'].values
        box_data.append(model_data)
        box_colors.append(colors.get(model, 'gray'))

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Labels
    ax.set_xticks(range(1, len(model_order) + 1))
    ax.set_xticklabels(model_order, fontsize=11)
    ax.set_ylabel('Error (True - Estimate)', fontsize=11)
    ax.set_title(f'Estimation Errors (Pooled)\nDGP = {A_flavor}-{Y_flavor}, Estimator = {Y_param}, k={k}',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    if save:
        dir_path = os.path.dirname(filepath)
        images_dir = dir_path.replace('/tables', '/images')
        os.makedirs(images_dir, exist_ok=True)

        out_filename = f"boxplot_pooled_k{k}_{A_flavor}_{Y_flavor}_with_{Y_param}.png"
        out_path = os.path.join(images_dir, out_filename)

        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig


# Example usage
if __name__ == "__main__":
    # Test with a sample file
    test_path = "./_1trt_effect/tables/simk2_logit_lognormal_est_with_lognormal.csv"

    if os.path.exists(test_path):
        print("Creating boxplots...")
        save_boxplots(test_path, save=True, show=True)
        save_pooled_boxplot(test_path, save=True, show=True)
    else:
        print(f"Test file not found: {test_path}")
