# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: boxplots.py
# Date: 2026-01-24
# Note: This script creates grouped boxplot visualizations for simulation results,
#       with separate boxplot sets for each treatment comparison (A_XX) combined
#       into a single figure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def parse_filename(filepath):
    """
    Parse the CSV filename to extract k, A_flavor, Y_flavor.

    Example: simk2_logit_lognormal.csv
    Returns: {'k': 2, 'A_flavor': 'logit', 'Y_flavor': 'lognormal'}
    """
    filename = os.path.basename(filepath)

    k_match = re.search(r'simk(\d+)', filename)
    k = int(k_match.group(1)) if k_match else None

    flavor_match = re.search(r'simk\d+_(\w+)_(\w+)\.csv', filename)
    if flavor_match:
        A_flavor = flavor_match.group(1)
        Y_flavor = flavor_match.group(2)
    else:
        A_flavor, Y_flavor = None, None

    return {'k': k, 'A_flavor': A_flavor, 'Y_flavor': Y_flavor}


def get_comparison_columns(df):
    """
    Extract the A_XX column names (excluding pval columns).
    """
    return [col for col in df.columns if col.startswith('A_') and not col.endswith('_pval')]


def estimate_to_model(estimate_type):
    """Map estimate label to display model name."""
    if estimate_type == 'Naive_est': return 'Naive'
    if estimate_type == 'NN_est':    return 'NN'
    return estimate_type.replace('Logit', '').replace('_est', '').upper()


def create_boxplot_data(df, A_columns):
    """
    Transform the data for boxplot visualization.
    Computes errors: True_diff - Estimate for each model.

    Returns a DataFrame with columns: dataset, comparison, model, error
    """
    rows = []

    for A_col in A_columns:
        for dataset in df['dataset'].unique():
            df_dataset = df[df['dataset'] == dataset]

            true_diff = df_dataset[df_dataset['estimate'] == 'True_diff'][A_col].values
            if len(true_diff) == 0:
                continue
            true_diff = true_diff[0]

            for _, row in df_dataset.iterrows():
                estimate_type = row['estimate']
                if estimate_type == 'True_diff':
                    continue
                estimate_val = row[A_col]
                if pd.notna(estimate_val):
                    rows.append({
                        'dataset': dataset,
                        'comparison': A_col,
                        'model': estimate_to_model(estimate_type),
                        'error': true_diff - estimate_val
                    })

    return pd.DataFrame(rows)


def save_boxplots(filepath, save=True, show=True, ncols=None):
    """
    Create and save a single figure with grouped boxplots for all treatment comparisons.

    Each treatment comparison (A_XX) gets its own subplot.
    Models shown are detected dynamically from the CSV estimate column.

    Parameters:
    - filepath: path to the CSV file
    - save: whether to save the plot
    - show: whether to display the plot
    - ncols: number of columns in subplot grid (default: auto-calculated)

    Returns:
    - fig: the matplotlib figure object
    - parsed: dictionary with parsed filename info
    """
    parsed = parse_filename(filepath)
    k = parsed['k']
    A_flavor = parsed['A_flavor']
    Y_flavor = parsed['Y_flavor']

    print(f"Parsed: |A|={k}, A_flavor={A_flavor}, Y_flavor={Y_flavor}")

    df = pd.read_csv(filepath)
    A_columns = get_comparison_columns(df)
    n_comparisons = len(A_columns)
    print(f"Found {n_comparisons} comparisons: {A_columns}")

    plot_df = create_boxplot_data(df, A_columns)

    # Consistent model order — keep only models present in the data
    preferred_order = ['Naive', 'OLS', 'EXPO', 'LOGNORMAL', 'NN']
    model_order = [m for m in preferred_order if m in plot_df['model'].unique()]
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)

    colors = {
        'Naive':     '#E57373',
        'OLS':       '#FFB74D',
        'EXPO':      '#64B5F6',
        'LOGNORMAL': '#64B5F6',
        'NN':        '#81C784'
    }

    if ncols is None:
        if n_comparisons == 1:    ncols = 1
        elif n_comparisons <= 3:  ncols = n_comparisons
        elif n_comparisons <= 6:  ncols = 3
        else:                     ncols = 5

    nrows = int(np.ceil(n_comparisons / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, A_col in enumerate(A_columns):
        ax = axes[idx]
        subset = plot_df[plot_df['comparison'] == A_col].copy()

        box_data = []
        box_colors = []
        for model in model_order:
            model_data = subset[subset['model'] == model]['error'].values
            box_data.append(model_data if len(model_data) > 0 else [np.nan])
            box_colors.append(colors.get(model, 'gray'))

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, fontsize=9)
        ax.set_title(f'Estimation errors of {A_col}\nDGP = {A_flavor}_{Y_flavor}, |A|={k}',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Error (True - Est)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 0.98])

    if save:
        images_dir = os.path.dirname(filepath).replace('/tables', '/images')
        os.makedirs(images_dir, exist_ok=True)
        out_path = os.path.join(images_dir, f"boxplot_k{k}_{A_flavor}_{Y_flavor}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig, parsed


if __name__ == "__main__":
    import glob
    tables_dir = "./_1trt_effect/tables/Results"
    files = sorted(glob.glob(f"{tables_dir}/simk2_*.csv"))

    if not files:
        print(f"No simk2_*.csv files found in {tables_dir}")
    else:
        print(f"Found {len(files)} files:")
        for f in files:
            print(f"  {f}")
        for filepath in files:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            save_boxplots(filepath, save=True, show=False)
