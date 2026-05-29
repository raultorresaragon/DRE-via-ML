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


def get_comparison_columns(df, baseline_only=False):
    """
    Extract the A_XX column names (excluding pval columns).
    If baseline_only=True, keep only comparisons against arm 0 (A_0X).
    """
    cols = [col for col in df.columns if col.startswith('A_') and not col.endswith('_pval')]
    if baseline_only:
        cols = [c for c in cols if c.startswith('A_0')]
    return cols


def estimate_to_model(estimate_type):
    """Map estimate label to display model name."""
    if estimate_type == 'Naive_est': return 'Naive'
    if estimate_type == 'NN_est':    return 'NN'
    return estimate_type.replace('Logit', '').replace('_est', '').upper()


def create_boxplot_data(df, A_columns, relative_bias=False):
    """
    Transform the data for boxplot visualization.
    Computes errors: True_diff - Estimate for each model.
    If relative_bias=True, divides by abs(True_diff): (True - Est) / |True|.

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
                    error = true_diff - estimate_val
                    if relative_bias and true_diff != 0:
                        error = error / abs(true_diff)
                    rows.append({
                        'dataset': dataset,
                        'comparison': A_col,
                        'model': estimate_to_model(estimate_type),
                        'error': error
                    })

    return pd.DataFrame(rows)


def save_boxplots(filepath, save=True, show=True, ncols=None, baseline_only=True,
                  relative_bias=False, greyscale=False):
    """
    Create and save a single figure with grouped boxplots for all treatment comparisons.

    Each treatment comparison (A_XX) gets its own subplot.
    Models shown are detected dynamically from the CSV estimate column.

    Parameters:
    - filepath: path to the CSV file
    - save: whether to save the plot
    - show: whether to display the plot
    - ncols: number of columns in subplot grid (default: auto-calculated)
    - baseline_only: if True (default), show only comparisons vs arm 0 (A_0X);
                     if False, show all pairwise comparisons
    - relative_bias: if True, plot (True - Est) / |True| instead of raw bias
    - greyscale: if True, use grey shades (NN=darkest, Naive=lightest) instead of colour

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
    A_columns = get_comparison_columns(df, baseline_only=baseline_only)
    n_comparisons = len(A_columns)
    print(f"Found {n_comparisons} comparisons: {A_columns}")

    plot_df = create_boxplot_data(df, A_columns, relative_bias=relative_bias)

    # Consistent model order — keep only models present in the data
    preferred_order = ['Naive', 'OLS', 'EXPO', 'LOGNORMAL', 'NN']
    model_order = [m for m in preferred_order if m in plot_df['model'].unique()]
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)

    if greyscale:
        colors = {
            'Naive':     '0.82',   # lightest
            'OLS':       '0.62',
            'EXPO':      '0.42',
            'LOGNORMAL': '0.42',
            'NN':        '0.20',   # darkest
        }
    else:
        colors = {
            'Naive':     '#E57373',
            'OLS':       '#FFB74D',
            'EXPO':      '#64B5F6',
            'LOGNORMAL': '#64B5F6',
            'NN':        '#81C784',
        }

    if ncols is None:
        if n_comparisons == 1:    ncols = 1
        elif n_comparisons <= 3:  ncols = n_comparisons
        elif n_comparisons == 4:  ncols = 2
        elif n_comparisons <= 6:  ncols = 3
        else:                     ncols = 5

    nrows = int(np.ceil(n_comparisons / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, A_col in enumerate(A_columns):
        ax = axes[idx]
        subset = plot_df[plot_df['comparison'] == A_col].copy()

        box_data = []
        box_colors = []
        for model in model_order:
            model_data = subset[subset['model'] == model]['error'].values
            if len(model_data) > 0:
                q1, q3 = np.percentile(model_data, [25, 75])
                iqr = q3 - q1
                model_data = model_data[
                    (model_data >= q1 - 2 * iqr) & (model_data <= q3 + 2 * iqr)
                ]
            box_data.append(model_data if len(model_data) > 0 else [np.nan])
            box_colors.append(colors.get(model, 'gray'))

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, fontsize=13)
        ax.tick_params(axis='y', labelsize=12)
        title_flavor = 'loggamma' if Y_flavor == 'gamma' else Y_flavor
        ax.set_title(f'Estimation errors of {A_col}\nDGP = {A_flavor}_{title_flavor}, |A|={k}',
                     fontsize=13, fontweight='bold')
        ylabel = 'Rel. Bias (True - Est) / |True|' if relative_bias else 'Error (True - Est)'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 0.98])

    if save:
        images_dir = os.path.dirname(filepath).replace('/tables', '/images')
        os.makedirs(images_dir, exist_ok=True)
        suffix = ('_relbias' if relative_bias else '') + ('_bw' if greyscale else '')
        out_path = os.path.join(images_dir, f"boxplot_k{k}_{A_flavor}_{Y_flavor}{suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig, parsed


if __name__ == "__main__":
    import glob
    zero_effect   = False
    K_FILTER      = 5       # set to 2, 3, or 5 to process only that k; None = all
    BASELINE_ONLY = True    # True = only A_0X comparisons; False = all pairwise
    RELATIVE_BIAS = True    # True = plot (True - Est) / |True|; False = raw bias
    GREYSCALE     = True   # True = grey shades (NN darkest, Naive lightest)

    root       = f"./_{'0' if zero_effect else '1'}trt_effect"
    tables_dir = f"{root}/tables/Results"

    if K_FILTER is not None:
        pattern = f"{tables_dir}/simk{K_FILTER}_*.csv"
    else:
        pattern = f"{tables_dir}/simk*.csv"

    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching {pattern}")
    else:
        print(f"Found {len(files)} files:")
        for f in files:
            print(f"  {f}")
        for filepath in files:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            save_boxplots(filepath, save=True, show=False, baseline_only=BASELINE_ONLY,
                          relative_bias=RELATIVE_BIAS, greyscale=GREYSCALE)
