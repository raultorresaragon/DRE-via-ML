# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OTR_assess.py
# Assess performance of estimated OTR (DRE) against the true OTR
#
# Metrics
# -------
# Agreement  : P(d_star_k = d_true_k) per stage and jointly
# Confusion  : confusion matrix per stage
# Value      : V(d_true) and V(d_star) using oracle Q2 values from the OTR file
# Regret     : V(d_true) - V(d_star)
#
# Value computation
# -----------------
# The true OTR file contains Q2_a{a} = E[Y | history, A2=a] under the true model.
# V(d) = (1/n) sum_i Q2_{d2_i}(X1_i, A1_i, X2_i)
# This uses the oracle Q-function evaluated at the estimated decisions — the standard
# "direct method" for value estimation in simulation studies.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')
images_dir   = os.path.join(script_dir, '../_1trt_effect/2stages/images')


def assess_otr(filename, save_plots=True, verbose=True):
    """
    Assess DRE estimated OTR against the true OTR for a single dataset.

    Parameters
    ----------
    filename   : str   Base dataset filename without extension (e.g. 's2_k2_logit_expo_0')
    save_plots : bool  Save figures to images_dir
    verbose    : bool  Print summary to console

    Returns
    -------
    dict with keys: accuracy, value, regret, confusion
    """
    # ------------------------------------------------------------------
    # Load OTR and DRE files
    # ------------------------------------------------------------------
    # True OTR: get_true_otr_two_stage.py names it {filename}_OTR_{i}.csv
    # We detect i from the filename suffix (e.g. '_0' → i=0)
    i = int(filename.split('_')[-1])

    otr_path = os.path.join(datasets_dir, f'{filename}_OTR_{i}.csv')
    dre_path = os.path.join(datasets_dir, f'{filename}_DRE.csv')

    otr = pd.read_csv(otr_path)
    dre = pd.read_csv(dre_path)

    # True optimal decisions
    d_true_1 = otr['d1_star'].values
    d_true_2 = otr['d2_star'].values

    # Estimated decisions
    d_star_1 = dre['d_star_1'].values
    d_star_2 = dre['d_star_2'].values

    # Number of treatment levels (inferred from Q2 columns in OTR file)
    k2 = sum(1 for c in otr.columns if c.startswith('Q2_a'))
    k1 = sum(1 for c in otr.columns if c.startswith('Q1_a'))
    n  = len(otr)

    # ==================================================================
    # AGREEMENT RATES
    # ==================================================================
    acc_1     = np.mean(d_star_1 == d_true_1)
    acc_2     = np.mean(d_star_2 == d_true_2)
    acc_joint = np.mean((d_star_1 == d_true_1) & (d_star_2 == d_true_2))

    # ==================================================================
    # CONFUSION MATRICES
    # ==================================================================
    labels   = list(range(max(k1, k2)))
    cm_1 = confusion_matrix(d_true_1, d_star_1, labels=list(range(k1)))
    cm_2 = confusion_matrix(d_true_2, d_star_2, labels=list(range(k2)))

    # ==================================================================
    # VALUE FUNCTION  (direct method using oracle Q2)
    # V(d) = mean_i  Q2_{d2_i}  evaluated at d2 = d_true_2 or d_star_2
    # ==================================================================
    Q2_cols = [f'Q2_a{a}' for a in range(k2)]
    Q2_mat  = otr[Q2_cols].values   # (n, k2)

    # V(d_true): mean Q2 under true optimal A2
    V_true  = np.mean(Q2_mat[np.arange(n), d_true_2])

    # V(d_star): mean Q2 under estimated A2
    V_star  = np.mean(Q2_mat[np.arange(n), d_star_2])

    regret          = V_true - V_star
    relative_regret = regret / V_true if V_true != 0 else np.nan

    # ==================================================================
    # PRINT SUMMARY
    # ==================================================================
    if verbose:
        print(f"\n{'='*55}")
        print(f"OTR Assessment: {filename}")
        print(f"{'='*55}")
        print(f"\n  Agreement rates")
        print(f"    Stage 1 : {acc_1:.1%}  ({int(acc_1*n)}/{n})")
        print(f"    Stage 2 : {acc_2:.1%}  ({int(acc_2*n)}/{n})")
        print(f"    Joint   : {acc_joint:.1%}  ({int(acc_joint*n)}/{n})")
        print(f"\n  Value function")
        print(f"    V(d_true) = {V_true:.4f}")
        print(f"    V(d_star) = {V_star:.4f}")
        print(f"    Regret    = {regret:.4f}  ({relative_regret:.1%} relative)")
        print(f"\n  Stage 1 classification report")
        print(classification_report(d_true_1, d_star_1, zero_division=0))
        print(f"  Stage 2 classification report")
        print(classification_report(d_true_2, d_star_2, zero_division=0))

    # ==================================================================
    # PLOTS
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"OTR Assessment: {filename}", fontsize=11)

    # --- Plot 1: Confusion matrix stage 1 ---
    sns.heatmap(cm_1, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'est a={a}' for a in range(k1)],
                yticklabels=[f'true a={a}' for a in range(k1)],
                ax=axes[0])
    axes[0].set_title(f'Stage 1  (acc={acc_1:.1%})')
    axes[0].set_xlabel('Estimated $d^*_1$')
    axes[0].set_ylabel('True $d^*_1$')

    # --- Plot 2: Confusion matrix stage 2 ---
    sns.heatmap(cm_2, annot=True, fmt='d', cmap='Oranges',
                xticklabels=[f'est a={a}' for a in range(k2)],
                yticklabels=[f'true a={a}' for a in range(k2)],
                ax=axes[1])
    axes[1].set_title(f'Stage 2  (acc={acc_2:.1%})')
    axes[1].set_xlabel('Estimated $d^*_2$')
    axes[1].set_ylabel('True $d^*_2$')

    # --- Plot 3: Value comparison bar chart ---
    axes[2].bar(['V(d_true)', 'V(d_star)'], [V_true, V_star],
                color=['steelblue', 'tomato'], width=0.4)
    axes[2].set_title(f'Value  (regret={regret:.3f})')
    axes[2].set_ylabel('E[Y]')
    for bar, val in zip(axes[2].patches, [V_true, V_star]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*abs(V_true),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_plots:
        out_path = os.path.join(images_dir, f'{filename}_assess.jpeg')
        fig.savefig(out_path)
        print(f"\n  Plot saved: {filename}_assess.jpeg")
    plt.close(fig)

    return {
        'accuracy':  {'stage1': acc_1, 'stage2': acc_2, 'joint': acc_joint},
        'value':     {'V_true': V_true, 'V_star': V_star},
        'regret':    {'absolute': regret, 'relative': relative_regret},
        'confusion': {'stage1': cm_1, 'stage2': cm_2},
    }


# ============================================================
# Run over all datasets in _info.csv
# ============================================================
if __name__ == '__main__':
    info = pd.read_csv(os.path.join(datasets_dir, '_info.csv'))
    results = []
    for _, row in info.iterrows():
        try:
            res = assess_otr(row['filename'], save_plots=True, verbose=False)
            results.append({
                'filename':        row['filename'],
                'k':               row['k1'],
                'flavor_Y':        row['flavor_Y'],
                'i':               row['i'],
                'acc_stage1':      res['accuracy']['stage1'],
                'acc_stage2':      res['accuracy']['stage2'],
                'acc_joint':       res['accuracy']['joint'],
                'V_true':          res['value']['V_true'],
                'V_star':          res['value']['V_star'],
                'regret':          res['regret']['absolute'],
                'regret_relative': res['regret']['relative'],
            })
        except FileNotFoundError as e:
            print(f"Skipping {row['filename']}: {e}")

    summary = pd.DataFrame(results)
    out_path = os.path.join(datasets_dir, '_assessment_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ Summary saved to _assessment_summary.csv")
    print(summary.groupby(['k', 'flavor_Y'])[
        ['acc_stage1', 'acc_stage2', 'acc_joint', 'regret', 'regret_relative']
    ].mean().round(4))
