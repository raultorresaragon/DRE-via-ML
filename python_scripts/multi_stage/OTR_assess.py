# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OTR_assess.py
# Assess performance of estimated OTR against the true OTR — two-stage version.
#
# Runs assessment for both DRE_ML and naive estimators, producing:
#   - Combined 2-row figure: row 1 = DRE_ML, row 2 = naive
#     (stage 1 confusion, stage 2 confusion, value bar chart)
#   - Summary CSV with a 'model' column tagging each row as 'DRE_ML' or 'naive'
#
# Metrics
# -------
# Agreement  : P(d_star_k = d_true_k) per stage and jointly
# Confusion  : confusion matrix per stage
# Value      : V(d_true) and V(d_star) using oracle Q2 values from the OTR file
# Regret     : V(d_true) - V(d_star)
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


def assess_otr(filename, model='DRE_ML', verbose=True):
    """
    Assess estimated OTR against the true OTR for a single dataset.

    Parameters
    ----------
    filename : str   Base dataset filename without extension
    model    : str   'DRE_ML' loads {filename}_DRE.csv;
                     'naive'  loads {filename}_NAIVE.csv
    verbose  : bool  Print summary to console

    Returns
    -------
    dict with keys: accuracy, value, regret, confusion, k
    """
    suffix = '_DRE' if model == 'DRE_ML' else '_NAIVE'

    dat = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
    otr = pd.read_csv(os.path.join(datasets_dir, f'{filename}_OTR.csv'))
    est = pd.read_csv(os.path.join(datasets_dir, f'{filename}{suffix}.csv'))

    A1_obs = dat['A1'].values
    A2_obs = dat['A2'].values
    Y_obs  = dat['Y'].values

    d_true_1 = otr['d1_star'].values
    d_true_2 = otr['d2_star'].values
    d_star_1 = est['d_star_1'].values
    d_star_2 = est['d_star_2'].values

    k1 = sum(1 for c in otr.columns if c.startswith('Q1_a'))
    k2 = sum(1 for c in otr.columns if c.startswith('Q2_a'))
    n  = len(otr)

    # Agreement
    acc_1     = np.mean(d_star_1 == d_true_1)
    acc_2     = np.mean(d_star_2 == d_true_2)
    acc_joint = np.mean((d_star_1 == d_true_1) & (d_star_2 == d_true_2))

    # Confusion matrices
    cm_1 = confusion_matrix(d_true_1, d_star_1, labels=list(range(k1)))
    cm_2 = confusion_matrix(d_true_2, d_star_2, labels=list(range(k2)))

    # Value function (oracle Q2)
    Q2_mat  = otr[[f'Q2_a{a}' for a in range(k2)]].values
    V_true  = np.mean(Q2_mat[np.arange(n), d_true_2])
    V_star  = np.mean(Q2_mat[np.arange(n), d_star_2])
    regret          = V_true - V_star
    relative_regret = regret / V_true if V_true != 0 else np.nan

    if verbose:
        print(f"\n{'='*55}")
        print(f"OTR Assessment [{model}]: {filename}")
        print(f"{'='*55}")
        print(f"\n  Agreement rates")
        print(f"    Stage 1 : {acc_1:.1%}  ({int(acc_1*n)}/{n})")
        print(f"    Stage 2 : {acc_2:.1%}  ({int(acc_2*n)}/{n})")
        print(f"    Joint   : {acc_joint:.1%}  ({int(acc_joint*n)}/{n})")
        col = 16
        print(f"\n  Value summary")
        print(f"    {'Observed mean Y':>{col}}  {'V(Optimal OTR)':>{col}}  {'V(Estimated)':>{col}}")
        print(f"    {np.mean(Y_obs):>{col}.4f}  {V_true:>{col}.4f}  {V_star:>{col}.4f}")
        print(f"\n  Regret = {regret:.4f}  ({relative_regret:.1%} relative)")
        print(f"\n  Stage 1 confusion matrix  (rows=optimal, cols=estimated)")
        print(pd.DataFrame(cm_1,
                           index=[f'opt a={a}' for a in range(k1)],
                           columns=[f'est a={a}' for a in range(k1)]).to_string())
        print(f"\n  Stage 1 classification report")
        print(classification_report(d_true_1, d_star_1, zero_division=0))
        print(f"  Stage 2 confusion matrix  (rows=optimal, cols=estimated)")
        print(pd.DataFrame(cm_2,
                           index=[f'opt a={a}' for a in range(k2)],
                           columns=[f'est a={a}' for a in range(k2)]).to_string())
        print(f"\n  Stage 2 classification report")
        print(classification_report(d_true_2, d_star_2, zero_division=0))

    return {
        'accuracy':  {'stage1': acc_1, 'stage2': acc_2, 'joint': acc_joint},
        'value':     {'V_true': V_true, 'V_star': V_star},
        'regret':    {'absolute': regret, 'relative': relative_regret},
        'confusion': {'stage1': cm_1, 'stage2': cm_2},
        'k':         (k1, k2),
    }


def plot_comparison(filename, res_dre, res_naive, save=True):
    """
    Two-row figure: row 1 = DRE_ML, row 2 = naive.
    Columns: stage 1 confusion, stage 2 confusion, value bar chart.
    """
    k1, k2 = res_dre['k']

    def _cm_annot(cm):
        annot = np.empty(cm.shape, dtype=object)
        total = cm.sum()
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f'{cm[r,c]}\n({cm[r,c]/total*100:.1f}%)'
        return annot

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"OTR Assessment: {filename}", fontsize=11)

    for row_idx, (res, label) in enumerate([(res_dre, 'DRE_ML'), (res_naive, 'naive')]):
        acc_1  = res['accuracy']['stage1']
        acc_2  = res['accuracy']['stage2']
        cm_1   = res['confusion']['stage1']
        cm_2   = res['confusion']['stage2']
        V_true = res['value']['V_true']
        V_star = res['value']['V_star']
        regret = res['regret']['absolute']

        sns.heatmap(cm_1, annot=_cm_annot(cm_1), fmt='', cmap='Greys',
                    xticklabels=[f'est a={a}' for a in range(k1)],
                    yticklabels=[f'opt a={a}' for a in range(k1)],
                    ax=axes[row_idx, 0])
        axes[row_idx, 0].set_title(f'[{label}] Stage 1  (acc={acc_1:.1%})')
        axes[row_idx, 0].set_xlabel('Estimated $d^*_1$')
        axes[row_idx, 0].set_ylabel('Optimal $d^*_1$')

        sns.heatmap(cm_2, annot=_cm_annot(cm_2), fmt='', cmap='Greys',
                    xticklabels=[f'est a={a}' for a in range(k2)],
                    yticklabels=[f'opt a={a}' for a in range(k2)],
                    ax=axes[row_idx, 1])
        axes[row_idx, 1].set_title(f'[{label}] Stage 2  (acc={acc_2:.1%})')
        axes[row_idx, 1].set_xlabel('Estimated $d^*_2$')
        axes[row_idx, 1].set_ylabel('Optimal $d^*_2$')

        axes[row_idx, 2].bar(['V(d_optimal)', 'V(d_star)'], [V_true, V_star],
                             color=['steelblue', 'tomato'], width=0.4)
        axes[row_idx, 2].set_title(f'[{label}] Value  (regret={regret:.3f})')
        axes[row_idx, 2].set_ylabel('E[Y]')
        for bar, val in zip(axes[row_idx, 2].patches, [V_true, V_star]):
            axes[row_idx, 2].text(bar.get_x() + bar.get_width()/2,
                                  bar.get_height() + 0.01 * abs(V_true),
                                  f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save:
        out_path = os.path.join(images_dir, f'{filename}_assess.jpeg')
        fig.savefig(out_path)
        print(f"  Plot saved: {filename}_assess.jpeg")
    plt.close(fig)


# ============================================================
# Run over all datasets in _info_simple.csv
# ============================================================
if __name__ == '__main__':
    info    = pd.read_csv(os.path.join(datasets_dir, '_info_simple.csv'))
    results = []

    for _, row in info.iterrows():
        fname = row['filename']
        try:
            res_dre   = assess_otr(fname, model='DRE_ML', verbose=False)
            res_naive = assess_otr(fname, model='naive',  verbose=False)

            plot_comparison(fname, res_dre, res_naive, save=True)

            for model, res in [('DRE_ML', res_dre), ('naive', res_naive)]:
                results.append({
                    'filename':        fname,
                    'model':           model,
                    'k':               row['k1'],
                    'flavor_Y':        row['flavor_Y'],
                    'acc_stage1':      res['accuracy']['stage1'],
                    'acc_stage2':      res['accuracy']['stage2'],
                    'acc_joint':       res['accuracy']['joint'],
                    'V_true':          res['value']['V_true'],
                    'V_star':          res['value']['V_star'],
                    'regret':          res['regret']['absolute'],
                    'regret_relative': res['regret']['relative'],
                })
        except FileNotFoundError as e:
            print(f"Skipping {fname}: {e}")

    summary = pd.DataFrame(results)
    out_path = os.path.join(datasets_dir, '_assessment_summary_simple.csv')
    summary.to_csv(out_path, index=False)
    print(f"\n✓ Summary saved to _assessment_summary_simple.csv")
    print(summary.groupby(['model', 'k', 'flavor_Y'])[
        ['acc_stage1', 'acc_stage2', 'acc_joint', 'regret', 'regret_relative']
    ].mean().round(4))
