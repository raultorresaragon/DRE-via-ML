# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# analysis_bmi.py
# DRE-ML on the BMI RWD dataset (two-stage).
#
# Reads train.csv / test.csv produced by eda_bmi.py.
# Fits DRE-ML on the 195-row training set (per estimate_dre_two_stage.py).
# Computes V_n on the 5-row test set using stage-2 predicted outcomes.
# Saves a greyscale bar chart of V(d*): DRE-ML vs Observed Y.
#
# Variable mapping
# ----------------
#   X1   : gender, race, parentBMI, baselineBMI  (baseline covariates)
#   X2   : (none — no new covariates collected at month 4)
#   Y1   : month4BMI   (intermediate outcome)
#   Y    : month12BMI  (final outcome)
#   A1/A2: CD=0, MR=1  (binary treatments)
#   k    : 2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import product

# ── Path setup — import NN helpers from two_stages/ ─────────────────────────
script_dir     = os.path.dirname(os.path.abspath(__file__))
two_stages_dir = os.path.join(script_dir, '../two_stages')
sys.path.insert(0, two_stages_dir)

from Y_nn_tuning import Y_model_nn
from A_nn_tuning import A_model_nn
from estimate_dre_two_stage import compute_mu_hat

images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# ── Tuning parameters (same as two_stages/ A_nn_tuning / Y_nn_tuning) ───────
HIDUNITS = [15, 18, 20, 22, 30, 34] #[38, 39, 85, 107, 109, 115]
EPS      = [80]
PENALS   = [0.00002, 0.00003, 0.04, 0.2, 0.3, 0.4]
CV_FOLDS = 4    # number of cross-validation folds (default in tuning scripts is 6)
K        = 2    # binary treatments


# ── Helpers (mirrors _fit_outcome_nn / _fit_pscore_nn) ───────────────────────
def _fit_Y(features_df, y_values, tag=''):
    dat = features_df.copy()
    dat['Y'] = y_values
    n = len(dat)
    print(f"    fitting outcome NN {tag}(n={n})...")
    small = n < 30
    return Y_model_nn(dat=dat, hidunits=HIDUNITS, eps=EPS, penals=PENALS,
                      cvs=min(3, n // 2) if small else CV_FOLDS,
                      early_stopping=not small, verbose=True)


def _fit_A(features_df, a_values, tag=''):
    dat = features_df.copy()
    dat['A'] = a_values.astype(int)
    print(f"    fitting pscore NN {tag}...")
    return A_model_nn(dat=dat, hidunits=HIDUNITS, eps=EPS, penals=PENALS, cvs=CV_FOLDS,
                      verbose=True)


# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(script_dir, 'train.csv'))
test  = pd.read_csv(os.path.join(script_dir, 'test.csv'))

# Codify treatments: CD=0, MR=1
for df in [train, test]:
    df['A1'] = (df['A1'] == 'MR').astype(int)
    df['A2'] = (df['A2'] == 'MR').astype(int)

X1_cols = ['gender', 'race', 'parentBMI', 'baselineBMI']

X1_tr = train[X1_cols].reset_index(drop=True)
A1_tr = train['A1'].values
A2_tr = train['A2'].values
Y1_tr = train['month4BMI'].values
Y_tr  = train['month12BMI'].values
n_tr  = len(train)

X1_te = test[X1_cols].reset_index(drop=True)
A1_te = test['A1'].values
A2_te = test['A2'].values
Y1_te = test['month4BMI'].values
Y_te  = test['month12BMI'].values
n_te  = len(test)

print(f"Training: {n_tr} rows  |  Test: {n_te} rows")

# ── Stage 1: outcome models ──────────────────────────────────────────────────
print(f"\n{'='*55}\n[Stage 1 — outcome models]\n{'='*55}")
Y1_hat_tr = np.zeros((n_tr, K))
Y1_hat_te = np.zeros((n_te, K))

for a in range(K):
    mask    = (A1_tr == a)
    model_a = _fit_Y(X1_tr[mask].reset_index(drop=True), Y1_tr[mask], tag=f'Y1 a={a} ')
    Y1_hat_tr[:, a] = model_a.predict(X1_tr)
    Y1_hat_te[:, a] = model_a.predict(X1_te)

# ── Stage 1: propensity scores ───────────────────────────────────────────────
print(f"\n{'='*55}\n[Stage 1 — propensity score]\n{'='*55}")
pscore1  = _fit_A(X1_tr, A1_tr, tag='A1 ')
pi1_tr   = pscore1.predict_proba(X1_tr)   # (n_tr, 2)

mu_hat_1    = compute_mu_hat(A1_tr, Y1_tr, Y1_hat_tr, pi1_tr, K)
d_star_1_tr = np.argmax(mu_hat_1, axis=1)
print(f"\n  d_star_1 (train): CD={( d_star_1_tr==0).sum()}  MR={(d_star_1_tr==1).sum()}")

# ── Stage 2: outcome models ──────────────────────────────────────────────────
# Features: [X1, Y1_resid_a]   (no X2 in this dataset)
print(f"\n{'='*55}\n[Stage 2 — outcome models]\n{'='*55}")
Y2_hat_tr = np.zeros((n_tr, K))
Y2_hat_te = np.zeros((n_te, K))

for a in range(K):
    resid_tr = Y1_tr - Y1_hat_tr[:, a]
    resid_te = Y1_te - Y1_hat_te[:, a]

    feat_tr = pd.concat([X1_tr,
                         pd.Series(resid_tr, name='Y1_resid')], axis=1)
    feat_te = pd.concat([X1_te.reset_index(drop=True),
                         pd.Series(resid_te, name='Y1_resid')], axis=1)

    mask    = (A2_tr == a)
    model_a = _fit_Y(feat_tr[mask].reset_index(drop=True), Y_tr[mask], tag=f'Y2 a={a} ')
    Y2_hat_tr[:, a] = model_a.predict(feat_tr)
    Y2_hat_te[:, a] = model_a.predict(feat_te)

# ── Stage 2: propensity scores ───────────────────────────────────────────────
# Features: [X1, A1, Y1]   (no X2)
print(f"\n{'='*55}\n[Stage 2 — propensity score]\n{'='*55}")
feat_ps2 = pd.concat([X1_tr,
                      pd.Series(A1_tr, name='A1'),
                      pd.Series(Y1_tr, name='Y1')], axis=1)
pscore2  = _fit_A(feat_ps2, A2_tr, tag='A2 ')
pi2_tr   = pscore2.predict_proba(feat_ps2)   # (n_tr, 2)

mu_hat_2    = compute_mu_hat(A2_tr, Y_tr, Y2_hat_tr, pi2_tr, K)
d_star_2_tr = np.argmax(mu_hat_2, axis=1)
print(f"\n  d_star_2 (train): CD={( d_star_2_tr==0).sum()}  MR={(d_star_2_tr==1).sum()}")

# ── ATE estimates (DRE-ML, training set) ────────────────────────────────────
# ATE_t = mean(mu_hat_t_a1 - mu_hat_t_a0)   i.e. MR vs CD at each stage
ATE_1 = float(np.mean(mu_hat_1[:, 1] - mu_hat_1[:, 0]))
ATE_2 = float(np.mean(mu_hat_2[:, 1] - mu_hat_2[:, 0]))

print(f"\n{'='*55}")
print(f"ATE estimates — DRE-ML  (MR vs CD, training set)")
print(f"{'='*55}")
print(f"  Stage 1  ATE(A1: MR vs CD) on month4BMI  : {ATE_1:.4f}")
print(f"  Stage 2  ATE(A2: MR vs CD) on month12BMI : {ATE_2:.4f}")

ate_df = pd.DataFrame({
    'stage':     [1, 2],
    'contrast':  ['MR vs CD', 'MR vs CD'],
    'outcome':   ['month4BMI', 'month12BMI'],
    'ATE_DRE_ML': [ATE_1, ATE_2],
})
ate_df.to_csv(os.path.join(script_dir, 'ate_estimates.csv'), index=False)
print("\n  Saved: ate_estimates.csv")

# ── V_n on test set ──────────────────────────────────────────────────────────
# d_star on test: argmax of predicted Y2_hat (no mu_hat needed — no IPW on test)
d_star_1_te = np.argmax(Y1_hat_te, axis=1)
d_star_2_te = np.argmax(Y2_hat_te, axis=1)

Vn_dre = float(np.mean(Y2_hat_te[np.arange(n_te), d_star_2_te]))
Vn_obs = float(test['month12BMI'].mean())

print(f"\n{'='*55}")
print(f"V_n results  (test set, n={n_te})")
print(f"{'='*55}")
print(f"  DRE-ML  d_star_1: {['CD','MR'][d_star_1_te[0]]} ... {d_star_1_te.tolist()}")
print(f"  DRE-ML  d_star_2: {['CD','MR'][d_star_2_te[0]]} ... {d_star_2_te.tolist()}")
print(f"  V_n DRE-ML   : {Vn_dre:.4f}")
print(f"  V_n Observed : {Vn_obs:.4f}")

# ── d_star frequency table (test set) ────────────────────────────────────────
arm_labels = {0: 'CD', 1: 'MR'}
A1_te_orig = test['A1'].values   # already coded 0/1
A2_te_orig = test['A2'].values
n_te_freq  = n_te
freq_rows  = []

# Stage-1 marginal
for a in range(K):
    freq_rows.append({
        'd_star':        f'{{{arm_labels[a]}}}',
        'observed_A':    round((A1_te_orig == a).sum() / n_te_freq, 4),
        'd_star_DRE-ML': round((d_star_1_te == a).sum() / n_te_freq, 4),
    })

# Joint (d1, d2)
for a1, a2 in product(range(K), range(K)):
    obs_mask = (A1_te_orig == a1) & (A2_te_orig == a2)
    dre_mask = (d_star_1_te == a1) & (d_star_2_te == a2)
    freq_rows.append({
        'd_star':        f'{{{arm_labels[a1]},{arm_labels[a2]}}}',
        'observed_A':    round(obs_mask.sum() / n_te_freq, 4),
        'd_star_DRE-ML': round(dre_mask.sum() / n_te_freq, 4),
    })

freq_df = pd.DataFrame(freq_rows)
freq_path = os.path.join(script_dir, 'dstar_freq_test.csv')
freq_df.to_csv(freq_path, index=False)
print(f"\n{'='*55}")
print(f"d_star frequency table  (test set, n={n_te_freq})")
print(f"{'='*55}")
print(freq_df.to_string(index=False))
print(f"\n  Saved: dstar_freq_test.csv")

# ── Paired t-test: DRE-ML per-subject V_n vs observed month12BMI ──────────────
vn_arr  = Y2_hat_te[np.arange(n_te), d_star_2_te]   # predicted Y under d_star
obs_arr = test['month12BMI'].values

tstat, pval = stats.ttest_rel(vn_arr, obs_arr)

def _stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1:  return '*'
    return 'n.s.'

print(f"\n{'='*55}")
print(f"Paired t-test: DRE-ML V_n vs Observed month12BMI  (n={n_te})")
print(f"{'='*55}")
print(f"  Mean DRE-ML  : {vn_arr.mean():.4f}")
print(f"  Mean Obs Y   : {obs_arr.mean():.4f}")
print(f"  Difference   : {vn_arr.mean() - obs_arr.mean():.4f}")
print(f"  t-statistic  : {tstat:.4f}")
print(f"  p-value      : {pval:.4f}  {_stars(pval)}")

# ── Bar plot ─────────────────────────────────────────────────────────────────
vals   = {'DRE-ML': Vn_dre, 'Obs Y': Vn_obs}
labels = list(vals.keys())
values = list(vals.values())
colors = ['0.20', '0.82']
v_max  = max(values)

fig, ax = plt.subplots(figsize=(4, 4))
bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.85,
              edgecolor='black', linewidth=0.8)
for bar, lbl, val in zip(bars, labels, values):
    annot = f'{val:.4f}{_stars(pval) if lbl == "DRE-ML" and _stars(pval) != "n.s." else ""}'
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 * abs(v_max),
            annot, ha='center', va='bottom', fontsize=9)
ax.set_title('V(d*) by model on test data (BMI)', fontsize=11)
ax.set_ylabel('Mean predicted month 12 BMI under policy', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_path = os.path.join(images_dir, 'vplot_bmi_dre.jpeg')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: vplot_bmi_dre.jpeg")
print("\nDone.")
