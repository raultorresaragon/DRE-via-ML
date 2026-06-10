# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# eda_bmi.py
# Train/test split and EDA for the BMI RWD dataset.
#
# Split: 195 training / 5 test rows  (seed=617)
# Saves: train.csv, test.csv  in the RWD folder
#
# Figures (greyscale, saved to RWD/images/):
#   1. eda_outcomes_overall.jpeg     — baselineBMI | month4BMI | month12BMI
#   2. eda_outcomes_by_gender.jpeg   — same, grouped by gender (Female/Male)
#   3. eda_outcomes_by_race.jpeg     — same, grouped by race (White/Non-white)
#   4. eda_month4bmi_by_A1.jpeg      — month4BMI by A1 (CD | MR)
#   5. eda_month12bmi_by_A2.jpeg     — month12BMI by A2 (CD | MR)
#   6. eda_baseline_month4bmi_by_A1.jpeg  — baselineBMI & month4BMI by A1
#   7. eda_month4_month12bmi_by_A2.jpeg   — month4BMI & month12BMI by A2
#   8. eda_trajectories_by_gender.jpeg    — mean BMI trajectories by gender × A1 × A2
#   9. eda_trajectories_by_race.jpeg      — mean BMI trajectories by race × A1 × A2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'bmiData.csv')
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

SEED = 1810

# ── Load and split ────────────────────────────────────────────────────────────
df    = pd.read_csv(data_path)
train = df.sample(n=195, random_state=SEED)
test  = df.drop(train.index).reset_index(drop=True)
train = train.reset_index(drop=True)

train.to_csv(os.path.join(script_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(script_dir,  'test.csv'),  index=False)

print(f"Training set : {len(train)} rows")
print(f"Test set     : {len(test)} rows")

print("\n── Outcome summary (training) ──────────────────────────────────────")
print(train[['baselineBMI', 'month4BMI', 'month12BMI']].describe().round(2))

print("\n── Covariate counts (training) ─────────────────────────────────────")
print(f"  Gender:  Female={( train['gender']==0).sum()}  Male={(train['gender']==1).sum()}")
print(f"  Race:    White={ (train['race']  ==0).sum()}  Non-white={(train['race']==1).sum()}")
print(f"  A1:      CD={(train['A1']=='CD').sum()}  MR={(train['A1']=='MR').sum()}")
print(f"  A2:      CD={(train['A2']=='CD').sum()}  MR={(train['A2']=='MR').sum()}")


# ── Shared helpers ────────────────────────────────────────────────────────────
OUTCOMES = ['baselineBMI', 'month4BMI', 'month12BMI']
TP_LABELS = ['Baseline BMI', 'Month 4 BMI', 'Month 12 BMI']


def _style_bp(bp, colors):
    """Apply facecolors and force black lines on a patch_artist boxplot."""
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')
    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5, markersize=4)


def _grouped_boxplot(ax, data_groups, tp_labels, grp_colors, grp_labels, width=0.55):
    """
    Grouped boxplot across time points.

    data_groups : list of lists  [ [arr_g0, arr_g1], ... ]  one per time point
    tp_labels   : x-axis labels for each time point group
    grp_colors  : greyscale shade per within-group position
    grp_labels  : legend labels per group
    """
    n_tp  = len(data_groups)
    n_grp = len(data_groups[0])

    all_data   = []
    positions  = []
    tick_pos   = []
    color_list = []

    pos = 1.0
    for tp_idx in range(n_tp):
        grp_pos = []
        for g in range(n_grp):
            all_data.append(data_groups[tp_idx][g])
            positions.append(pos)
            color_list.append(grp_colors[g])
            grp_pos.append(pos)
            pos += 1.0
        tick_pos.append(np.mean(grp_pos))
        pos += 0.8   # gap between time-point groups

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=width)
    _style_bp(bp, color_list)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tp_labels, fontsize=11)
    ax.set_xlim(0.3, pos - 0.5)

    patches = [mpatches.Patch(facecolor=grp_colors[g], edgecolor='black',
                               alpha=0.85, label=grp_labels[g])
               for g in range(n_grp)]
    ax.legend(handles=patches, fontsize=10)
    ax.grid(axis='y', alpha=0.3)


# ── Figure 1: Overall outcomes ────────────────────────────────────────────────
colors_overall = ['0.82', '0.50', '0.20']

fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot([train[c].values for c in OUTCOMES], patch_artist=True, widths=0.5)
_style_bp(bp, colors_overall)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(TP_LABELS, fontsize=11)
ax.set_ylabel('BMI', fontsize=11)
ax.set_title('BMI outcomes over time  (training data)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_outcomes_overall.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: eda_outcomes_overall.jpeg")


# ── Figure 2: By gender ───────────────────────────────────────────────────────
data_gender = [
    [train.loc[train['gender'] == g, c].values for g in [0, 1]]
    for c in OUTCOMES
]

fig, ax = plt.subplots(figsize=(7, 5))
_grouped_boxplot(ax, data_gender, TP_LABELS,
                 grp_colors=['0.75', '0.25'],
                 grp_labels=['Female', 'Male'])
ax.set_ylabel('BMI', fontsize=11)
ax.set_title('BMI outcomes over time by gender  (training data)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_outcomes_by_gender.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_outcomes_by_gender.jpeg")


# ── Figure 3: By race ─────────────────────────────────────────────────────────
data_race = [
    [train.loc[train['race'] == r, c].values for r in [0, 1]]
    for c in OUTCOMES
]

fig, ax = plt.subplots(figsize=(7, 5))
_grouped_boxplot(ax, data_race, TP_LABELS,
                 grp_colors=['0.75', '0.25'],
                 grp_labels=['White', 'Non-white'])
ax.set_ylabel('BMI', fontsize=11)
ax.set_title('BMI outcomes over time by race  (training data)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_outcomes_by_race.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_outcomes_by_race.jpeg")


# ── Figure 4: month4BMI by A1 ─────────────────────────────────────────────────
arms = ['CD', 'MR']
colors_trt = ['0.30', '0.70']

fig, ax = plt.subplots(figsize=(4, 5))
bp = ax.boxplot([train.loc[train['A1'] == a, 'month4BMI'].values for a in arms],
                patch_artist=True, widths=0.5)
_style_bp(bp, colors_trt)
ax.set_xticks([1, 2])
ax.set_xticklabels(arms, fontsize=11)
ax.set_ylabel('Month 4 BMI', fontsize=11)
ax.set_title('Month 4 BMI by A1  (training data)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_month4bmi_by_A1.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_month4bmi_by_A1.jpeg")


# ── Figure 5: month12BMI by A2 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 5))
bp = ax.boxplot([train.loc[train['A2'] == a, 'month12BMI'].values for a in arms],
                patch_artist=True, widths=0.5)
_style_bp(bp, colors_trt)
ax.set_xticks([1, 2])
ax.set_xticklabels(arms, fontsize=11)
ax.set_ylabel('Month 12 BMI', fontsize=11)
ax.set_title('Month 12 BMI by A2  (training data)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_month12bmi_by_A2.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_month12bmi_by_A2.jpeg")

# ── Figure 6: baselineBMI and month4BMI by A1 ─────────────────────────────────
data_by_A1 = [
    [train.loc[train['A1'] == a, c].values for a in arms]
    for c in ['baselineBMI', 'month4BMI']
]

fig, ax = plt.subplots(figsize=(5, 5))
_grouped_boxplot(ax, data_by_A1,
                 tp_labels=['Baseline BMI', 'Month 4 BMI'],
                 grp_colors=['0.30', '0.70'],
                 grp_labels=['CD', 'MR'])
ax.set_ylabel('BMI', fontsize=11)
ax.set_title('Baseline and Month 4 BMI by A1  (training data)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_baseline_month4bmi_by_A1.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_baseline_month4bmi_by_A1.jpeg")

# ── Figure 7: month4BMI and month12BMI by A2 ──────────────────────────────────
data_by_A2 = [
    [train.loc[train['A2'] == a, c].values for a in arms]
    for c in ['month4BMI', 'month12BMI']
]

fig, ax = plt.subplots(figsize=(5, 5))
_grouped_boxplot(ax, data_by_A2,
                 tp_labels=['Month 4 BMI', 'Month 12 BMI'],
                 grp_colors=['0.30', '0.70'],
                 grp_labels=['CD', 'MR'])
ax.set_ylabel('BMI', fontsize=11)
ax.set_title('Month 4 and Month 12 BMI by A2  (training data)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'eda_month4_month12bmi_by_A2.jpeg'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: eda_month4_month12bmi_by_A2.jpeg")

# ── Trajectory helper ─────────────────────────────────────────────────────────
def _trajectory_plot(train, group_col, group_map, title, save_name):
    """
    Mean BMI trajectory plot over three time points.

    Segment 1  baseline → month4   : mean at (group × A1) level
    Segment 2  month4   → month12  : mean at (group × A1 × A2) level
                                     (lines fork at month4)

    group_col : column name ('gender' or 'race')
    group_map : {value: (label, grey_shade)}  e.g. {0: ('Female', '0.05')}
    """
    # One marker per (A1, A2) combination; line style also tracks A1
    combo_styles = {
        ('CD', 'CD'): ('-',  'o'),
        ('CD', 'MR'): ('-',  '^'),
        ('MR', 'CD'): ('--', 's'),
        ('MR', 'MR'): ('--', '*'),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    legend_handles = []

    for g_val, (g_label, g_color) in group_map.items():
        for (a1, a2), (a1_ls, mk) in combo_styles.items():
            mask = (
                (train[group_col] == g_val) &
                (train['A1'] == a1) &
                (train['A2'] == a2)
            )
            y_base = train.loc[mask, 'baselineBMI'].mean()
            y_m4   = train.loc[mask, 'month4BMI'].mean()
            y_m12  = train.loc[mask, 'month12BMI'].mean()

            ax.plot([0, 1, 2], [y_base, y_m4, y_m12],
                    color=g_color, linestyle=a1_ls, linewidth=2,
                    marker=mk, markersize=7, zorder=2)

            legend_handles.append(
                plt.Line2D([0], [0],
                           color=g_color, linestyle=a1_ls, linewidth=2,
                           marker=mk, markersize=7,
                           label=f'{g_label}, A1={a1}, A2={a2}')
            )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Baseline BMI', 'Month 4 BMI', 'Month 12 BMI'], fontsize=11)
    ax.set_ylabel('Mean BMI', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(handles=legend_handles, fontsize=9, loc='best',
              framealpha=0.8, edgecolor='0.7')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_name}")


# ── Figure 8: Trajectories by gender ─────────────────────────────────────────
_trajectory_plot(
    train, 'gender',
    group_map={0: ('Female', '0.05'), 1: ('Male', '0.50')},
    title='Mean BMI trajectories by gender × A1 × A2  (training data)',
    save_name='eda_trajectories_by_gender.jpeg',
)

# ── Figure 9: Trajectories by race ───────────────────────────────────────────
_trajectory_plot(
    train, 'race',
    group_map={0: ('White', '0.05'), 1: ('Non-white', '0.50')},
    title='Mean BMI trajectories by race × A1 × A2  (training data)',
    save_name='eda_trajectories_by_race.jpeg',
)

print("\nDone.")
