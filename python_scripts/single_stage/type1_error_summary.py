# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# type1_error_summary.py
# Collect Type_I_error_rate rows from all type_1_error_simk*.csv files and produce
# one summary table per k.
#
# Columns:  DGP | model | delta_01 [| delta_02 [| delta_03 | delta_04]]
#   DGP   : "{A_flavor}-{Y_flavor}"
#   model : NN / OLS / Expo / Lognormal
#   delta_XY: pooled-variance type I error rate for baseline comparison arm-0 vs arm-Y
#
# Output: _0trt_effect/tables/Results/type1_error_summary_k{k}.csv  (one per k found)
#
# Run from: python_scripts/single_stage/
# Prerequisite: typeIerrorNN.py must have been run first (adds Type_I_error_rate row).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import glob
import os
import re
import pandas as pd

input_dir  = './_0trt_effect/tables/Results/Type I error rates'
output_dir = './_0trt_effect/tables/Results'

SUFFIX_TO_MODEL = {
    'nn':              'NN',
    'ols_param':       'OLS',
    'expo_param':      'Expo',
    'lognormal_param': 'Lognormal',
}

# Filename pattern: type_1_error_simk{k}_{A_flavor}_{Y_flavor}_est_with_{suffix}.csv
_FNAME_RE = re.compile(
    r'^type_1_error_simk(\d+)_(\w+)_(\w+)_est_with_(.+)\.csv$'
)


def parse_fname(fname):
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return {
        'k':        int(m.group(1)),
        'A_flavor': m.group(2),
        'Y_flavor': m.group(3),
        'suffix':   m.group(4),
    }


files = sorted(glob.glob(os.path.join(input_dir, 'type_1_error_simk*.csv')))
if not files:
    raise FileNotFoundError(f'No files found in: {input_dir}')

records = []
for fpath in files:
    info = parse_fname(os.path.basename(fpath))
    if info is None:
        print(f'  Skipping (unrecognised name): {os.path.basename(fpath)}')
        continue

    df = pd.read_csv(fpath)

    # Locate the Type_I_error_rate row (added by typeIerrorNN.py)
    mask = df['dataset'].astype(str) == 'Type_I_error_rate'
    if not mask.any():
        print(f'  No Type_I_error_rate row — skipping: {os.path.basename(fpath)}')
        continue

    row   = df[mask].iloc[0]
    k     = info['k']
    model = SUFFIX_TO_MODEL.get(info['suffix'], info['suffix'])

    record = {
        'k':     k,
        'DGP':   f"{info['A_flavor']}-{info['Y_flavor']}",
        'model': model,
    }

    # Extract pooled-variance rates for baseline comparisons only (arm 0 vs arm i)
    for i in range(1, k):
        col = f'pvals_pooled_var_0{i}'
        record[f'delta_0{i}'] = row[col] if col in row.index else None

    records.append(record)

if not records:
    print('No Type_I_error_rate rows found. Did you run typeIerrorNN.py first?')
else:
    all_df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)

    for k_val, grp in all_df.groupby('k'):
        out = grp.drop(columns='k').reset_index(drop=True)
        # Sort: DGP then model
        out = out.sort_values(['DGP', 'model']).reset_index(drop=True)
        out_path = os.path.join(output_dir, f'type1_error_summary_k{k_val}.csv')
        out.to_csv(out_path, index=False)
        print(f'\nk={k_val}  →  {out_path}')
        print(out.to_string(index=False))

    print('\nDone.')
