# ----------
# Compute Vn (Optimal Treatment Regime)
# ----------

import pandas as pd
import numpy as np

def get_Vn(fit_Y_nn, X_new):
    """
    Compute optimal treatment regime for new observations

    Parameters:
    - fit_Y_nn: dictionary of fitted outcome models keyed by 'A_ji'
                each value is [results, g_i, g_j, ghat_j, ghat_i]
                where g_i is the model for the higher-indexed treatment (i)
                and   g_j is the model for the lower-indexed treatment (j)
    - X_new: new covariate matrix for prediction

    Returns:
    - DataFrame with per-arm averaged predicted values and OTR column
    """
    # Collect predictions per treatment arm across all pairwise comparisons
    preds = {}
    for A_type, fit in fit_Y_nn.items():
        j = int(A_type[2])  # lower treatment index  → model is fit[2]
        i = int(A_type[3])  # higher treatment index → model is fit[1]
        preds.setdefault(i, []).append(fit[1].predict(X_new))
        preds.setdefault(j, []).append(fit[2].predict(X_new))

    # Average predictions across comparisons for each arm, then take argmax
    V_n = pd.DataFrame({
        f'V_g{a}': np.mean(preds[a], axis=0) for a in sorted(preds)
    })
    V_n['OTR'] = V_n.idxmax(axis=1).str.extract(r'_g(\d+)')[0]

    return V_n