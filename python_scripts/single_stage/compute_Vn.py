# ----------
# Compute Vn (Optimal Treatment Regime)
# ----------

import pandas as pd
import numpy as np

def get_Vn(fit_Y_nn, X_new):
    """
    Compute optimal treatment regime for new observations
    
    Parameters:
    - fit_Y_nn: dictionary of fitted outcome models
    - X_new: new covariate matrix for prediction
    
    Returns:
    - DataFrame with predicted values and optimal treatment assignments
    """
    
    V_n = pd.DataFrame({'V_': [np.nan] * X_new.shape[0]})
    
    # Predict outcomes for each treatment comparison
    for A_type in fit_Y_nn.keys():
        for j in [1, 2]:  # Corresponds to g_i and g_j models
            if j < len(fit_Y_nn[A_type]):
                model = fit_Y_nn[A_type][j]
                
                # Predict using the model
                V_pred = model.predict(X_new)
                
                # Create column name
                V_type = A_type.replace('A', 'V')
                treatment_idx = A_type[2] if j == 1 else A_type[3]  # Extract treatment index
                col_name = f"{V_type}_g{treatment_idx}"
                
                V_n[col_name] = V_pred
    
    # Remove the initial dummy column
    V_n = V_n.drop('V_', axis=1)
    
    # Find optimal treatment (column with maximum predicted value)
    if len(V_n.columns) > 0:
        optimal_cols = V_n.idxmax(axis=1)
        # Extract treatment assignment from column name
        V_n['OTR'] = optimal_cols.str.extract(r'_g(\d+)')[0]
    else:
        V_n['OTR'] = '0'
    
    return V_n