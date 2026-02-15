# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: get_typeI_error_rate_NN.py
# Date: 2026-01-24
# Note: Script to calculate the proportion of simulated datasets where the 
#       estimator has p-value < 0.05.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import glob
import numpy as np
import pandas as pd

def get_typeIerror_rate(file_path):
    """
    Calculate the proportion of simulated datasets wherep-value <= 0.05.
    """
    df = pd.read_csv(file_path)
    n_row = len(df)
    cols = df.columns.tolist()
    df.loc[n_row, cols[0]] = 'Type_I_error_rate' 
    for c in cols[1:len(cols)]:
        type_I_error = np.mean(df[c]<0.05)
        df.loc[n_row, c] = type_I_error

    return df


for file_path in glob.glob('./_0trt_effect/tables/Results/Type I error rates/*.csv'):                              
    result = get_typeIerror_rate(file_path)                                                                        
    result.to_csv(file_path, index=False)  # overwrite with new row added  
