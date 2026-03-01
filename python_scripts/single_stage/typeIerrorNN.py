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
      df = pd.read_csv(file_path)                                                                                      
      cols = df.columns.tolist()                                                                                       
                                                                                                                       
      # Remove existing Type_I_error_rate row if present                                                               
      if len(df) > 0 and df.iloc[-1, 0] == 'Type_I_error_rate':                                                        
          df = df.iloc[:-1]                                                                                            
                                                                                                                       
      # Calculate BEFORE adding new row                                                                                
      n_row = len(df)                                                                                                  
      type_I_errors = {}                                                                                               
      for c in cols[1:]:                                                                                               
          numeric_col = pd.to_numeric(df[c], errors='coerce')                                                          
          type_I_errors[c] = np.round(np.mean(numeric_col < 0.05), 2)                                   
                                                                                                                       
      # Add the row                                                                                                    
      df.loc[n_row, cols[0]] = 'Type_I_error_rate'                                                                     
      for c in cols[1:]:                                                                                               
          df.loc[n_row, c] = type_I_errors[c]                                                                          
                                                                                                                       
      return df 

for file_path in glob.glob('./_0trt_effect/tables/Results/Type I error rates/*.csv'):                              
    result = get_typeIerror_rate(file_path)                                                                        
    result.to_csv(file_path, index=False)  # overwrite with new row added  