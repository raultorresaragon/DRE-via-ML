# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: run_sims_k3plus.py
# Date: 2026-01-08
# Note: This script runs M simulations of
#       k=k given DGPs and gamma trt effect
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import time
import os
from itertools import product
from scipy.stats import norm
from itertools import combinations

# Import custom functions
from one_sim import one_sim
from get_type1error import get_type1error

# Set random seed
np.random.seed(1857)

# Set parameters
export_tables = True
export_images = True
zero_effect = True
root = f"./_{'1' if not zero_effect else '0'}trt_effect/"

M = 50  # Number of simulations
K = [2,3,5]                    #[2,3,5]               # Treatment levels to test
Y_params = ["ols", "expo", "lognormal"]     #["ols", "expo", "lognormal"] # Parametric model for outcome model
pflavs = ["l","t"]             #["l", "t"]            # DGP Propensity model flavors: logit, tanh
oflavs = ["e", "s", "l", "g"]  #["e", "s", "l", "g"]  # DGP Outcome model flavors: expo, sigmoid, lognormal, gamma

# Create flavor combinations
flavors = [p + o for p, o in product(pflavs, oflavs)]
if len(flavors) == 8:
    flavors = [flavors[i] for i in [0, 2, 5, 6, 7]]  # Select subset

print(f"Testing flavors: {flavors}")

# Create output directories
os.makedirs(f"{root}/datasets", exist_ok=True)
os.makedirs(f"{root}/tables", exist_ok=True)
os.makedirs(f"{root}/images", exist_ok=True)

for Y_param in Y_params:
    for k in K:
        # Set parameters based on k
        if k == 2:
            p = 3
        elif k == 3:
            p = 8
        elif k == 5:
            p = 12
    
        n = k * 200
        eps = [120, 180]
        penals = [0.001, 0.005]
        hidunits = [2, 8]
    
        print(f"\n{'='*50}")
        print(f"Running simulations for k={k}, n={n}, p={p}")
        print(f"{'='*50}")
    
        # Run simulations for each flavor
        for flav in flavors:
            print(f"\nProcessing flavor: {flav}")
        
            # Set flavor options
            if flav == "le":
                flavor_ops = ["logit", "expo", 1, 0.5]
            elif flav == "ls":
                flavor_ops = ["logit", "sigmoid", 1, 1]
            elif flav == "ll":
                flavor_ops = ["logit", "lognormal", 1, 1]
            elif flav == "lg":
                flavor_ops = ["logit", "gamma", 1, 0.5]
            elif flav == "te":
                flavor_ops = ["tanh", "expo", 1, 0.5]
            elif flav == "ts":
                flavor_ops = ["tanh", "sigmoid", 1, 1]
            elif flav == "tl":
                flavor_ops = ["tanh", "lognormal", 1, 1]
            elif flav == "tg":
                flavor_ops = ["tanh", "gamma", 1, 0.5]
        
            A_flavor, Y_flavor = flavor_ops[0], flavor_ops[1]
            beta_Y_scalar = flavor_ops[3]
            
            ### Only run DGP 'll' under outcome model lognormal ###
            if Y_param == "lognormal" and flav != "ll":
                continue
            

            print(f"DGP A_flavor: {A_flavor}, DGP Y_flavor: {Y_flavor}, Param outcome model: {Y_param}")
        
            # Initialize results storage
            mytable = None
            otr_table = None
            muhat_pooled_all = None
            muhat_pooled_all_param = None
        
            total_start_time = time.time()
        
            # Run M iterations
            for i in range(1, M + 1):
                print(f"\nIteration {i}/{M}")
            
                # Generate random parameters for this iteration
                rho = round(np.random.uniform(0.4, 0.6), 1)
                Xmu = np.round(np.random.uniform(-1, 1, p), 1)
            
                # Treatment model coefficients
                beta_A = np.vstack([
                    np.full((1, k-1), -0.1),
                    np.round(np.random.uniform(-1.5, 1.5, (p, k-1)), 1)
                ])
            
                # Outcome model coefficients
                beta_Y = np.concatenate([
                    [1], 
                    np.round(np.random.uniform(-1, 1, p), 1)
                ]) * beta_Y_scalar
            
                # Treatment effects
                gamma = np.array([0.6, 0.4, 0.75, 0.17])[:(k-1)] * (1 if not zero_effect else 0)
            
                # Run simulation
                iter_start_time = time.time()
            
                try:
                    r = one_sim(
                        n=n, p=p, Xmu=Xmu, iter=i, k=k, verbose=True,
                        A_flavor=A_flavor, beta_A=beta_A, gamma=gamma,
                        Y_flavor=Y_flavor, beta_Y=beta_Y,
                        Y_param=Y_param,
                        hidunits=hidunits, eps=eps, penals=penals,
                        export_images=export_images, root=root, rho=rho
                    )
                
                    iter_time = time.time() - iter_start_time
                    print(f"  ...run time: {iter_time/60:.2f} mins")
                
                    # Store results
                    print("Results:")
                    print(r['Vn_df'])
                    
                    if mytable is None:
                        mytable = r['my_k_row'].copy()
                        otr_table = pd.concat([r['Xnew_Vn'], r['Vn_df']], axis=1)
                        muhat_pooled_all = r['muhat_pooled'].copy()
                        muhat_pooled_all_param = r['muhat_pooled_param'].copy()
                    else:
                        mytable = pd.concat([mytable, r['my_k_row']], ignore_index=True)
                        new_otr = pd.concat([r['Xnew_Vn'], r['Vn_df']], axis=1)
                        otr_table = pd.concat([otr_table, new_otr], ignore_index=True)
                        muhat_pooled_all = pd.concat([muhat_pooled_all, r['muhat_pooled']], ignore_index=True)
                        muhat_pooled_all_param = pd.concat([muhat_pooled_all_param, r['muhat_pooled_param']], ignore_index=True)
                
                    # Clean up OTR table columns
                    otr_cols = ['dataset', 'OTR'] + [col for col in otr_table.columns 
                                                   if col.startswith('X') or col.startswith('V_')]
                    otr_table = otr_table[[col for col in otr_cols if col in otr_table.columns]]
                
                except Exception as e:
                    print(f"Error in iteration {i}: {str(e)}")
                    continue
        
            total_time = time.time() - total_start_time
            print(f"\nTotal run time: {total_time/60:.2f} mins")
        
            if mytable is not None:
                print(f"\nResults for k={k}_{A_flavor}_{Y_flavor}")
                print(mytable)
            
                # Export results
                if export_tables:
                    # Convert numeric columns
                    numeric_cols = [col for col in mytable.columns 
                                  if col not in ['dataset', 'estimate']]
                    for col in numeric_cols:
                        mytable[col] = pd.to_numeric(mytable[col], errors='coerce')
                
                    # Save tables
                    mytable.to_csv(
                        f"{root}/tables/simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}.csv",
                        index=False
                    )
                
                    otr_table.to_csv(
                        f"{root}/tables/OTR_simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}.csv",
                        index=False
                    )
                
                    muhat_pooled_all.to_csv(
                        f"{root}/tables/muhat_pooled_simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}_nn.csv",
                        index=False
                    )

                    muhat_pooled_all_param.to_csv(
                        f"{root}/tables/muhat_pooled_simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}_param.csv",
                        index=False
                    )
                
                    print(f"Tables saved for k={k}_{A_flavor}_{Y_flavor}")
            
            # This uses muhat_pooled_simk{k}_{A_flavor}_{Y_flavor}_est_with_{Y_param}.csv
            get_type1error(k, A_flavor, Y_flavor, Y_param, zero_effect) 

print("\nAll simulations completed!")
