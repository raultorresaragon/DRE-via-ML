# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: run_sims_two_stage.py
# Date: 2026-02-19
# Note: This script runs multiple simulations for two-stage DTR and aggregates results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from one_sim_two_stage import one_sim_two_stage

def run_sims_two_stage(M, n, p1, p2, k1, k2,
                       beta_A1, beta_A2,
                       gamma1_X2, beta_X2,
                       gamma1_Y, gamma2_Y, beta_Y,
                       A_flavor, Y_flavor,
                       hidunits=[5, 20], eps=[100, 250], penals=[0.001, 0.01],
                       verbose=False, export_images=False, root="./", rho=0.5):
    """
    Run M simulations for two-stage DTR
    
    Parameters:
    M = number of simulations
    (other parameters same as one_sim_two_stage)
    
    Returns:
    - Aggregated results across simulations
    """
    
    print("=" * 80)
    print(f"RUNNING {M} TWO-STAGE DTR SIMULATIONS")
    print("=" * 80)
    print(f"n={n}, p1={p1}, p2={p2}, k1={k1}, k2={k2}")
    print(f"A_flavor={A_flavor}, Y_flavor={Y_flavor}")
    print("=" * 80)
    
    # Storage for results
    results_list = []
    
    # Run simulations
    for m in range(1, M + 1):
        print(f"\n{'='*80}")
        print(f"Simulation {m}/{M}")
        print(f"{'='*80}")
        
        # Set seed for reproducibility
        np.random.seed(m * 100)
        
        # Run one simulation
        result = one_sim_two_stage(
            n=n, p1=p1, p2=p2, k1=k1, k2=k2,
            beta_A1=beta_A1, beta_A2=beta_A2,
            gamma1_X2=gamma1_X2, beta_X2=beta_X2,
            gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y, beta_Y=beta_Y,
            A_flavor=A_flavor, Y_flavor=Y_flavor,
            hidunits=hidunits, eps=eps, penals=penals,
            verbose=verbose, iter=m, export_images=export_images, root=root, rho=rho
        )
        
        # Extract key metrics
        dat = result['dataset']
        
        sim_result = {
            'sim': m,
            'observed_mean_Y': dat['Y'].mean(),
            'observed_sd_Y': dat['Y'].std(),
            # NN results
            'nn_expected_value': np.mean(result['regime_nn']['Y_tilde']),
            'nn_optimal_A1_mode': np.argmax(np.bincount(result['regime_nn']['optimal_A1'])),
            'nn_optimal_A2_mode': np.argmax(np.bincount(result['regime_nn']['optimal_A2'])),
            # OLS results
            'ols_expected_value': np.mean(result['regime_ols']['Y_tilde']),
            'ols_optimal_A1_mode': np.argmax(np.bincount(result['regime_ols']['optimal_A1'])),
            'ols_optimal_A2_mode': np.argmax(np.bincount(result['regime_ols']['optimal_A2'])),
            # Expo results
            'expo_expected_value': np.mean(result['regime_expo']['Y_tilde']),
            'expo_optimal_A1_mode': np.argmax(np.bincount(result['regime_expo']['optimal_A1'])),
            'expo_optimal_A2_mode': np.argmax(np.bincount(result['regime_expo']['optimal_A2'])),
        }
        
        # Lognormal if applicable
        if result['regime_lognormal'] is not None:
            sim_result['lognormal_expected_value'] = np.mean(result['regime_lognormal']['Y_tilde'])
            sim_result['lognormal_optimal_A1_mode'] = np.argmax(np.bincount(result['regime_lognormal']['optimal_A1']))
            sim_result['lognormal_optimal_A2_mode'] = np.argmax(np.bincount(result['regime_lognormal']['optimal_A2']))
        
        results_list.append(sim_result)
    
    # Aggregate results
    results_df = pd.DataFrame(results_list)
    
    # Save results
    os.makedirs(f"{root}/results", exist_ok=True)
    output_file = f"{root}/results/two_stage_k{k1}k{k2}_{A_flavor}_{Y_flavor}_M{M}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS SIMULATIONS")
    print("=" * 80)
    
    print("\nObserved outcomes:")
    print(f"  Mean Y: {results_df['observed_mean_Y'].mean():.2f} (SD: {results_df['observed_mean_Y'].std():.2f})")
    
    print("\nExpected values under optimal regimes:")
    print(f"  NN:   {results_df['nn_expected_value'].mean():.2f} (SD: {results_df['nn_expected_value'].std():.2f})")
    print(f"  OLS:  {results_df['ols_expected_value'].mean():.2f} (SD: {results_df['ols_expected_value'].std():.2f})")
    print(f"  Expo: {results_df['expo_expected_value'].mean():.2f} (SD: {results_df['expo_expected_value'].std():.2f})")
    if 'lognormal_expected_value' in results_df.columns:
        print(f"  Lognormal: {results_df['lognormal_expected_value'].mean():.2f} (SD: {results_df['lognormal_expected_value'].std():.2f})")
    
    print("\nMost common optimal treatments (mode across simulations):")
    print(f"  NN:   A1={results_df['nn_optimal_A1_mode'].mode()[0]}, A2={results_df['nn_optimal_A2_mode'].mode()[0]}")
    print(f"  OLS:  A1={results_df['ols_optimal_A1_mode'].mode()[0]}, A2={results_df['ols_optimal_A2_mode'].mode()[0]}")
    print(f"  Expo: A1={results_df['expo_optimal_A1_mode'].mode()[0]}, A2={results_df['expo_optimal_A2_mode'].mode()[0]}")
    if 'lognormal_optimal_A1_mode' in results_df.columns:
        print(f"  Lognormal: A1={results_df['lognormal_optimal_A1_mode'].mode()[0]}, A2={results_df['lognormal_optimal_A2_mode'].mode()[0]}")
    
    print("\n" + "=" * 80)
    print(f"✓ Completed {M} simulations successfully!")
    print("=" * 80)
    
    return results_df


# ========================================
# Example usage
# ========================================
if __name__ == "__main__":
    
    # Parameters
    M = 5  # Number of simulations
    n = 200
    p1 = 3
    p2 = 2
    k1 = 3
    k2 = 3
    
    # Coefficients
    beta_A1 = np.array([[0.5, 0.3], [-0.3, 0.4], [0.2, -0.1], [0.1, 0.2]])
    beta_A2 = np.array([[0.3, 0.2], [-0.2, 0.3], [0.1, -0.1], [0.2, 0.1], [0.4, -0.2], [-0.3, 0.5], [0.1, -0.2]])
    
    gamma1_X2 = np.array([0.5, 1.0])
    beta_X2 = np.array([0.0, 0.3, 0.2, 0.1])
    
    gamma1_Y = np.array([1.0, 2.0])
    gamma2_Y = np.array([1.5, 3.0])
    beta_Y = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.3])
    
    # Run simulations
    results = run_sims_two_stage(
        M=M, n=n, p1=p1, p2=p2, k1=k1, k2=k2,
        beta_A1=beta_A1, beta_A2=beta_A2,
        gamma1_X2=gamma1_X2, beta_X2=beta_X2,
        gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y, beta_Y=beta_Y,
        A_flavor="logit", Y_flavor="expo",
        hidunits=[5, 10], eps=[100], penals=[0.01],
        verbose=False, export_images=False, root="./test_output"
    )
