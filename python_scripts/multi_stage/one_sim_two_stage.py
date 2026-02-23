# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: one_sim_two_stage.py
# Date: 2026-02-19
# Note: This script runs one simulation iteration for two-stage DTR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from YAX_funs import gen_X, gen_A, gen_A2, gen_X2, gen_Y_two_stage
from pscores_models import estimate_A_nn, estimate_A_logit
from outcome_models import estimate_optimal_regime_two_stage
from get_true_optimal_regime import compute_true_optimal_regime, evaluate_regime_accuracy

def one_sim_two_stage(n, p1, p2, k1, k2,
                      beta_A1, beta_A2, gamma_stay,
                      gamma1_X2, beta_X2,
                      gamma1_Y, gamma2_Y, beta_Y,
                      A_flavor, Y_flavor,
                      hidunits=[5, 20], eps=[100, 250], penals=[0.001, 0.01],
                      verbose=False, iter=1, export_images=False, root="./", rho=0.5,
                      compute_true_regime=False, n_samples_true=500):
    """
    Run one simulation iteration for two-stage DTR

    Parameters:
    n = sample size
    p1 = number of stage 1 covariates
    p2 = number of stage 2 covariates
    k1 = number of stage 1 treatment levels
    k2 = number of stage 2 treatment levels
    beta_A1 = stage 1 treatment model coefficients
    beta_A2 = stage 2 treatment model coefficients
    gamma_stay = stay-probability parameter (higher -> more likely to stay on A1 when X2 is high)
    gamma1_X2 = effect of A1 on X2
    beta_X2 = effect of X1 on X2
    gamma1_Y = effect of A1 on Y
    gamma2_Y = effect of A2 on Y
    beta_Y = outcome model coefficients
    A_flavor = treatment model type ("logit" or "tanh")
    Y_flavor = outcome model type ("expo", "sigmoid", "gamma", "lognormal")
    hidunits = hidden units for NN
    eps = epochs for NN
    penals = regularization parameters
    verbose = print details
    iter = iteration number
    export_images = save plots
    root = root directory
    rho = correlation parameter
    
    Returns:
    - Dictionary with results
    """
    
    print(f"\n{'='*70}")
    print(f"SIMULATION {iter}: Two-Stage DTR")
    print(f"{'='*70}")
    
    # ========================================
    # STAGE 1: Generate baseline data
    # ========================================
    print("\nStage 1: Generating baseline data...")
    X1 = gen_X(n=n, p=p1, rho=rho, p_bin=1)
    A1 = gen_A(X=X1, beta_A=beta_A1, flavor_A=A_flavor, k=k1)
    
    print(f"  A1 distribution: {np.bincount(A1)}")
    for i in range(k1):
        print(f"    P(A1={i}) = {np.mean(A1 == i):.2f}")
    
    # ========================================
    # STAGE 2: Generate intermediate data
    # ========================================
    print("\nStage 2: Generating intermediate data...")
    X2 = gen_X2(X1=X1, A1=A1, p2=p2, gamma1_X2=gamma1_X2, beta_X2=beta_X2, 
                flavor_X2=Y_flavor, rho=rho, p_bin=1)

    # Stage 2 treatment depends on full history + stay-probability
    # If X2 is high (patient responding), increase P(A2 = A1)
    A2 = gen_A2(X1=X1, A1=A1, X2=X2, beta_A2=beta_A2, gamma_stay=gamma_stay,
                flavor_A=A_flavor, k2=k2)
    
    print(f"  A2 distribution: {np.bincount(A2)}")
    for i in range(k2):
        print(f"    P(A2={i}) = {np.mean(A2 == i):.2f}")
    
    # ========================================
    # Generate final outcome
    # ========================================
    print("\nGenerating final outcome...")
    Y_result = gen_Y_two_stage(
        gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y,
        X1=X1, A1=A1, X2=X2, A2=A2,
        beta_Y=beta_Y, flavor_Y=Y_flavor
    )
    Y = Y_result['Y']
    
    print(f"  Y: min={Y.min():.2f}, mean={Y.mean():.2f}, max={Y.max():.2f}")
    
    # Create dataset
    dat = pd.concat([
        X1,
        pd.Series(A1, name='A1'),
        X2,
        pd.Series(A2, name='A2'),
        pd.Series(Y, name='Y')
    ], axis=1)
    
    # Save dataset
    os.makedirs(f"{root}/datasets", exist_ok=True)
    dat.to_csv(f"{root}/datasets/df_two_stage_k{k1}k{k2}_{A_flavor}_{Y_flavor}_dset{iter}.csv", index=False)
    
    # ========================================
    # Estimate propensity scores
    # ========================================
    print("\nEstimating propensity scores...")
    start_time = time.time()
    
    # Stage 1 propensity scores
    dat_stage1 = pd.concat([pd.Series(A1, name='A'), X1], axis=1)
    fit_A1_nn = estimate_A_nn(X=None, dat=dat_stage1, k=k1,
                              hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    fit_A1_logit = estimate_A_logit(X=None, dat=dat_stage1, k=k1, verbose=verbose)
    
    # Stage 2 propensity scores
    dat_stage2 = pd.concat([pd.Series(A2, name='A'), X_history], axis=1)
    fit_A2_nn = estimate_A_nn(X=None, dat=dat_stage2, k=k2,
                              hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    fit_A2_logit = estimate_A_logit(X=None, dat=dat_stage2, k=k2, verbose=verbose)
    
    print(f"  Propensity score estimation time: {time.time() - start_time:.2f}s")
    
    # ========================================
    # Q-learning: Estimate optimal regimes
    # ========================================
    print("\nEstimating optimal regimes via Q-learning...")
    start_time = time.time()
    
    # NN with NN propensity scores
    regime_nn = estimate_optimal_regime_two_stage(
        dat=dat,
        pscores_A1=fit_A1_nn['pscores'],
        pscores_A2=fit_A2_nn['pscores'],
        k1=k1, k2=k2,
        model_type='nn',
        hidunits=hidunits, eps=eps, penals=penals, verbose=verbose
    )
    
    # OLS with logit propensity scores
    regime_ols = estimate_optimal_regime_two_stage(
        dat=dat,
        pscores_A1=fit_A1_logit['pscores'],
        pscores_A2=fit_A2_logit['pscores'],
        k1=k1, k2=k2,
        model_type='ols'
    )
    
    # Expo with logit propensity scores
    regime_expo = estimate_optimal_regime_two_stage(
        dat=dat,
        pscores_A1=fit_A1_logit['pscores'],
        pscores_A2=fit_A2_logit['pscores'],
        k1=k1, k2=k2,
        model_type='expo'
    )
    
    # Lognormal if Y_flavor is lognormal
    regime_lognormal = None
    if Y_flavor == "lognormal":
        regime_lognormal = estimate_optimal_regime_two_stage(
            dat=dat,
            pscores_A1=fit_A1_logit['pscores'],
            pscores_A2=fit_A2_logit['pscores'],
            k1=k1, k2=k2,
            model_type='lognormal'
        )
    
    print(f"  Q-learning time: {time.time() - start_time:.2f}s")
    
    # ============================
    # Compute true optimal regime 
    # ============================
    true_regime_result = None
    accuracy_results = None
    
    if compute_true_regime:
        print("\n" + "="*70)
        print("COMPUTING TRUE OPTIMAL REGIME")
        print("="*70)
        
        true_regime_result = compute_true_optimal_regime(
            X1=X1, X2=X2, A1=A1, k1=k1, k2=k2,
            gamma1_X2=gamma1_X2, beta_X2=beta_X2,
            gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y, beta_Y=beta_Y,
            p2=p2, rho=rho, flavor_Y=Y_flavor,
            n_samples=n_samples_true
        )
        
        # Evaluate accuracy
        accuracy_results = {
            'nn': evaluate_regime_accuracy(
                regime_nn['optimal_A1'], regime_nn['optimal_A2'],
                true_regime_result['true_optimal_A1'], true_regime_result['true_optimal_A2']
            ),
            'ols': evaluate_regime_accuracy(
                regime_ols['optimal_A1'], regime_ols['optimal_A2'],
                true_regime_result['true_optimal_A1'], true_regime_result['true_optimal_A2']
            ),
            'expo': evaluate_regime_accuracy(
                regime_expo['optimal_A1'], regime_expo['optimal_A2'],
                true_regime_result['true_optimal_A1'], true_regime_result['true_optimal_A2']
            )
        }
        
        if regime_lognormal:
            accuracy_results['lognormal'] = evaluate_regime_accuracy(
                regime_lognormal['optimal_A1'], regime_lognormal['optimal_A2'],
                true_regime_result['true_optimal_A1'], true_regime_result['true_optimal_A2']
            )
    
    # ========================================
    # Summarize results
    # ========================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nOptimal regime distributions:")
    print(f"  NN:  A1={np.bincount(regime_nn['optimal_A1'])}, A2={np.bincount(regime_nn['optimal_A2'])}")
    print(f"  OLS: A1={np.bincount(regime_ols['optimal_A1'])}, A2={np.bincount(regime_ols['optimal_A2'])}")
    print(f"  Expo: A1={np.bincount(regime_expo['optimal_A1'])}, A2={np.bincount(regime_expo['optimal_A2'])}")
    if regime_lognormal:
        print(f"  Lognormal: A1={np.bincount(regime_lognormal['optimal_A1'])}, A2={np.bincount(regime_lognormal['optimal_A2'])}")
    
    print("\nExpected values under optimal regimes:")
    print(f"  NN:   {np.mean(regime_nn['Y_tilde']):.2f}")
    print(f"  OLS:  {np.mean(regime_ols['Y_tilde']):.2f}")
    print(f"  Expo: {np.mean(regime_expo['Y_tilde']):.2f}")
    if regime_lognormal:
        print(f"  Lognormal: {np.mean(regime_lognormal['Y_tilde']):.2f}")
    print(f"  Observed: {Y.mean():.2f}")
    
    if accuracy_results:
        print("\nRegime accuracy vs. true optimal:")
        for model_name, acc in accuracy_results.items():
            print(f"  {model_name.upper()}:")
            print(f"    A1: {acc['A1_accuracy']:.1%}, A2: {acc['A2_accuracy']:.1%}, Joint: {acc['joint_accuracy']:.1%}")
    
    # ========================================
    # Return results
    # ========================================
    return {
        'dataset': dat,
        'regime_nn': regime_nn,
        'regime_ols': regime_ols,
        'regime_expo': regime_expo,
        'regime_lognormal': regime_lognormal,
        'pscores_A1_nn': fit_A1_nn['pscores'],
        'pscores_A1_logit': fit_A1_logit['pscores'],
        'pscores_A2_nn': fit_A2_nn['pscores'],
        'pscores_A2_logit': fit_A2_logit['pscores'],
        'true_regime': true_regime_result,
        'accuracy': accuracy_results,
        'iter': iter
    }


# ========================================
# Example usage
# ========================================
if __name__ == "__main__":
    np.random.seed(123)
    
    # Parameters
    n = 200
    p1 = 3
    p2 = 2
    k1 = 3
    k2 = 3
    
    # Coefficients
    beta_A1 = np.array([[0.5, 0.3], [-0.3, 0.4], [0.2, -0.1], [0.1, 0.2]])
    beta_A2 = np.array([[0.3, 0.2], [-0.2, 0.3], [0.1, -0.1], [0.2, 0.1], [0.4, -0.2], [-0.3, 0.5], [0.1, -0.2]])
    gamma_stay = 0.5  # stay-probability: higher X2 -> more likely to stay on A1

    gamma1_X2 = np.array([0.5, 1.0])
    beta_X2 = np.array([0.0, 0.3, 0.2, 0.1])
    
    gamma1_Y = np.array([1.0, 2.0])
    gamma2_Y = np.array([1.5, 3.0])
    beta_Y = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.3])
    
    # Run simulation
    result = one_sim_two_stage(
        n=n, p1=p1, p2=p2, k1=k1, k2=k2,
        beta_A1=beta_A1, beta_A2=beta_A2, gamma_stay=gamma_stay,
        gamma1_X2=gamma1_X2, beta_X2=beta_X2,
        gamma1_Y=gamma1_Y, gamma2_Y=gamma2_Y, beta_Y=beta_Y,
        A_flavor="logit", Y_flavor="expo",
        hidunits=[5, 10], eps=[100], penals=[0.01],
        verbose=False, iter=1, export_images=False, root="./test_output"
    )
    
    print("\n✓ Simulation complete!")
