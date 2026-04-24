# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get_true_OTR_two_stage_appendix.py
#
# Appendix: Monte Carlo computation of the true optimal two-stage treatment regime
#
# Setting : binary A1 and A2, two stages, p1=3 stage-1 covariates
#           (X1_1, X1_2, X1_3 where X1_3 is the binary effect modifier)
# Dataset : s2_k2_logit_expo_0  (i=0, k=2, expo flavor)
# Method  : backward induction approximated by Monte Carlo simulation
#
# Algorithm (per individual i):
#   For each candidate a1 in {0, 1}:
#     Simulate N_SIM trajectories of (Y_1, X2, Y) given (X1_i, A1=a1)
#     For each candidate a2 in {0, 1}:
#       Average simulated Y  →  E_hat[Y | X1_i, A1=a1, A2=a2]
#     best_a2(i, a1) = argmax_{a2} E_hat[Y | X1_i, A1=a1, A2=a2]
#     V(i, a1)       = max_{a2}    E_hat[Y | X1_i, A1=a1, A2=a2]
#   d_star_1(i) = argmax_{a1} V(i, a1)
#   d_star_2(i) = best_a2(i, d_star_1(i))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
import os
from sim_params import make_sim_params

script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')


# ============================================================
# STEP 1: Load dataset and subset to n=100
# ============================================================
filename = 's2_k2_logit_expo_0'

dat = pd.read_csv(os.path.join(datasets_dir, f'{filename}.csv'))
dat = dat.iloc[:100].reset_index(drop=True)
n   = len(dat)

print(f"Loaded: {filename}  (n={n})")

# Extract observed columns
X1      = dat[['X1_1', 'X1_2', 'X1_3']].values   # (n, 3)
X1_bin  = dat['X1_3'].values                       # binary effect modifier (last col of X1)
Y1_obs  = dat['Y_1'].values                        # observed intermediate outcome


# ============================================================
# STEP 2: Recover true DGP parameters from seed stored in _info.csv
# ============================================================
info = pd.read_csv(os.path.join(datasets_dir, '_info.csv'))
row  = info[info['filename'] == filename].iloc[0]

p1, p2 = int(row['p1']), int(row['p2'])   # p1=3, p2=4 (=p1+1)
k1, k2 = int(row['k1']), int(row['k2'])   # k1=2, k2=2
seed   = int(row['seed'])                  # seed = 1810 + i = 1810

params     = make_sim_params(p1=p1, p2=p2, k1=k1, k2=k2, seed=seed)
beta_Y1    = params['beta_Y1']    # (p1+1,) = (4,)  effects of [intercept, X1] on Y_1
delta1     = params['delta1']     # (1,)             A1 main effect on Y_1
Delta1     = params['Delta1']     # (1,)             A1 × X1_bin interaction on Y_1
beta_Y2    = params['beta_Y2']    # (9,)    = (1+p1+1+p2,)  effects of [1,X1,A1,X2] on Y
delta2     = params['delta2']     # (1,)             A2 main effect on Y
Delta2     = params['Delta2']     # (1,)             A2 × Y1_bin interaction on Y

# Print true parameter values for reference (useful for the appendix)
print(f"\nTrue parameters  (seed={seed}):")
print(f"  beta_Y1    = {np.round(beta_Y1, 4)}")
print(f"  delta1     = {np.round(delta1,  4)}   <- A1 main effect on Y_1")
print(f"  Delta1     = {np.round(Delta1,  4)}   <- A1 x X1_bin interaction on Y_1")
print(f"  beta_Y2    = {np.round(beta_Y2, 4)}")
print(f"  delta2     = {np.round(delta2,  4)}   <- A2 main effect on Y")
print(f"  Delta2     = {np.round(Delta2,  4)}   <- A2 x Y1_bin interaction on Y")


# ============================================================
# STEP 3: Monte Carlo settings
# ============================================================
N_SIM = 300    # simulated trajectories per individual per (a1, a2) path
RHO   = 0.5   # AR(1) correlation for remaining X2 covariates (matches gen_X2)

# Covariance matrix for the remaining X2 covariates (X2_1, X2_2, X2_3)
# AR(1) structure with rho=0.5, matching gen_X2
n_X2_remaining = p2 - 1   # = 3
Sigma_X2 = np.array([[RHO**abs(i - j) for j in range(n_X2_remaining)]
                      for i in range(n_X2_remaining)])


# ============================================================
# STEP 4: Fix Y_1 binarization threshold from observed data
#
# In gen_Y_two_stage, Y1_bin = I(Y_1 > median(Y_1)) is used in the
# Delta2 interaction term.  We fix the threshold at the observed median
# so that the MC simulations are consistent with the DGP.
# ============================================================
Y1_threshold = np.median(Y1_obs)
print(f"\nY_1 binarization threshold (observed median): {Y1_threshold:.4f}")


# ============================================================
# STEP 5: Monte Carlo loop — compute true OTR for each individual
# ============================================================
d_star_1 = np.zeros(n, dtype=int)
d_star_2 = np.zeros(n, dtype=int)

print(f"\nRunning Monte Carlo OTR  (N_SIM={N_SIM},  n={n} individuals)...\n")

for i in range(n):

    if (i + 1) % 10 == 0:
        print(f"  individual {i+1}/{n}")

    # Fixed covariates for individual i
    x1_i              = X1[i]                              # (3,)
    x1_bin_i          = X1_bin[i]                         # scalar, binary
    x1_with_int_i     = np.concatenate([[1.0], x1_i])     # (4,) = [1, X1_1, X1_2, X1_3]

    V_a1             = np.zeros(2)       # V(i, a1): value under optimal a2, for each a1
    best_a2_given_a1 = np.zeros(2, dtype=int)

    for a1 in [0, 1]:

        # ----------------------------------------------------------
        # Step 5a: Simulate N_SIM intermediate outcomes Y_1
        #
        # DGP (expo flavor, from gen_X2):
        #   eta_Y1 = [1, X1_i] @ beta_Y1
        #            + delta1[0]          * I(a1 == 1)
        #            + Delta1[0] * x1_bin * I(a1 == 1)
        #   Y_1 = exp(eta_Y1) + Normal(0, 0.5),  clipped at 0.01
        # ----------------------------------------------------------
        eta_Y1 = (x1_with_int_i @ beta_Y1
                  + (a1 > 0) * delta1[0]
                  + (a1 > 0) * x1_bin_i * Delta1[0])

        Y1_sim = np.exp(eta_Y1) + np.random.normal(0, 0.5, N_SIM)
        Y1_sim = np.maximum(Y1_sim, 0.01)

        # ----------------------------------------------------------
        # Step 5b: Simulate N_SIM sets of remaining X2 covariates
        #          (X2_1, X2_2, X2_3 — the columns beyond Y_1)
        #
        # DGP (from gen_X2):
        #   [X2_1, X2_2, X2_3] ~ MVN(mean = eta_Y1 * 1_3, cov = Sigma_X2)
        #
        # Note: gen_X2 binarizes X2_3 using the population median. We omit
        # that step here because in the per-individual MC the notion of a
        # population median does not apply cleanly.
        # ----------------------------------------------------------
        X2_remaining_sim = np.random.multivariate_normal(
            mean=np.full(n_X2_remaining, eta_Y1),
            cov=Sigma_X2,
            size=N_SIM
        )   # (N_SIM, 3)

        # Assemble full X2: [Y_1 | X2_1, X2_2, X2_3]  shape (N_SIM, 4)
        X2_sim = np.column_stack([Y1_sim, X2_remaining_sim])

        # Binarize simulated Y_1 for the Delta2 interaction term
        Y1_bin_sim = (Y1_sim > Y1_threshold).astype(float)   # (N_SIM,)

        # Build combined feature matrix [1, X1_i, a1, X2_sim] for all N_SIM draws
        # Shape: (N_SIM, 1 + p1 + 1 + p2) = (N_SIM, 9)
        X_combined_sim = np.column_stack([
            np.ones(N_SIM),              # intercept
            np.tile(x1_i, (N_SIM, 1)),   # X1_1, X1_2, X1_3  (repeated N_SIM times)
            np.full(N_SIM, a1),          # A1 = a1 for all
            X2_sim                       # Y_1, X2_1, X2_2, X2_3
        ])

        # Base linear predictor (shared across both a2 values)
        eta_Y_base = X_combined_sim @ beta_Y2   # (N_SIM,)

        # ----------------------------------------------------------
        # Step 5c: For each candidate a2, simulate N_SIM final outcomes
        #          Y and average  →  E_hat[Y | X1_i, A1=a1, A2=a2]
        #
        # DGP (expo flavor, from gen_Y_two_stage):
        #   eta_Y = eta_Y_base
        #           + delta2[0]               * I(a2 == 1)
        #           + Delta2[0] * Y1_bin_sim  * I(a2 == 1)
        #   Y = exp(eta_Y) + Normal(0, 0.5),  clipped at 0
        # ----------------------------------------------------------
        E_Y_given_a2 = np.zeros(2)

        for a2 in [0, 1]:

            eta_Y = eta_Y_base.copy()
            if a2 > 0:
                eta_Y += delta2[0]
                eta_Y += Delta2[0] * Y1_bin_sim

            Y_sim          = np.exp(eta_Y) + np.random.normal(0, 0.5, N_SIM)
            Y_sim          = np.abs(Y_sim)           # ensure positive
            E_Y_given_a2[a2] = np.mean(Y_sim)

        # ----------------------------------------------------------
        # Step 5d: Best A2 and value V(i, a1)
        # ----------------------------------------------------------
        best_a2_given_a1[a1] = int(np.argmax(E_Y_given_a2))
        V_a1[a1]             = np.max(E_Y_given_a2)

    # ----------------------------------------------------------
    # Step 5e: Best A1 = argmax_{a1} V(i, a1)
    # Step 5f: Record optimal decisions for individual i
    # ----------------------------------------------------------
    d_star_1[i] = int(np.argmax(V_a1))
    d_star_2[i] = best_a2_given_a1[d_star_1[i]]


print("\nDone.")
print(f"  d_star_1 distribution: {np.bincount(d_star_1)}")
print(f"  d_star_2 distribution: {np.bincount(d_star_2)}")


# ============================================================
# STEP 6: Save output in the same format as get_true_otr_two_stage.py
# ============================================================
out = pd.DataFrame({
    'd1_star': d_star_1,
    'd2_star': d_star_2,
})

out_filename = f'{filename}_trueOTR_appendix'
out_path     = os.path.join(datasets_dir, f'{out_filename}.csv')
out.to_csv(out_path, index=False)
print(f"\n✓ Saved: {out_filename}.csv")
