# --------------------------------------------
# Author: Raul
# Date: 2025-08-31
# Script: get_diff.R
# Note: This script creates the function that outputs
#       the estimated difference in means
#       given a fit model
# --------------------------------------------



# -----------------------------
# Estimate difference in means
# -----------------------------
get_diff <- function(ghat_B, delta_B, ghat_A, delta_A, pi_hat, Y) {
  muhat_B <- ghat_B + (delta_B*(Y - ghat_B)/(pi_hat))/(mean(delta_B/pi_hat))
  muhat_A <- ghat_A + (delta_A*(Y - ghat_A)/(1-pi_hat))/(mean(delta_A/(1-pi_hat)))
  diff_means <- mean(muhat_B - muhat_A)
  diff_var <- var(muhat_B - muhat_A)
  o <-list(diff_means = diff_means, 
           muhat_B = muhat_B, 
           muhat_A = muhat_A,
           diff_var = diff_var,
           pval = 2*(1-pnorm(abs(diff_means)/sqrt(diff_var))))
}


#get_diff <- function(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y) {
#  muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pi_hat))/(mean(delta_1/pi_hat)))
#  muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pi_hat))/(mean(delta_0/(1-pi_hat))))
#  diff_means <- muhat_1 - muhat_0
#  o <-list(diff_means = diff_means, muhat_1 = muhat_1, muhat_0 = muhat_0)
#}