# --------------------------------------------
# Author: Raul
# Date: 2025-08-31
# Script: get_diff.R
# Note: This script creates the function that outputs
#       difference in means
#
# --------------------------------------------



# -----------------------------
# Estimate difference in means
# -----------------------------
get_diff <- function(ghat_L, delta_L, ghat_R, delta_R, pi_hat, Y) {
  muhat_L <- mean(ghat_L + (delta_L*(Y - ghat_L)/(pi_hat))/(mean(delta_L/pi_hat)))
  muhat_R <- mean(ghat_R + (delta_R*(Y - ghat_R)/(1-pi_hat))/(mean(delta_R/(1-pi_hat))))
  diff_means <- muhat_L - muhat_R
  o <-list(diff_means = diff_means, muhat_L = muhat_L, muhat_R = muhat_R)
}


#get_diff <- function(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y) {
#  muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pi_hat))/(mean(delta_1/pi_hat)))
#  muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pi_hat))/(mean(delta_0/(1-pi_hat))))
#  diff_means <- muhat_1 - muhat_0
#  o <-list(diff_means = diff_means, muhat_1 = muhat_1, muhat_0 = muhat_0)
#}