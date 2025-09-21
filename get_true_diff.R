# --------------------------------------------
# Author: Raul
# Date: 2025-08-31
# Script: get_trie_diff.R
# Note: This script outputs the true diff
#       in means between two groups
#       given the true data generating process
# --------------------------------------------

get_true_diff <- function(a_lvl, xb_Y, gamma, Y_flavor) {
  gamma_allvals <- c(0, gamma)
  i <- a_lvl[[1]];
  j <- a_lvl[[2]];
  gamma_i <- gamma_allvals[i+1]
  gamma_j <- gamma_allvals[j+1]
  
  if(Y_flavor == "expo"){
    EY_j <- mean(exp(xb_Y + gamma_j))
    EY_i <- mean(exp(xb_Y + gamma_i))
    d <- EY_j - EY_i
  }
  if(Y_flavor == "sigmoid"){
    logistic = function(x) {1/(1+exp(-x))}
    EY_j <- mean(10*logistic(xb_Y + gamma_j))
    EY_i <- mean(10*logistic(xb_Y + gamma_i))
    d <- EY_j - EY_i
  }
  if(Y_flavor == "lognormal"){
    EY_j <- mean(plnorm(xb_Y + gamma_j, 0, 1) * 100)
    EY_i <- mean(plnorm(xb_Y + gamma_i, 0, 1) * 100)
    #EY_j <- mean(exp(xb_Y + gamma_j + 0.5 * 1))
    #EY_i <- mean(exp(xb_Y + gamma_i + 0.5 * 1))
    d <- EY_j - EY_i
  }
  if(Y_flavor == "gamma"){
    EY_j <- mean(pgamma(xb_Y + gamma_j,  shape = 3, rate = 2) * 100)
    EY_i <- mean(pgamma(xb_Y + gamma_i,  shape = 3, rate = 2) * 100)   
    #EY_j <- mean(abs(xb_Y + gamma_j))
    #EY_i <- mean(abs(xb_Y + gamma_i))
    d <- EY_j - EY_i
  }
  round_d <- round(d, 3)
  cat(paste0("\n  True diff means_{", j,",",i, "} = ", round_d))
  d
}
