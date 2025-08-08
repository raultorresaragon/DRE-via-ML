# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: YAX_functions.R
# Date: 2025-05-14
# Note: This script creates functions needed for 
#       simulating DRE for k=3+ where the differences
#       in means are compared pairwise: combn(k,2)
#       
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(dplyr)

# --------------------------
# Generate X (design matrix)
# --------------------------
gen_X <- function(p, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}


# -----------
# Generate A 
# -----------
gen_A <- function(X, beta_A, flavor_A) {
  
  xb <- (as.matrix(cbind(1,X))%*%beta_A) 
  if(flavor_A == "logit")   {
    exp_xb <- exp(xb)
    denom <- 1 + rowSums(exp_xb)
    probs <- 1/denom
    for(i in 1:dim(beta_A)[2]) {
      probs <- cbind(probs, exp_xb[,i]/denom)
    }
  }
  if(flavor_A == "tanh")    {
    # When k>2,this should output p(A=0),p(A=1),p(A=2)...,p(A=k-1) for rmultinom()
    raw_scores <- 0.5 * (tanh(xb)+1)
    if(dim(raw_scores)[2]>1) sum_raws <- rowSums(raw_scores)
    if(dim(raw_scores)[2]<2) sum_raws <- 1
    probs <- raw_scores / sum_raws
  }
  
  if(ncol(as.matrix(beta_A)) > 1) {
    A_mat <- t(apply(probs, 1, function(pr) rmultinom(1, size=1, prob=pr)))
    A <- max.col(A_mat, ties.method = "first") - 1 #<-get column index where the max is
  } else {
    A <- rbinom(n, 1, probs) 
  }
  A
}


# ------------
# Generate Y 
# ------------
#gamma <- dplyr::if_else(X[,1]>0, 0.7, 0.1) #<-with trt heterogeneity
gen_Y <- function(gamma, X, A, beta_Y, flavor_Y) {
  
  as <- sort(unique(A)) #<--lowest a value will be baseline
  A_mat <- sapply(as[-1], function(a) as.integer(A==a))
  colnames(A_mat) <-paste0("A_", as[-1])
  xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + A_mat %*% gamma
  
  if(flavor_Y == "expo") { fun_Y = exp}
  if(flavor_Y == "square") { fun_Y = function(x) x^2}
  if(flavor_Y == "sigmoid"){ fun_Y = function(x) 1/(1+exp(-x)) * 10}
  if(flavor_Y == "nonlin") { fun_Y = function(x) (x^4) / (1 + abs(x)^3)}
  if(flavor_Y == "piece") { fun_Y = function(x) ifelse(x<0, log1p(x^2), 1/(1+exp(-x)))*6}
  if(flavor_Y == "atan") { fun_Y = function(x) 6 * (atan(x) / pi + 0.5)}
  if(flavor_Y == "expf") { fun_Y = function(x) rexp(length(x), rate = 1/x)}
  
  Y <- fun_Y(xb_gamma_a) + rnorm(n, 0, 0.1)
  Y[Y<=0] <- abs(Y[Y<=0])
  threshold <- qexp(0.9995, rate = 1/mean(Y))  # 99.95th percentile cutoff
  #Y[Y < threshold] <- threshold
  Y
}


# -----------------------------
# Estimate difference in means
# -----------------------------
get_diff <- function(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y) {
  muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pi_hat))/(mean(delta_1/pi_hat)))
  muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pi_hat))/(mean(delta_0/(1-pi_hat))))
  diff_means <- muhat_1 - muhat_0
  o <-list(diff_means = diff_means, muhat_1 = muhat_1, muhat_0 = muhat_0)
}