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
gen_X <- function(p, rho=0.6, mu, n, p_bin=1) {
  
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- 
    MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
  
  # Binary covariates
  X_bin <- sapply((p-p_bin+1):p, function(j) {
    ifelse(X[, j] < mean(X[, j]), 0, 1)
  })
  X <- cbind(X[, 1:(p-p_bin)], X_bin)
  colnames(X) <- Xnames
  X
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
    #xb <- cbind(xb, 0) # adding the baseline class (a=0)
    raw_scores <- 
      as.data.frame(0.5 * (tanh(xb)+1)) |> 
      mutate(dummyzero1=0, dummyzero2=0) # add zero so rowSums works with k<=3
    
    probs <- data.frame(class1 = rep(NA, nrow(X)))
    for(i in 1:ncol(xb)) {
      probs[[paste0("class",i)]] <- raw_scores[,i] / (1 + rowSums(raw_scores[,-i]))
    }
    if(k==2) {
      sum_all_other_classes <- 1 - probs$class1
    } else {
      sum_all_other_classes <- 1 - rowSums(cbind(probs[,1:ncol(probs)], rep(0,nrow(probs))))
    }
    probs[[paste0("class",k)]] <- sum_all_other_classes |> as.vector()
  }
  
  if(ncol(as.matrix(beta_A)) > 1) {
    A_mat <- t(apply(probs, 1, function(pr) rmultinom(1, size=1, prob=pr)))
    A <- max.col(A_mat, ties.method = "first") - 1 #<-get column index where the max is
  } else {
    A <- rbinom(n, 1, probs[,1]) 
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
  xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + (A_mat %*% gamma)
  
  if(flavor_Y == "expo") { 
    fun_Y = function(x) exp(x) + rnorm(n, 0, 0.1)
  }
  if(flavor_Y == "sigmoid"){ 
    fun_Y = function(x) 1/(1+exp(-x)) * 10 + rnorm(n, 0, 0.1)
  }
  if(flavor_Y == "gamma") { fun_Y = function(x) {
    mu = abs(x)
    shape <- 2
    scale <- mu / shape
    rgamma(n, shape = shape, scale = scale)
  }}
  if(flavor_Y == "lognormal") { fun_Y = function(x) {
    sigma_true <- 1
    mu_log = x
    logY = mu_log + rnorm(n, 0, sigma_true)
    exp(logY)
  }}

  Y <- fun_Y(xb_gamma_a)
  Y[Y<=0] <- abs(Y[Y<=0])
  threshold <- qexp(0.999, rate = 1/mean(Y))  # 99.5th percentile cutoff
  Y[Y>threshold] <- threshold
  list(Y=Y, fun_Y = fun_Y)
}
