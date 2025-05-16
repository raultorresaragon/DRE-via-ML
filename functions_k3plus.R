# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k3plus.R
# Date: 2025-05-14
# Note: This script creates functions needed for 
#       simulating DRE for k=3+ where the differences
#       in means are compared pairwise: combn(k,2)
#       
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tictoc)
source("A_nn_tunning.R")
source("Y_nn_tunning.R")

# --------------------------
# Generate X (design matrix)
# --------------------------
gen_X <- function(p=8, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}


# -----------
# Generate A 
# -----------
gen_A <- function(X, beta_A, flavor_A, k) {
  
  xb <-(as.matrix(cbind(1,X))%*%beta_A) 
  if(flavor_A == "logit")   {probs <- 1/(1 + exp(-1*(xb)))}
  if(flavor_A == "pnorm")   {probs <- pnorm(xb)}
  if(flavor_A == "gomertz") {probs <- exp(-exp(xb))}
  if(flavor_A == "arctan")  {probs <- (atan(xb)/pi) + 0.5}
  if(flavor_A == "tanh")    {probs <- 0.5* (tanh(xb)+1)}
  
  A <- rbinom(n, k-1, probs) 
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
  
  Y <- fun_Y(xb_gamma_a) + abs(rnorm(n, 0, 0.01))
}


# ----------------------------------
# Estimate A: propensity score model
# ----------------------------------
estimate_A_nn <- function(X, dat, k, hidunits, eps, penals, verbose=FALSE) {
  
  H_nns <- list()
  pscores <- data.frame(prob = rep(0,n))
  for(i in 1:k-1) {

    dat_i <- dat |> mutate(A = if_else(A == i, 1, 0)) |> dplyr::select(-Y)
    H_nn <- A_model_nn(a_func = "A~.",
                       dat = dat_i,
                       hidunits=hidunits, eps=eps, penals=penals, 
                       verbose=verbose)
    pscores_nn_i <- 
      predict(H_nn, new_data = dat_i |> select(-A), type = "raw") |> 
      as.vector()
    
    pscores <- pscores |> mutate(prob = pscores_nn_i)  
    colnames(pscores)[stringr::str_detect(colnames(pscores),"prob")]<-paste0("pscores_",i)
    rowsums <- rowSums(pscores)
    scale <- function(x) x/rowsums
    pscores <- pscores |> mutate_all(scale)
    
    #H_nns[i] <- H_nn
  }
  list(pscores = pscores, H_nns = H_nns)
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


# --------------------------
# Estimate Y: outcome  model
# --------------------------
estimate_Y_nn <- function(dat, pscores_df, verbose=FALSE,
                          hidunits=c(5,25), eps=c(50,200), penals=c(0.001,0.01)) {
  
  Y <- dat$Y
  
  # compute muhat_0 vs muhat_1, muhat_0 vs muhat_2, muhat_1 vs muhat_2, ... combn(k,2)
  m <- combn(k, 2)-1
  get_d_ij <- function(x) {
    i <- x[[2]] # this is g1 on iteration 1
    j <- x[[1]] # this is g0 on iteration 1
    pi_hat_i <- pscores_df[,i] |> as.vector()
    A_i <- dat |> mutate(A_i = case_when(A==i~1, A==j~0, TRUE ~99)) |> pull("A_i")
    delta_i <- as.numeric(A_i==1) # The 99s are retained so length = n
    delta_j <- as.numeric(A_i==0) # The 99s are retained so length = n

    g_i <- Y_model_nn(dat=dat[A_i==1,] |> dplyr::select(-A), y_func = "Y~.", 
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_i <- predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
    
    g_j <- Y_model_nn(dat=dat[A_i==0, ] |> dplyr::select(-A), y_func = "Y~.",
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_j <- predict(g_j, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  NN est diff means [a=",j, " vs. a=",i,"]=", 
               round(d_ij$diff_means, 3)))
    
    names(d_ij) <- c(paste0("diff_means_",j,i), paste0("muhat_",i), paste0("muhat_",j))
    
    list(d_ij, g_i, g_j)
  }
  o <- apply(m, 2, get_d_ij)
  names(o) <- apply(m, 2, function(x) { 
    i <- x[[2]] 
    j <- x[[1]]
    paste0("A_",j,i)
  })
  o
}

# ----------
# Compute Vn
# ----------
get_Vn <- function(fit_Y_nn, X_new) {
  V_n <- tibble(V_ = rep(NA, nrow(X_new)))
  for(A_type in names(fit_Y_nn)) {
    for(j in c(2,3)) {
      V_n <- 
        V_n |>
        mutate(V_ = predict(fit_Y_nn[[A_type]][[j]], new_data = X_new, type = "raw") |>
                 as.vector())
      
      V_type <- stringr::str_replace(A_type, "A", "V")
      s <- ifelse(j==2, j+2, j)
      r <- stringr::str_sub(A_type, s, s)
      colnames(V_n)[colnames(V_n) == "V_"] <- paste0(V_type, "_g", r)
    }
  }
  V_n |> mutate(OTR = stringr::str_sub(colnames(V_n)[max.col(V_n)], 6, 7))
}


