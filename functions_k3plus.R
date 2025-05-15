# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k2.R
# Date: 2025-05-14
# Note: This script creates functions needed for 
#       simulating DRE for k=2 version 01
#       We simulate A with logistic link
#       and Y with exponential link
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
  
  xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + gamma * A
  
  if(flavor_Y == "expo") { fun_Y = exp}
  if(flavor_Y == "square") { fun_Y = function(x) x^2}
  if(flavor_Y == "sigmoid"){ fun_Y = function(x) 1/(1+exp(-x)) * 10}
  if(flavor_Y == "nonlin") { fun_Y = function(x) (x^4) / (1 + abs(x)^3)}
  if(flavor_Y == "piece") { fun_Y = function(x) ifelse(x<0, log1p(x^2), 1/(1+exp(-x)))*6}
  if(flavor_Y == "atan") { fun_Y = function(x) 6 * (atan(x) / pi + 0.5)}
  if(flavor_Y == "expf") { fun_Y = function(x) rexp(length(x), rate = 1/x)}
  
  Y <- fun_Y(xb_gamma_a) + abs(rnorm(n, 0, 0.01))
  
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


# ----------------------------------
# Estimate A: propensity score model
# ----------------------------------
estimate_A_nn <- function(X, dat, k, hidunits, eps, penals) {
  
  H_nns <-list()
  pscores <- data.frame(prob = rep(0,n))
  for(i in 1:k-1) {

    dat_i <- dat |> mutate(A = if_else(A == i, 1, 0)) |> dplyr::select(-Y)
    H_nn <- A_model_nn(a_func = "A~.",
                       dat = dat_i,
                       hidunits=hidunits, eps=eps, penals=penals, 
                       verbose=TRUE)
    pscores_nn_i <- 
      predict(H_nn, new_data = dat_i |> select(-A), type = "raw") |> 
      as.vector()
    
    pscores <- pscores |> mutate(prob = pscores_nn_i)  
    colnames(pscores)[stringr::str_detect(colnames(pscores),"prob")]<-paste0("pscores_",i)
    rowsums <- rowSums(pscores)
    scale <- function(x) x/rowsums
    pscores <- pscores |> mutate_all(scale)
    
    #H_nns[[i]] <- H_nn
  }
  list(pscores = pscores, H_nns = H_nns)
}

# --------------------------
# Estimate Y: outcome  model
# --------------------------

estimate_Y_nn <- function(dat, pscores_df, verbose=FALSE,
                          hidunits=c(5,25), eps=c(50,200), penals=c(0.001,0.01)) {
  
  Y <- dat$Y
  
  # computes muhat_i vs muhat_noti
  ### for(i in 1:k-1) {
  ###   
  ###   pi_hat_i <- pscores_df[,i+1] |> as.vector()
  ###   A_i <- if_else(dat$A==i, 1, 0)
  ###   delta_i <- as.numeric(A_i==1)
  ###   delta_0 <- as.numeric(A_i==0)
  ###   
  ###   g_i <- Y_model_nn(dat=dat[A_i==1,] |> dplyr::select(-A), y_func = "Y~.", 
  ###                     hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
  ###   ghat_i <- predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
  ###   
  ###   g_0 <- Y_model_nn(dat=dat[A_i==0, ] |> dplyr::select(-A), y_func = "Y~.",
  ###                     hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
  ###   ghat_0 <- predict(g_0, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
  ###   
  ###   d <- get_diff(ghat_i, delta_i, ghat_0, delta_0, pi_hat_i, Y)
  ###   
  ###   muhat_i <- d$muhat_1
  ###   
  ###   cat(paste0("\n  NN est diff means [k=",i, " vs. k=~",i,"]=", round(d$diff_means, 3)))
  ### 
  ###   o_i<- list(muhat_i = muhat_i, g_i = g_i, diff_i = d$diff_means)
  ###   
  ###   names(o_i) <- c(paste0("muhat_", i), paste0("g_", i), paste0("diff_", i))
  ###   
  ###   o[[i+1]] <- o_i
  ### }
  ### o
  
  # compute muhat_0 vs muhat_1, muhat_0 vs muhat_2, muhat_1 vs muhat_2, ... combn(k,2)
  m <- combn(k, 2)-1
  get_d_ij <- function(x) {
    i <- x[[2]] # this is g1 on iteration 1
    j <- x[[1]] # this is g0 on iteration 1
    pi_hat_i <- pscores_df[,i] |> as.vector()
    A_i <- dat |> mutate(A_i = case_when(A==i~1, A==j~0, TRUE ~99)) |> pull("A_i")
    delta_i <- as.numeric(A_i==1)
    delta_j <- as.numeric(A_i==0)

    g_i <- Y_model_nn(dat=dat[A_i==1,] |> dplyr::select(-A), y_func = "Y~.", 
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_i <- predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
    
    g_j <- Y_model_nn(dat=dat[A_i==0, ] |> dplyr::select(-A), y_func = "Y~.",
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_j <- predict(g_j, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  NN est diff means [k=",j, " vs. k=",i,"]=", 
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
get_Vn <- function(g_s, X_new) {
  V_n <- tibble(V_ = NA)
  i = -1
  for(g in g_s) {
    i = i + 1
    V_n <- 
      df |> 
      mutate(V_ = predict(g, new_data = X_new, type = "raw"))
    colnames(V_n)[colnames(V_n) == "V_"] <- paste0("V_", i)
  }
  V_n
}


# ---------------------------
# One iteration function k=3+
# ---------------------------

one_sim <- function(n=n, p=8, Xmu, beta_A, beta_Y, gamma, Y_fun, A_flavor, Y_flavor,
                    nn_hidunits, nn_eps, nn_penals, verbose = FALSE) {

  X <- gen_X(n=n, p=p, rho=rho, mu=Xmu)
  A <- gen_A(X=X, beta=beta_A, flavor_A=A_flavor)
  Y <- gen_Y(X=X, A=A, beta_Y=beta_Y, gamma=gamma, flavor_Y=Y_flavor)
  dat <- cbind(Y,A,X) 
  
  stopifnot(Y>0)
  for(i in 1:k-1) {cat(paste0("\n  P(A=",i,")= ", mean(A==i) |> round(1)))}
  
  get_true_diff <- function(x) {
    i <- x[[2]]
    j <- x[[1]]
    d <- mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y) + gamma*(i))) - 
           mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y) + gamma*j))
    cat(paste0("\n  True diff means ", j, i, " = ", round(d, 3)))
    d
  }
  true_diffs <- apply(as.matrix(combn(k,2)-1), 2, get_true_diff)
  dat <- cbind(Y,A,X) 
  
  # Estimating A (propensity model)
  fit_A_nn <- estimate_A_nn(X=X, dat=dat, k=3, 
                            hidunits=nn_hidunits, 
                            eps=nn_eps, 
                            penals=nn_penals)
  
  # Estimating Y (outcome model)
  fit_Y_nn <- estimate_Y_nn(dat, pscores_df=fit_A_nn$pscores,
                            hidunits=nn_hidunits, 
                            eps=nn_eps, 
                            penals=nn_penals, 
                            verbose=TRUE)
  
  toc <- toc(quiet=TRUE)
  print(paste0("  Y nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))  
  
  # Computing Vn
  X_new <- gen_X(n=25, p=p, rho=rho, mu=Xmu)
  Vn_df <- get_Vn(fit_Y_nn$LIST_OF_g_s, X_new = X_new[1,]) 
  
  # Packing results into k rows
  get_naive_est <- function(x) {
    i <- x[[1]]
    j <- x[[2]]
    d <- mean(Y[A==j]) - mean(Y[A==i])
    cat(paste0("\n  Naive diff means ", i, j, " = ", round(d, 3)))
    d
  }
  naive_est <- apply(as.matrix(combn(3,2) - 1), 2, get_naive_est)
  nn_model_est <- sapply(fit_Y_nn, function(x) {x[[1]]})

  my_k_rows <- 
    t(as.data.frame(nn_model_est)) |> 
    rbind(naive_est, true_diffs) |>
    mutate(dataset = i)
  
  list(my_k_rows = my_k_rows, Vn_df = Vn_df)
  
}

#toc <- toc(quiet=TRUE)






















