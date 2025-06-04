# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: one_sim_k3plus.R
# Date: 2025-05-15
# Note: This script creates a function
#       to run one iteration of k3 plus sims
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------------------------
# PARAMERTERS FOR DEBUGGING

 ## rm(list = ls())
 ## set.seed(1811)
 ## M <- 1
 ## n <- 1000
 ## k <- 5
 ## gamma <- c(0.8, 0.6, 0.52, 0.37)
 ## flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
 ## #flavor_ops <- c("logit","expo", function(x) exp(x))
 ## p <- 12
 ## rho   <- round(runif(1, 0.4, 0.6),1)
 ## Xmu   <- round(runif(p, -1, 1),1)
 ## beta_A <- matrix(rep(1,k), nrow=1) |> rbind(matrix(round(runif(p*k, -2, 2),1), nrow=p))
 ## beta_Y <- c(1, round(runif(p, -1, 1),1))
 ## hidunits = c(2,6)
 ## eps = c(120,150)
 ## penals = c(0.001,0.005)
 ## A_flavor = flavor_ops[[1]]; 
 ## Y_flavor = flavor_ops[[2]]; Y_fun = flavor_ops[[3]];
 ## 
 ## iter = 1; 
 ## source("YAX_funs.R")
 ## source("functions_k3plus.R")
 ## #source("functions_k3plus_dnn.R")
 ## verbose=TRUE

# ---------------------------

one_sim <- function(n, p, Xmu, beta_A, beta_Y, gamma, k,
                    A_flavor, Y_flavor, Y_fun, 
                    hidunits, eps, penals, verbose = FALSE, iter = 1, 
                    nntype = "1nn") {
  
  if(nntype == "dnn") { source("functions_k3plus_dnn.R") } 
  else { source("functions_k3plus.R") }
  
  X <- gen_X(n=n, p=p, rho=rho, mu=Xmu)
  A <- gen_A(X=X, beta=beta_A, flavor_A=A_flavor)
  Y <- gen_Y(X=X, A=A, beta_Y=beta_Y, gamma=gamma, flavor_Y=Y_flavor)
  dat <- cbind(Y,A,X) 
  stopifnot(Y>0)
  
  # print P(A=j)
  hist(Y)
  for(i in 1:k-1) {cat(paste0("\n  P(A=",i,")= ", mean(A==i) |> round(1)))}
  
  # print delta_ij
  get_true_diff <- function(x) {
    gamma_allvals <- c(0, gamma)
    i <- x[[2]];
    j <- x[[1]];
    gamma_i <- gamma_allvals[i+1]
    gamma_j <- gamma_allvals[j+1]
    d <- mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y) + gamma_i)) - 
      mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y) + gamma_j))
    cat(paste0("\n  True diff means ", j, i, " = ", round(d, 3)))
    d
  }
  true_diffs <- apply(as.matrix(combn(k,2)-1), 2, get_true_diff)
  
  # Estimate A (propensity model)
  tic("\nA model")
  fit_A_nn <- estimate_A_nn(X=X, dat=dat, k=k, 
                            hidunits=hidunits,
                            eps=eps, 
                            penals=penals,
                            verbose=verbose)
  fit_A_logit <- estimate_A_logit(X=X, dat=dat, k=k, verbose=verbose)
  toc()
  
  # Estimate Y (outcome model)
  tic("\nY model")
  fit_Y_nn <- estimate_Y_nn(dat, pscores_df=fit_A_nn$pscores, k=k,
                            hidunits=hidunits,
                            eps=eps, 
                            penals=penals, 
                            verbose=verbose)
  fit_Y_expo <- estimate_Y_expo(dat, pscores_df=fit_A_logit$pscores, k=k)
  toc()
  
  # Compute Vn
  X_new <- matrix(round(runif(5*dim(X)[2], -8, 8),1), 
                  nrow = 5, 
                  byrow = TRUE) |> as.data.frame()
  colnames(X_new) <- colnames(X)
  Vn_df <- 
    get_Vn(fit_Y_nn, X_new = X_new) |> 
    mutate(dataset = iter) |> 
    dplyr::select(dataset, everything())
  
  # Pack results into k rows
  get_naive_est <- function(x) {
    i <- x[[2]]
    j <- x[[1]]
    d <- mean(Y[A==i]) - mean(Y[A==j])
    cat(paste0("\n  Naive diff means ", j, i, " = ", round(d, 3)))
    d
  }
  naive_est <- apply(as.matrix(combn(k,2) - 1), 2, get_naive_est)
  nn_model_est <- sapply(fit_Y_nn, function(x) {x[[1]]}) |> data.frame() 
  logit_expo_est <- sapply(fit_Y_expo, function(x) {x[[1]]}) |> data.frame()
  
  my_k_rows <- 
    nn_model_est[1,] |> 
    rbind(logit_expo_est[1,]) |> 
    rbind(naive_est, true_diffs) |> 
    `row.names<-`(c("NN_est", "LogitExpo_est", "Naive_est","True_diff")) |>
    rownames_to_column("estimate") |>
    mutate(dataset = iter) |> 
    dplyr::select(dataset, everything())
  my_k_rows <- rbind(my_k_rows[4,], my_k_rows[-4,]) |> `row.names<-.data.frame`(1:4)
  
  list(my_k_rows = my_k_rows, Vn_df = Vn_df, Xnew_Vn = X_new)
  
}


