# --------------------------------------------
# Author: Raul
# Date: 2025-08-31
# Script: outcome_models.R
# Note: This script fits models for estimating 
#       outcome Y
#
#
# --------------------------------------------
source("Y_nn_tuning.R")

estimate_Y_ols <- function(dat, pscores_df, k) {
  
  Y <- dat$Y 
  
  # compute muhat_i vs muhat_j for i,j = combn(k,2)
  m <- combn(k, 2)-1 
  get_d_ij <- function(x) {
    i <- x[[2]] # this is g1 on iteration 1
    j <- x[[1]] # this is g0 on iteration 1
    pi_hat_i <- pscores_df[,i] |> as.vector()
    A_i <- dat |> mutate(A_i = case_when(A==i~1, A==j~0, TRUE ~99)) |> pull("A_i")
    delta_i <- as.numeric(A_i==1) # The 99s are retained so length = n
    delta_j <- as.numeric(A_i==0) # The 99s are retained so length = n
    
    g_i <- glm(as.formula("Y~.-A"), family = gaussian(link="identity"), data = dat[A_i==1,])
    ghat_i <- predict(g_i, newdata = dat, type = "response")
    g_j <- glm(as.formula("Y~.-A"), family = gaussian(link="identity"), data = dat[A_i==0,])
    ghat_j <- predict(g_j, newdata = dat, type = "response")
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  logit-expo est diff means [a=",j, " vs. a=",i,"]=", 
               round(d_ij$diff_means, 3)))
    names(d_ij) <- c(paste0("diff_means_",j,i), 
                     paste0("muhat_",i), 
                     paste0("muhat_",j),
                     paste0("diff_var_",j,i),
                     paste0("pval_",j,i))
    
    list(d_ij, g_i, g_j, ghat_j, ghat_i) #'j' is the smaller in "02"of A_02, e.g.
  }
  o <- apply(m, 2, get_d_ij)
  names(o) <- apply(m, 2, function(x) { 
    i <- x[[2]] 
    j <- x[[1]]
    paste0("A_",j,i)
  })
  o
}


estimate_Y_expo <- function(dat, pscores_df, k, link="log") {
  
  Y <- dat$Y 
  
  # compute muhat_i vs muhat_j for i,j = combn(k,2)
  m <- combn(k, 2)-1 
  get_d_ij <- function(x) {
    i <- x[[2]] # this is g1 on iteration 1
    j <- x[[1]] # this is g0 on iteration 1
    pi_hat_i <- pscores_df[,i] |> as.vector()
    A_i <- dat |> mutate(A_i = case_when(A==i~1, A==j~0, TRUE ~99)) |> pull("A_i")
    delta_i <- as.numeric(A_i==1) # The 99s are retained so length = n
    delta_j <- as.numeric(A_i==0) # The 99s are retained so length = n
    
    g_i <- glm(as.formula("Y~.-A"), family = gaussian(link=link), data = dat[A_i==1,])
    ghat_i <- predict(g_i, newdata = dat, type = "response")
    g_j <- glm(as.formula("Y~.-A"), family = gaussian(link=link), data = dat[A_i==0,])
    ghat_j <- predict(g_j, newdata = dat, type = "response")
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  logit-expo est diff means [a=",j, " vs. a=",i,"]=", 
               round(d_ij$diff_means, 3)))
    names(d_ij) <- c(paste0("diff_means_",j,i), 
                     paste0("muhat_",i), 
                     paste0("muhat_",j),
                     paste0("diff_var_",j,i),
                     paste0("pval_",j,i))
    
    list(d_ij, g_i, g_j, ghat_j, ghat_i) #'j' is the smaller in "02"of A_02, e.g.
  }
  o <- apply(m, 2, get_d_ij)
  names(o) <- apply(m, 2, function(x) { 
    i <- x[[2]] 
    j <- x[[1]]
    paste0("A_",j,i)
  })
  o
}

estimate_Y_nn <- function(dat, pscores_df, hidunits, eps, penals, k, verbose=FALSE) {
  
  Y <- dat$Y
  
  # compute muhat_i vs muhat_j for i,j = combn(k,2)
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
    ghat_i <- predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") #|> 
    #as.vector()
    
    
    g_j <- Y_model_nn(dat=dat[A_i==0, ] |> dplyr::select(-A), y_func = "Y~.",
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_j <- predict(g_j, new_data = dat %>% select(-Y, -A), type = "raw") #|> 
    #as.vector()
    
    
    d_ij <- get_diff(as.vector(ghat_i), delta_i, as.vector(ghat_j), delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  NN est diff means [a=",j, " vs. a=",i,"]=", 
               round(d_ij$diff_means, 3)))
    
    names(d_ij) <- c(paste0("diff_means_",j,i), 
                     paste0("muhat_",i), 
                     paste0("muhat_",j),
                     paste0("diff_var_",j,i),
                     paste0("pval_",j,i))
    
    list(d_ij, g_i, g_j, ghat_j, ghat_i) #'j' is the smaller in "02"of A_02, e.g.
  }
  o <- apply(m, 2, get_d_ij)
  names(o) <- apply(m, 2, function(x) { 
    i <- x[[2]] 
    j <- x[[1]]
    paste0("A_",j,i)
  })
  o
}
