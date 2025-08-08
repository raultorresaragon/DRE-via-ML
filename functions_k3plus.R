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
source("A_nn_tuning.R")
source("Y_nn_tuning.R")


# ----------------------------------
# Estimate A: propensity score model
# ----------------------------------
estimate_A_nn <- function(X, dat, k, hidunits, eps, penals, verbose=FALSE) {
  cat("\n   ...fitting 1 hidden-layer neural networks")
  H_nn <- A_model_nn(a_func = "A~.",
                     dat = dat[,colnames(dat)!="Y"],
                     hidunits=hidunits, eps=eps, penals=penals, 
                     verbose=verbose)
  pscores <- predict(H_nn, new_data=dat[,colnames(dat)!="A"], type="raw")
  if(k==2) { 
    pscores_names <- "pscores_1"
  } else {
    pscores_names <- paste0("pscores_",0:(k-1))
  }
  colnames(pscores) <- pscores_names
  list(pscores = pscores, H_nn = H_nn)
  
  
  ## H_nns <- list()
  ## pscores <- data.frame(prob = rep(0,n))
  ## for(i in 1:k-1) {
  ##   
  ##   dat_i <- dat |> mutate(A = if_else(A == i, 1, 0)) |> dplyr::select(-Y)
  ##   H_nn <- A_model_nn(a_func = "A~.",
  ##                      dat = dat_i,
  ##                      hidunits=hidunits, eps=eps, penals=penals, 
  ##                      verbose=verbose)
  ##   pscores_nn_i <- 
  ##     predict(H_nn, new_data = dat_i |> select(-A), type = "raw") |> 
  ##     as.vector()
  ##   
  ##   
  ##   pscores <- pscores |> mutate(prob = pscores_nn_i)  
  ##   colnames(pscores)[stringr::str_detect(colnames(pscores),"prob")]<-paste0("pscores_",i)
  ##   H_nns[i+1] <- list(H_nn) 
  ## }
  ## 
  ## myrowsums <- rowSums(pscores)
  ## pscores_df <- apply(pscores, 2, function(x) x/myrowsums) |> as.data.frame()
  ## list(pscores = pscores, H_nns = H_nns)
}

estimate_A_logit <- function(X, dat,k, verbose=FALSE){
  cat("\n   ...fitting logistic model")
  H_logit <- nnet::multinom(A~., data = dat[,colnames(dat)!="Y"])
  pscores <- predict(H_logit, type="probs") |> as.data.frame()
  if(k==2) { 
    pscores_names <- "pscores_1"
  } else {
    pscores_names <- paste0("pscores_",0:(k-1))
  }
  colnames(pscores) <- pscores_names
  list(pscores = pscores, H_logits = H_logit)
  
  
  ## H_logits <- list()
  ## pscores <- data.frame(prob = rep(0,n))
  ## 
  ## for(i in 1:k-1) {
  ##   
  ##   dat_i <- dat |> mutate(A = if_else(A == i, 1, 0)) |> dplyr::select(-Y)
  ##   H_logit <- glm(as.formula(A~.), family=binomial(link="logit"), data=dat_i)
  ##   pscores_logit_i <- predict(H_logit, type = "response", new_data = dat_i)
  ##   pscores <- pscores |> mutate(prob = pscores_logit_i)  
  ##   colnames(pscores)[stringr::str_detect(colnames(pscores),"prob")]<-paste0("pscores_",i)
  ##   H_logits[i+1] <- list(H_logit) 
  ## }
  ## myrowsums <- rowSums(pscores)
  ## pscores_df <- apply(pscores, 2, function(x) x/myrowsums) |> as.data.frame()
  ## list(pscores = pscores, H_logits = H_logits)
}


# --------------------------
# Estimate Y: outcome  model
# --------------------------

estimate_Y_expo <- function(dat, pscores_df, k) {
  
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
    
    g_i <- glm(as.formula("Y~.-A"), family = gaussian(link="log"), data = dat[A_i==1,])
    ghat_i <- predict(g_i, newdata = dat, type = "response")
    g_j <- glm(as.formula("Y~.-A"), family = gaussian(link="log"), data = dat[A_i==0,])
    ghat_j <- predict(g_j, newdata = dat, type = "response")
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y)
    
    cat(paste0("\n  logit-expo est diff means [a=",j, " vs. a=",i,"]=", 
               round(d_ij$diff_means, 3)))
    names(d_ij) <- c(paste0("diff_means_",j,i), paste0("muhat_",i), paste0("muhat_",j))
    
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
    
    names(d_ij) <- c(paste0("diff_means_",j,i), paste0("muhat_",i), paste0("muhat_",j))
    
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
