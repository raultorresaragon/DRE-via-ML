# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k3plus_aj_vs_notaj.R
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
  cat("\n   ...fitting 1 hidden-layer neural network")
  H_nn <- A_model_nn(a_func = "A~.",
                     dat = dat |> dplyr::select(-Y),
                     hidunits=hidunits, eps=eps, penals=penals, 
                     verbose=verbose)
  pscores_df <- 
    predict(H_nn, new_data = dat |> select(-A), type = "raw") |> 
    as.data.frame()
  list(pscores = pscores_df, H_nns = H_nns)
}


estimate_A_mlog <- function(X, dat, k, verbose=FALSE){
  cat("\n   ...fitting multinomial logit")
  H_mlog <- nnet::multinom(A~.-Y, data = dat)
  pscores_df <- predict(H_mlog, newdata = dat, type = "probs") |> as.data.frame()
  list(pscores = pscores_df, H_mlog = H_mlog)
}


# --------------------------
# Estimate Y: outcome  model
# --------------------------
estimate_Y_nn <- function(dat, pscores_df, hidunits, eps, penals, verbose=FALSE) {
  Y <- dat$Y
  A <- dat$A
  ghat <- tibble(index = 1:length(Y))
  g_s = list()
  d_ijs = list()
  for(i in sort(unique(A))) {
    delta_i <- as.numeric(A==i) 
    delta_j <- as.numeric(A!=i) 
    pi_hat_i <- pscores_df[,i+1] |> as.vector()
    
    g_i <- Y_model_nn(dat=dat[A==i,] |> dplyr::select(-A), y_func = "Y~.", 
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_i <- 
      predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") |> 
      as.vector()
    
    g_j <- Y_model_nn(dat=dat[A!=i, ] |> dplyr::select(-A), y_func = "Y~.",
                      hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
    ghat_j <- 
      predict(g_j, new_data = dat %>% select(-Y, -A), type = "raw") |> 
      as.vector()
    
    d_ij <- get_diff(ghat_i, delta_i, ghat_j, delta_j, pi_hat_i, Y) # this is not right in this case
    
    ghat <- cbind(ghat, ghat_i)
    g_s[i+1] <- list(g_i)
    d_ijs[i+1] <- list(d_ij)
  }
  ghat <- ghat[, 1:length(unique(A))+1]
  colnames(ghat) <- paste0("ghat", sort(unique(A)))
  names(g_s) <- paste0("g_", sort(unique(A)))
  names(d_ijs) <- paste0("g_", sort(unique(A)))
  list(ghat_df = ghat, g_s = g_s, d_ijs = d_ijs)
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