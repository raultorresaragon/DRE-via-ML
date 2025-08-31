# --------------------------------------------
# Author: Raul
# Date: 2025-08-31
# Script: pscores_models.R
# Note: This script fits models for estimating 
#       propensity score for k>2
#
#
# --------------------------------------------
source("A_nn_tuning.R")

# --------------
# Neural Network
# --------------
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
  
}


# -------------------
# Logit / multinomial
# -------------------
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
  
}